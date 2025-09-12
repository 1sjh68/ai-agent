# workflows/graph_nodes.py

import os
import sys
import uuid # 导入 uuid 以备 context_manager 使用

# --- [核心修复] 动态添加项目根目录到Python路径 ---
# 这段代码确保无论从哪里运行此脚本，它都能找到顶层的'config'等模块。
try:
    # __file__ 是当前文件的路径
    # os.path.abspath(__file__) 获取绝对路径
    # os.path.dirname() 获取目录
    # 我们需要向上两级目录来到达项目根目录 (workflows -> Deep Research)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    # 在某些环境下 __file__ 可能未定义, 使用 getcwd() 作为备用方案
    # 假设当前工作目录是 workflows
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# --- [修复结束] ---


import json
import logging
import time
from typing import List, Dict, Optional

from pydantic import ValidationError

from config import Config
from core.context_manager import ContextManager
from core.patch_manager import apply_fine_grained_edits
from planning.outline import generate_document_outline_with_tools, allocate_content_lengths
from planning.tool_definitions import FineGrainedPatchList
from services.llm_interaction import call_ai
from services.web_research import run_research_cycle_async
from services.vector_db import VectorDBManager, EmbeddingModel
from utils.text_processor import extract_knowledge_gaps, truncate_text_for_context
from workflows.graph_state import GraphState
from workflows.sub_workflows.drafting import generate_section_content
from workflows.sub_workflows.memory import accumulate_experience
from workflows.sub_workflows.planning import generate_style_guide
from workflows.sub_workflows.polishing import perform_final_polish

# --- Structured Logging Helper ---
def log_progress(node_name: str, step_name: str, state: dict):
    """Logs a structured progress message for the frontend."""
    refinement_count = state.get("refinement_count", 0)
    config = state.get("config")
    max_iterations = getattr(config, 'max_iterations', 4) if config else 4
    current_iteration_display = refinement_count + 1

    progress_data = {
        "type": "progress",
        "node": node_name,
        "step_name": step_name,
        "iteration": current_iteration_display,
        "total_iterations": max_iterations,
        "timestamp": time.time(),
        "message": f"Step: {step_name} (Iteration {current_iteration_display}/{max_iterations})"
    }
    logging.info(json.dumps(progress_data))

# --- Graph Node Implementations ---

async def style_guide_node(state: GraphState) -> Dict:
    """Node to generate the style guide."""
    log_progress("style_guide_node", "Generating Style Guide", state)
    config = state['config']
    style_guide = generate_style_guide(config)
    return {"style_guide": style_guide}


async def plan_node(state: GraphState) -> Dict:
    """Node to generate the document outline."""
    log_progress("plan_node", "Generating Outline", state)
    config = state['config']
    topic = state['topic']

    raw_outline = generate_document_outline_with_tools(config, topic)
    if not raw_outline or "outline" not in raw_outline:
        raise ValueError("Failed to generate a valid document outline.")

    document_outline_data = allocate_content_lengths(
        config, raw_outline, config.initial_solution_target_chars
    )

    all_chapter_titles = [ch.get('title', 'Untitled') for ch in document_outline_data.get("outline", [])]

    return {
        "outline": document_outline_data,
        "all_chapter_titles": all_chapter_titles,
    }


async def draft_node(state: GraphState) -> Dict:
    """Node to generate the initial draft based on the outline."""
    log_progress("draft_node", "Generating Initial Draft", state)
    config = state['config']
    outline = state['outline']
    style_guide = state['style_guide']
    external_data = state.get('external_data', '')

    embedding_model = getattr(config, 'embedding_model_instance', None)
    context_manager = ContextManager(config, style_guide, outline, external_data, embedding_model)

    system_prompt_base = """You are a top-tier domain expert and academic writer, known for your ability to transform complex concepts into clear, rigorous, and in-depth academic discourse. Your task is to write a specific section of a comprehensive report based on the provided outline and rich contextual information.
**Your writing must adhere to the following core principles:**
1.  **Fidelity to Context**: The user prompt you receive will contain all necessary background information. You must treat this information as "absolute truth" and write on this basis.
2.  **Academic Rigor**: Maintain an objective, neutral, and professional academic tone.
3.  **Task Focus**: Your task is to write **only the currently requested section**. Start writing the body text directly, without repeating the chapter title.
4.  **Depth and Detail**: Elaborate on ideas as deeply as possible, providing detailed analysis and explanation.
"""

    assembled_parts = [f"# {outline.get('title', 'Untitled Document')}\n\n"]
    for chapter in outline.get("outline", []):
        context_for_chapter = context_manager.get_context_for_standalone_chapter(chapter.get("title"))
        chapter_content = generate_section_content(
            config,
            section_data=chapter,
            system_prompt=system_prompt_base,
            model_name=config.main_ai_model,
            overall_context=context_for_chapter,
            is_subsection=False
        )
        assembled_parts.append(chapter_content)
        context_manager.update_completed_chapter_content(chapter.get("title"), chapter_content)

    draft_content = "".join(assembled_parts)

    return {"draft_content": draft_content}


async def critique_node(state: GraphState) -> Dict:
    """Node to critique the current draft and identify knowledge gaps."""
    log_progress("critique_node", "Critiquing Draft", state)
    config = state['config']
    draft_content = state['draft_content']
    topic = state['topic']

    solution_for_critic = truncate_text_for_context(
        config, draft_content, config.max_context_for_long_text_review_tokens
    )

    secondary_ai_critique_prompt = """You are a top academic journal editor known for being extremely critical and having high standards. Your sole purpose is to identify all potential issues in the draft, no matter how small.

Your review must include two parts:
**Part 1: Overall Critique** (Assess core argument, structure, depth, and language)
**Part 2: Knowledge Gaps** (Under a '### KNOWLEDGE GAPS' heading, list all points needing external information to be more credible.)
"""

    critic_prompt = f"Original problem:\n---\n{topic}\n---\nSolution to be reviewed:\n---\n{solution_for_critic}\n---\nPlease provide your review:"

    feedback = call_ai(
        config,
        config.secondary_ai_model,
        [
            {"role": "system", "content": secondary_ai_critique_prompt},
            {"role": "user", "content": critic_prompt}
        ],
        temperature=config.temperature_factual
    )

    knowledge_gaps = extract_knowledge_gaps(feedback)

    logging.info(f"Critique generated. Found {len(knowledge_gaps)} knowledge gaps.")

    return {"feedback": feedback, "research_query": knowledge_gaps}


async def research_node(state: GraphState) -> Dict:
    """Node to perform web research based on knowledge gaps."""
    log_progress("research_node", "Performing Research", state)
    config = state['config']
    knowledge_gaps = state['research_query']
    draft_content = state['draft_content']

    if not knowledge_gaps:
        return {"research_results": ""}

    research_brief = await run_research_cycle_async(config, knowledge_gaps, draft_content)

    return {"research_results": research_brief or ""}


async def refine_node(state: GraphState) -> Dict:
    """Node to generate patches based on feedback and research."""
    log_progress("refine_node", "Generating Refinements", state)
    config = state['config']
    feedback = state['feedback']
    research_results = state.get('research_results', '')
    external_data = state.get('external_data', '')
    draft_content = state['draft_content']
    outline = state['outline']
    style_guide = state['style_guide']
    topic = state['topic']

    combined_context_data = f"{external_data}\n{research_results}"
    embedding_model = getattr(config, 'embedding_model_instance', None)
    context_manager = ContextManager(config, style_guide, outline, combined_context_data, embedding_model)

    first_chapter_title_in_feedback = None
    if outline and outline.get("outline"):
        for chap in outline["outline"]:
            if chap.get("title", "a_very_unlikely_string") in feedback:
                first_chapter_title_in_feedback = chap.get("title")
                break
        if not first_chapter_title_in_feedback:
            first_chapter_title_in_feedback = outline["outline"][0].get("title")

    precise_context_for_patcher = ""
    if first_chapter_title_in_feedback:
        precise_context_for_patcher = context_manager.get_context_for_chapter_critique(
            first_chapter_title_in_feedback, draft_content
        )

    patch_tool_sys_prompt = """You are a meticulous revision editor. Your task is to analyze reviewer feedback and create sentence-level revisions for the document. Your output must be a JSON object conforming to the `FineGrainedPatchList` model.
**Instructions:**
1.  **Sentence-Level Edits**: All revisions must be at the sentence level.
2.  **Match `target_id`**: You must find the `target_id` from the context and copy it exactly.
3.  **Minimal Changes**: Only modify sentences that are explicitly problematic according to the feedback.
4.  **Strict JSON Format**: Your output must be a valid JSON object.
"""

    patch_user_prompt = f"""[Original Problem]\n{topic}\n
[Latest Research Brief]\n{research_results or "None"}\n
[Revision Feedback]\n---\n{feedback}\n---
[Target Chapter Context for Revision]\n---\n{precise_context_for_patcher}\n---
Now, generate a patch list to address the issues based on the feedback.
"""

    patches = []
    try:
        json_response_text = call_ai(
            config,
            config.patcher_model_name,
            messages=[
                {"role": "system", "content": patch_tool_sys_prompt},
                {"role": "user", "content": patch_user_prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=config.temperature_factual,
            max_tokens_output=8192
        )
        if json_response_text and not json_response_text.isspace():
            edit_obj = FineGrainedPatchList.model_validate_json(json_response_text)
            patches = edit_obj.patches
            logging.info(f"Successfully generated and validated {len(patches)} patches.")
    except (ValidationError, json.JSONDecodeError) as e:
        logging.error(f"Invalid JSON from Patcher AI: {e}")
    except Exception as e:
        logging.error(f"Error generating patches: {e}")

    return {"patches": patches}


async def apply_patches_node(state: GraphState) -> Dict:
    """Node to apply the generated patches to the draft."""
    log_progress("apply_patches_node", "Applying Patches", state)
    patches = state['patches']
    draft_content = state['draft_content']

    if not patches:
        logging.info("No patches to apply.")
        new_content = draft_content
    else:
        new_content = apply_fine_grained_edits(draft_content, patches)

    return {
        "draft_content": new_content,
        "refinement_count": state.get('refinement_count', 0) + 1
    }


async def polish_node(state: GraphState) -> Dict:
    """Node to perform a final polish on the document."""
    log_progress("polish_node", "Polishing Final Document", state)
    config = state['config']
    draft_content = state['draft_content']
    style_guide = state['style_guide']

    polished_solution = perform_final_polish(config, draft_content, style_guide)

    return {"final_solution": polished_solution}


async def memory_node(state: GraphState):
    """Node to save the final work as "experience" for future runs."""
    log_progress("memory_node", "Saving Experience", state)
    config = state['config']
    vector_db_manager = getattr(config, 'vector_db_manager', None)

    if vector_db_manager:
        accumulate_experience(
            config,
            vector_db_manager,
            state['topic'],
            state['final_solution'],
            [state['feedback']],
            state.get('patches', []),
            [state.get('research_results', '')]
        )
    else:
        logging.warning("Vector DB Manager not found, skipping experience accumulation.")

    return {}