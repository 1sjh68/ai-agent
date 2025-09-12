# main.py

import os
import sys
import logging
import asyncio
from datetime import datetime
import json
import nest_asyncio
import re

# --- 路径修正代码 ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# --- 核心模块导入 ---
from config import Config
from services.vector_db import EmbeddingModel, VectorDBManager
from workflows.graph_runner import run_graph_workflow
from utils.text_processor import (
    consolidate_document_structure,
    final_post_processing,
    quality_check,
    truncate_text_for_context
)
from planning.outline import generate_document_outline_with_tools
from services.llm_interaction import call_ai
from utils.file_handler import parse_and_validate_paths
from planning.outline import generate_document_outline_with_tools
from services.llm_interaction import call_ai

# --- Section ID 注入函数 ---
def inject_section_ids_into_content(content: str, outline_data: dict) -> str:
    """
    将大纲中的章节ID精确注入到重构后的Markdown文本中，为后续的补丁应用做准备。
    """
    logging.info("--- 正在为重构后的内容注入Section ID ---")
    id_map = {}

    def clean_and_simplify_title(text: str) -> str:
        """一个更强大的辅助函数，用于深度清理和简化标题以进行匹配。"""
        # 移除Markdown标记，如`##`, `###`, `**`等
        text = re.sub(r'^[#\s]*', '', text).strip()
        text = re.sub(r'[#*`]', '', text).strip()
        # 移除数字和点号前缀，例如 "1. "
        text = re.sub(r'^\d+\.\s*', '', text).strip()
        # 将多个空格合并为一个
        text = re.sub(r'\s+', ' ', text)
        return text

    def build_id_map(chapters):
        """递归构建一个从清理后的标题到ID的映射。"""
        for chapter in chapters:
            title = chapter.get("title", "").strip()
            clean_title_key = clean_and_simplify_title(title)
            if clean_title_key:
                id_map[clean_title_key] = chapter.get("id")
            if "sections" in chapter and chapter["sections"]:
                build_id_map(chapter["sections"])

    if "outline" in outline_data:
        build_id_map(outline_data["outline"])

    if not id_map:
        logging.error("无法从大纲中构建标题到ID的映射。ID注入失败。")
        return content

    lines = content.split('\n')
    new_lines = []
    for line in lines:
        match = re.match(r'^(#+)\s*(.*)', line.strip())
        if match:
            heading_level_text = match.group(1)
            full_title_text = match.group(2).strip()
            
            cleaned_line_title = clean_and_simplify_title(full_title_text)
            
            section_id = id_map.get(cleaned_line_title)
            
            if section_id:
                # 注入ID，格式为HTML注释
                new_line = f"{heading_level_text} {full_title_text} "
                new_lines.append(new_line)
                logging.info(f"  - 成功注入ID for title: '{full_title_text}'")
            else:
                new_lines.append(line)
                logging.warning(f"  - 未能为标题 '{full_title_text}' (清理后为: '{cleaned_line_title}') 找到匹配的ID。")
        else:
            new_lines.append(line)
            
    return "\n".join(new_lines)


async def main():
    config = Config()
    config.setup_logging()

    final_answer_processed = ""

    config.user_problem = os.getenv("USER_PROBLEM", "请详细阐述一下人工智能的未来发展趋势。")
    external_files_str = os.getenv("EXTERNAL_FILES", "")
    config.external_data_files = parse_and_validate_paths(external_files_str)
    logging.info(f"任务问题 (前100字符): {config.user_problem[:100]}...")

    try:
        config._initialize_deepseek_client()
    except Exception as e:
        logging.critical(f"致命错误：无法初始化 DeepSeek 客户端: {e}. 程序即将退出。")
        sys.exit(1)

    vector_db_manager_instance = None
    try:
        embedding_model_instance = EmbeddingModel(config)
        if embedding_model_instance and embedding_model_instance.client:
            vector_db_manager_instance = VectorDBManager(config, embedding_model_instance)
    except Exception as e:
        logging.error(f"初始化嵌入或向量数据库管理器时出错: {e}。功能将受限。", exc_info=True)

    initial_solution_from_file = None
    # --- [核心修改] 升级为由全局大纲驱动的分块重写机制 ---
    if config.replay_from_file and os.path.exists(config.replay_from_file):
        logging.info(f"--- [智能重构模式] 已启用 (支持长文本分块) ---")
        logging.info(f"--- 正在从 '{config.replay_from_file}' 加载内容进行结构优化 ---")
        try:
            with open(config.replay_from_file, "r", encoding="utf-8") as f:
                original_content = f.read()

            if not original_content.strip():
                raise ValueError("输入文件内容为空。")

            # 步骤1: AI规划师 - 为全文生成一个全局的、理想的大纲
            logging.info("步骤1: 正在为全文生成一个全局的结构化大纲...")
            outline_prompt = f"请仔细分析以下提供的完整文本文档，并为其创建一个逻辑清晰、结构合理的文档大纲。大纲应该能最好地组织和呈现文中的信息。原始文本如下：\n\n---\n{truncate_text_for_context(config, original_content, 15000)}\n---"
            outline_data = generate_document_outline_with_tools(config, outline_prompt)
            
            if not outline_data or "outline" not in outline_data:
                raise RuntimeError("未能为输入文本生成有效大纲，操作终止。")
            
            logging.info(f"成功生成全局大纲，共 {len(outline_data.get('outline', []))} 个主章节。")

            # 步骤2: AI编辑 - 遍历新大纲，逐个章节进行内容填充
            logging.info("步骤2: 开始根据新大纲，逐个章节进行约束性重写...")
            
            truncated_original_content = truncate_text_for_context(config, original_content, 30000, "middle")
            restructured_parts = [f"# {outline_data.get('title', '重构后的文档')}"]

            chapters_to_process = outline_data.get("outline", [])
            for i, chapter in enumerate(chapters_to_process):
                chapter_title = chapter.get("title", f"未命名章节 {i+1}")
                logging.info(f"  -> 正在重写章节 {i+1}/{len(chapters_to_process)}: '{chapter_title}'")
                
                rewrite_prompt = f"""
                你是一位顶级的技术编辑，你的任务是严格按照给定的新大纲，重组和格式化提供的原始文本。

                **你的行为必须遵循以下铁律：**
                1.  **唯一信源**：你只能使用【原始文本】中的句子和信息，绝不允许自己创造、杜撰或引入任何外部信息。
                2.  **内容完整**：不能遗漏【原始文本】中的任何关键论点、数据或段落。你的工作是重新组织，而不是删减。
                3.  **聚焦任务**：在本次任务中，你只需要撰写标题为 **“{chapter_title}”** 的这一个章节的内容。
                4.  **格式规范**：请使用标准的Markdown格式进行输出，并以正确的标题级别（`## {chapter_title}`）开始。

                **【完整的文档大纲 (供你理解整体结构)】**
                ```json
                {json.dumps(outline_data, ensure_ascii=False, indent=2)}
                ```

                **【原始文本 (你的唯一内容来源，可能已被截断)】**
                ```text
                {truncated_original_content}
                ```

                现在，请开始你的工作，仅输出 **“{chapter_title}”** 章节的完整内容。
                """
                messages = [{"role": "user", "content": rewrite_prompt}]
                
                chapter_content = call_ai(
                    config, 
                    config.main_ai_model_heavy, 
                    messages,
                    max_tokens_output=4096 
                )

                if "AI模型调用失败" in chapter_content or not chapter_content.strip():
                    logging.warning(f"    - 章节 '{chapter_title}' 重写失败，将跳过。")
                    chapter_content = f"## {chapter_title}\n\n[内容重构失败]\n\n"
                
                restructured_parts.append(chapter_content)

            # 步骤3: [关键修复] 将ID注入到重构后的内容中
            logging.info("步骤3: 正在将章节ID注入到重构后的内容中，为后续优化做准备...")
            restructured_content_without_ids = "\n\n".join(restructured_parts)
            initial_solution_from_file = inject_section_ids_into_content(
                restructured_content_without_ids, 
                outline_data
            )

            logging.info(f"--- 智能重构与ID注入完成，内容将作为第0轮初稿进入优化流程 ---")

        except Exception as e:
            logging.critical(f"[智能重构模式] 预处理文件时发生错误: {e}", exc_info=True)
            return
    else:
        logging.info("--- [标准模式] 启动完整AI内容创作框架 ---")

    # --- 后续主流程保持不变 ---
    raw_final_result = "错误：由于未知原因，工作流未能成功运行。"
    try:
        raw_final_result = await run_graph_workflow(
            config,
            vector_db_manager_instance,
            initial_solution=initial_solution_from_file
        )

        if raw_final_result and not raw_final_result.startswith("错误："):
            logging.info("\n--- 工作流完成，正在进行最终的后处理、评估与保存 ---")

            structured_answer = consolidate_document_structure(raw_final_result)
            final_answer_processed = final_post_processing(structured_answer)

            logging.info("\n--- 最终产出质量评估报告 ---")
            quality_report = quality_check(config, final_answer_processed)
            logging.info(quality_report)
        else:
            final_answer_processed = raw_final_result

    except Exception as e:
        logging.critical(f"主工作流程 'generate_extended_content_workflow' 发生未捕获的严重异常: {e}", exc_info=True)
        final_answer_processed = f"错误：工作流程因严重失败而终止。详情请查看日志。 {e}"

    if final_answer_processed and not final_answer_processed.startswith("错误："):
        try:
            prefix = "restructured_output" if config.replay_from_file else "final_solution"
            output_filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            if config.session_dir and os.path.isdir(config.session_dir):
                output_filepath = os.path.join(config.session_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(final_answer_processed)
                logging.info(f"🎉 最终报告已成功保存至: {output_filepath}")
            else:
                logging.error("会话目录不存在，无法保存最终文件。")
        except Exception as e:
            logging.error(f"保存最终报告时发生错误: {e}")
    else:
        logging.error(f"脚本执行结束，但最终结果是错误或为空: {final_answer_processed}")


if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户手动中断。")
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.critical(f"在启动或运行主异步任务时发生致命错误: {e}", exc_info=True)