# core/context_manager.py

import logging
import json
import collections
import re
# --- 新增了必要的导入 ---
import os
import uuid
import chromadb
from typing import Optional

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai
from utils.text_processor import truncate_text_for_context, chunk_document_for_rag
# 导入 VectorDBManager 和 EmbeddingModel 以便在此模块中使用
from services.vector_db import VectorDBManager, EmbeddingModel

class ContextManager:
    """
    管理长文档生成过程中的上下文信息，防止超出模型限制。
    新版架构采用RAG模型，通过对外部文档建立索引并按需检索，来处理长文本。
    """
    def __init__(self, config: Config, style_guide: str, outline: dict, external_data: str, embedding_model_instance: Optional[EmbeddingModel] = None):
        self.config = config
        self.style_guide = style_guide
        self.outline = outline
        
        # --- RAG 相关初始化 ---
        self.rag_db_manager = None
        if external_data and embedding_model_instance:
            # 传递一个已经存在的 embedding_model 实例
            self._initialize_rag_index(external_data, embedding_model_instance)

        # 存储已完成主章节的摘要: {chapter_title: summary_text}
        self.chapter_summaries = {}
        # 存储已生成子章节的完整内容: {chapter_title: {subsection_title: content}}
        self.generated_subsection_content = collections.defaultdict(dict)
        # 存储已完成主章节的完整内容: {chapter_title: full_content}
        self.generated_chapter_content = {}

    def _initialize_rag_index(self, external_data: str, embedding_model: EmbeddingModel):
        """
        [新函数] 对外部参考数据进行分块、向量化并建立索引。
        """
        logging.info(f"  上下文管理器：正在为 {len(external_data)} 字符的外部数据建立RAG索引...")

        # --- [核心修复] 修正集合名称生成逻辑 ---
        session_id_part = self.config.session_dir.split('_')[-1]
        # 确保 session_id_part 不为空且有效，如果为空则生成一个随机值
        if not session_id_part or not session_id_part.isalnum():
            session_id_part = str(uuid.uuid4().hex[:8])
        doc_id = f"session_doc_{session_id_part}"

        # 调用更新后的、更健壮的分块函数
        pdf_chunks, metadatas = chunk_document_for_rag(
            self.config, external_data, doc_id
        )


    def _retrieve_rag_context(self, query_text: str, n_results: int = 3) -> str:
        """
        [新函数] 根据查询文本，从已建立的RAG索引中检索相关上下文。
        """
        if not self.rag_db_manager:
            return ""

        relevant_chunks = self.rag_db_manager.retrieve_experience(query_text, n_results=n_results)
        
        if not relevant_chunks:
            return ""

        retrieved_context = "\n\n--- 从参考PDF中检索到的高度相关原文片段 ---\n"
        for i, chunk in enumerate(relevant_chunks):
            retrieved_context += f"\n[相关原文片段 {i+1}]\n"
            retrieved_context += chunk.get('document', '内容缺失') + "\n"
        retrieved_context += "--- 原文片段结束 ---\n\n"
        
        return retrieved_context

    def record_completed_subsection(self, chapter_title: str, subsection_title: str, subsection_content: str):
        """记录一个已完成子章节的完整内容。"""
        logging.info(f"  上下文管理器：正在记录章节“{chapter_title}”中子章节“{subsection_title}”的内容。")
        self.generated_subsection_content[chapter_title][subsection_title] = subsection_content

    def update_completed_chapter_content(self, chapter_title: str, full_chapter_content: str):
        """存储一个已完成主章节的完整内容，并为其生成摘要。"""
        logging.info(f"  上下文管理器：正在存储章节“{chapter_title}”的完整内容并生成摘要。")
        self.generated_chapter_content[chapter_title] = full_chapter_content
        
        content_for_summary = truncate_text_for_context(self.config, full_chapter_content, 6000)
        summary_prompt_text = (
            f"请为以下标题为“{chapter_title}”的章节提供一个简洁的摘要（约200-300字）。"
            "重点关注本章内的关键论点、发现和结论。"
            "此摘要将用作撰写后续章节的上下文，因此需要信息丰富且简短。\n\n"
            f"--- 章节内容开始 ---\n{content_for_summary}\n--- 章节内容结束 ---"
        )
        
        summary = call_ai(
            self.config, 
            self.config.summary_model_name,
            [{"role": "user", "content": summary_prompt_text}],
            max_tokens_output=512
        )
        
        if "AI模型调用失败" in summary or not summary.strip():
            logging.warning(f"  上下文管理器：为章节“{chapter_title}”生成摘要失败。AI 响应: {summary}")
            self.chapter_summaries[chapter_title] = "本章节摘要生成失败。"
        else:
            self.chapter_summaries[chapter_title] = summary.strip()
            logging.info(f"  上下文管理器：章节“{chapter_title}”的摘要已创建并存储 ({len(summary)} 字符)。")

    def get_context_for_subsection(self, current_chapter_title: str, current_subsection_index: int) -> str:
        """为生成特定子章节组装上下文包，实现章节内的精确感知。"""
        logging.info(f"  上下文管理器：正在为章节“{current_chapter_title}”，子章节索引 {current_subsection_index} 组装上下文。")

        full_doc_outline_str = json.dumps(self.outline, ensure_ascii=False, indent=2)
        style_guide_str = self.style_guide if self.style_guide else "无特定风格指南。"
        
        outline_chapters = self.outline.get('outline', [])
        current_chapter_obj = next((ch for ch in outline_chapters if ch.get('title') == current_chapter_title), None)
        
        if not current_chapter_obj:
            logging.error(f"  上下文管理器：在提纲中未找到章节“{current_chapter_title}”。")
            return "[错误：无法定位当前主章节信息]"
        
        main_chapter_index = outline_chapters.index(current_chapter_obj)

        # RAG 检索步骤
        subsections = current_chapter_obj.get("sections", [])
        rag_context = ""
        if current_subsection_index < len(subsections):
            current_subsection_obj = subsections[current_subsection_index]
            query_text = f"{current_chapter_title}: {current_subsection_obj.get('title', '')} - {current_subsection_obj.get('description', '')}"
            rag_context = self._retrieve_rag_context(query_text)
        
        prev_main_chapter_full_text_str = "这是报告的第一个主章节。"
        if main_chapter_index > 0:
            prev_main_chapter_title = outline_chapters[main_chapter_index - 1].get("title")
            if prev_main_chapter_title:
                content = self.generated_chapter_content.get(prev_main_chapter_title, f"前一主章节“{prev_main_chapter_title}”的内容尚未记录。")
                prev_main_chapter_full_text_str = truncate_text_for_context(self.config, content, 4000, "tail")

        content_of_chapter_N_so_far = []
        if current_subsection_index > 0:
            for sub_idx_prev in range(current_subsection_index):
                if sub_idx_prev < len(subsections):
                    prev_sub_title_iter = subsections[sub_idx_prev].get("title")
                    if prev_sub_title_iter:
                        sub_content = self.generated_subsection_content.get(current_chapter_title, {}).get(prev_sub_title_iter)
                        if sub_content:
                            content_of_chapter_N_so_far.append(f"--- 内容来自：{prev_sub_title_iter} ---\n{sub_content}")
        
        chapter_N_content_str = "\n\n".join(content_of_chapter_N_so_far)
        if not chapter_N_content_str:
            chapter_N_content_str = "这是本章的第一个子章节，之前没有内容。"
        else:
            chapter_N_content_str = f"--- 本章《{current_chapter_title}》已生成内容开始 ---\n{chapter_N_content_str}\n--- 本章《{current_chapter_title}》已生成内容结束 ---"
        chapter_N_content_str = truncate_text_for_context(self.config, chapter_N_content_str, 4000, "tail")

        next_context_element_str = "这是报告的最后一个部分。"
        if current_subsection_index < len(subsections) - 1:
            next_subsection_obj = subsections[current_subsection_index + 1]
            next_subsection_title = next_subsection_obj.get("title", "未命名子章节")
            next_subsection_desc = next_subsection_obj.get("description", "无详细描述。")
            next_context_element_str = f"下一个子章节《{next_subsection_title}》计划阐述：{next_subsection_desc}"
        elif main_chapter_index < len(outline_chapters) - 1:
            next_main_chapter_title = outline_chapters[main_chapter_index + 1].get("title")
            if next_main_chapter_title:
                next_main_chapter_desc = outline_chapters[main_chapter_index + 1].get("description", "无描述")
                next_context_element_str = f"完成本章节后，下一个主章节是《{next_main_chapter_title}》，其核心目标是：{next_main_chapter_desc}"
        
        context_packet = f"""
[报告的完整大纲]
{full_doc_outline_str}

[风格与声音指南]
{style_guide_str}
{rag_context}
[上一主章节《{outline_chapters[main_chapter_index - 1].get("title") if main_chapter_index > 0 else "N/A"}》的核心内容回顾]
{prev_main_chapter_full_text_str}

[当前主章节《{current_chapter_title}》的核心目标]
{current_chapter_obj.get('description', '无详细描述。')}

[当前主章节《{current_chapter_title}》已生成的小节内容（你正在续写）]
{chapter_N_content_str}

[为后续内容的铺垫信息]
{next_context_element_str}
"""
        logging.info(f"  上下文管理器：子章节上下文包已创建。大约长度 {len(context_packet)} 字符。")
        return context_packet

    def get_context_for_standalone_chapter(self, chapter_title: str) -> str:
        """为生成一个没有子章节、打算一次性写完的主章节组装上下文包。"""
        logging.info(f"  上下文管理器：正在为独立章节“{chapter_title}”组装上下文。")
        
        outline_chapters = self.outline.get('outline', [])
        current_chapter_obj = next((ch for ch in outline_chapters if ch.get('title') == chapter_title), None)
        
        if not current_chapter_obj:
            logging.error(f"  上下文管理器：在提纲中未找到独立章节“{chapter_title}”。")
            return "[错误：无法定位当前独立章节信息]"

        # RAG 检索步骤
        query_text = f"{chapter_title}: {current_chapter_obj.get('description', '')}"
        rag_context = self._retrieve_rag_context(query_text)

        full_doc_outline_str = json.dumps(self.outline, ensure_ascii=False, indent=2)
        style_guide_str = self.style_guide if self.style_guide else "无特定风格指南。"
        all_chapter_titles = [ch.get('title', '未命名章节') for ch in self.outline.get('outline', [])]
        other_chapter_titles_list = [t for t in all_chapter_titles if t != chapter_title]
        other_chapter_titles_str = "\n - ".join(other_chapter_titles_list)
        if not other_chapter_titles_str: other_chapter_titles_str = "无其他章节。"
        
        main_chapter_index = outline_chapters.index(current_chapter_obj)
        current_chapter_desc_str = current_chapter_obj.get('description', '无详细描述。')
        
        prev_main_chapter_full_text_str = "这是报告的第一个主章节。"
        if main_chapter_index > 0:
            prev_main_chapter_title = outline_chapters[main_chapter_index - 1].get("title")
            if prev_main_chapter_title:
                content = self.generated_chapter_content.get(prev_main_chapter_title, f"前一主章节“{prev_main_chapter_title}”的内容尚未记录。")
                prev_main_chapter_full_text_str = truncate_text_for_context(self.config, content, 6000, "tail")

        summary_N_plus_1_str = "这是报告的最后一个主章节。"
        if main_chapter_index < len(outline_chapters) - 1:
            next_main_chapter_title = outline_chapters[main_chapter_index + 1].get("title")
            if next_main_chapter_title:
                next_main_chapter_desc = outline_chapters[main_chapter_index + 1].get("description", f"下一主章节“{next_main_chapter_title}”的描述未定义。")
                summary_N_plus_1_str = f"下一主章节《{next_main_chapter_title}》计划阐述：{next_main_chapter_desc}"
        
        context_packet = f"""
[报告的完整大纲]
{full_doc_outline_str}

[风格与声音指南]
{style_guide_str}
{rag_context}
[其他章节标题列表 (供结构参考)]
 - {other_chapter_titles_str}

[【章节 N-1】上一主章节《{outline_chapters[main_chapter_index - 1].get("title") if main_chapter_index > 0 else "N/A"}》的完整内容回顾]
--- 前一章节内容开始 ---
{prev_main_chapter_full_text_str}
--- 前一章节内容结束 ---

[【章节 N】当前主章节《{chapter_title}》的核心目标与描述]
{current_chapter_desc_str}
重要提示: 你将一次性完成本章节的全部内容。

[【章节 N+1】下一主章节《{outline_chapters[main_chapter_index + 1].get("title") if main_chapter_index < len(outline_chapters) - 1 else "N/A"}》的核心目标]
{summary_N_plus_1_str}
"""
        logging.info(f"  上下文管理器：独立章节上下文包已创建。大约长度 {len(context_packet)} 字符。")
        return context_packet

    def get_context_for_chapter_critique(self, chapter_title_being_critiqued: str, full_document_text: str) -> str:
        """为评论/修补特定章节 (N) 组装上下文包。"""
        logging.info(f"  上下文管理器：正在为评论章节“{chapter_title_being_critiqued}”组装精确上下文。")

        # RAG 检索步骤
        query_text = f"Reviewing section: {chapter_title_being_critiqued}"
        rag_context = self._retrieve_rag_context(query_text, n_results=5)

        full_doc_outline_str = json.dumps(self.outline, ensure_ascii=False, indent=2)
        style_guide_str = self.style_guide if self.style_guide else "无特定风格指南。"
        
        outline_chapters = self.outline.get('outline', [])
        chapter_N_obj, chapter_N_minus_1_obj, chapter_N_plus_1_obj = None, None, None
        
        try:
            chapter_N_index = next(i for i, ch in enumerate(outline_chapters) if ch.get('title') == chapter_title_being_critiqued)
            chapter_N_obj = outline_chapters[chapter_N_index]
            if chapter_N_index > 0:
                chapter_N_minus_1_obj = outline_chapters[chapter_N_index - 1]
            if chapter_N_index < len(outline_chapters) - 1:
                chapter_N_plus_1_obj = outline_chapters[chapter_N_index + 1]
        except StopIteration:
            logging.error(f"  上下文管理器：在提纲中未找到用于评论上下文的章节“{chapter_title_being_critiqued}”。")
            return "[错误：无法定位被评审章节信息]"

        text_N_minus_1 = "这是报告的第一个主章节。"
        title_N_minus_1 = "N/A"
        if chapter_N_minus_1_obj:
            title_N_minus_1 = chapter_N_minus_1_obj.get("title", "未知前文章节")
            content = self.generated_chapter_content.get(title_N_minus_1, f"章节《{title_N_minus_1}》的内容尚未记录。")
            text_N_minus_1 = truncate_text_for_context(self.config, content, 6000, "middle")

        text_N = f"未能从文档中提取章节《{chapter_title_being_critiqued}》的完整内容。"
        escaped_title_N = re.escape(chapter_title_being_critiqued)
        pattern_N = re.compile(rf"^(##\s*{escaped_title_N}.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
        match_N = pattern_N.search(full_document_text)
        if match_N:
            text_N = match_N.group(1).strip()
        else:
            logging.warning(f"无法使用正则表达式从完整文档中提取章节 N：“{chapter_title_being_critiqued}”的内容。")

        summary_N_plus_1 = "这是报告的最后一个主章节。"
        title_N_plus_1 = "N/A"
        if chapter_N_plus_1_obj:
            title_N_plus_1 = chapter_N_plus_1_obj.get("title", "未知后续章节")
            summary_N_plus_1 = self.chapter_summaries.get(title_N_plus_1, f"章节《{title_N_plus_1}》的摘要尚未生成。")
        
        context_packet = f"""
[报告的完整大纲]
{full_doc_outline_str}

[风格与声音指南]
{style_guide_str}
{rag_context}
[【章节 N-1】《{title_N_minus_1}》的全文回顾]
--- 内容开始 ---
{text_N_minus_1}
--- 内容结束 ---

[【章节 N】《{chapter_title_being_critiqued}》的当前全文 (此为重点评审/修改对象)]
--- 内容开始 ---
{text_N}
--- 内容结束 ---

[【章节 N+1】《{title_N_plus_1}》的核心摘要]
--- 内容开始 ---
{summary_N_plus_1}
--- 内容结束 ---
"""
        # --- [核心修复] 对最终的上下文包进行安全截断，防止超出API限制 ---
        final_context_packet = truncate_text_for_context(
            self.config,
            context_packet,
            self.config.max_context_for_long_text_review_tokens,
            "middle" # 从中间截断以保留开头和结尾的关键信息
        )
        
        logging.info(f"  上下文管理器：章节评论的精确上下文已创建。原始长度 {len(context_packet)}，截断后 {len(final_context_packet)} 字符。")
        return final_context_packet
