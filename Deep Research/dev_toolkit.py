# dev_toolkit.py

import os
import sys
import logging
import asyncio
import argparse
from datetime import datetime

# --- 路径修正代码 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 核心模块导入 ---
from config import Config
from services.vector_db import EmbeddingModel, VectorDBManager
from utils.file_handler import load_external_data, parse_and_validate_paths

# [核心修正] 从正确的位置导入所有需要的函数
from utils.text_processor import (
    chunk_document_for_rag,
    consolidate_document_structure,
    final_post_processing,
    quality_check
)
from workflows.sub_workflows.polishing import perform_final_polish
from workflows.sub_workflows.planning import generate_style_guide

from planning.outline import generate_document_outline_with_tools
from services.web_research import run_research_cycle_async


# --- 初始化 ---
config = Config()
config.setup_logging()

# --- 异步函数运行器 ---
def run_async(coro):
    """
    一个简单的包装器，用于在同步代码中运行异步函数。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # nest_asyncio.apply() might be needed in some environments like Jupyter
    return loop.run_until_complete(coro)

# --- 功能实现 ---

def seed_knowledge_base(args):
    """功能一：将个人数据存入向量数据库"""
    logging.info("--- 开发者模式：启动知识库构建工具 ---")
    valid_files = parse_and_validate_paths(args.files)
    if not valid_files:
        logging.error("未能找到任何有效的输入文件。操作终止。")
        return

    try:
        logging.info("正在初始化嵌入模型和向量数据库...")
        embedding_model_instance = EmbeddingModel(config)
        db_manager = VectorDBManager(config, embedding_model_instance)
        logging.info("初始化完成。")

        doc_id = args.doc_id or f"custom_seed_{datetime.now().strftime('%Y%m%d')}"
        
        logging.info(f"正在加载和处理 {len(valid_files)} 个文件...")
        full_content = load_external_data(config, valid_files)
        
        if not full_content:
            logging.error("加载的文件内容为空。操作终止。")
            return
            
        chunks, metadatas = chunk_document_for_rag(config, full_content, doc_id)
        
        if not chunks:
            logging.error("未能从文件中切分出任何知识块。操作终止。")
            return

        logging.info(f"成功切分出 {len(chunks)} 个知识块，准备存入数据库...")
        success = db_manager.add_experience(texts=chunks, metadatas=metadatas)
        
        if success:
            logging.info(f"🎉 成功将 {len(chunks)} 个知识块从 {len(valid_files)} 个文件中添加到向量数据库！")
        else:
            logging.error("存入数据库时发生错误。")

    except Exception as e:
        logging.critical(f"执行知识库构建时发生严重错误: {e}", exc_info=True)

def polish_document(args):
    """功能二：修复与润色现有文档"""
    logging.info("--- 开发者模式：启动独立文档润色工具 ---")
    
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return
        
    try:
        logging.info(f"正在读取文件: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()

        # 为了润色，我们需要一个风格指南。这里我们即时生成一个。
        logging.info("正在为文档生成临时风格指南...")
        temp_config = Config()
        temp_config.user_problem = f"对以下文档进行高质量的润色和格式修复：\n\n{content[:500]}..."
        style_guide = generate_style_guide(temp_config)
        
        logging.info("正在进行AI润色...")
        polished_content = perform_final_polish(config, content, style_guide)
        
        logging.info("正在进行结构整合...")
        structured_content = consolidate_document_structure(polished_content)
        
        logging.info("正在进行最终格式后处理...")
        final_content = final_post_processing(structured_content)
        
        output_path = args.output or f"{os.path.splitext(args.input)[0]}_polished.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        logging.info(f"🎉 润色和修复完成！已保存至: {output_path}")

    except Exception as e:
        logging.critical(f"执行文档润色时发生严重错误: {e}", exc_info=True)

def generate_outline(args):
    """妙用3：快速大纲生成器"""
    logging.info("--- 开发者模式：启动快速大纲生成器 ---")
    try:
        config._initialize_deepseek_client()
        outline_data = generate_document_outline_with_tools(config, args.prompt)
        if outline_data:
            output_path = args.output or "generated_outline.json"
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outline_data, f, ensure_ascii=False, indent=2)
            logging.info(f"🎉 大纲已生成！已保存至: {output_path}")
        else:
            logging.error("生成大纲失败。")
    except Exception as e:
        logging.critical(f"执行大纲生成时发生严重错误: {e}", exc_info=True)
        
def research_gaps(args):
    """妙用4：AI研究助理"""
    logging.info("--- 开发者模式：启动AI研究助理 ---")
    try:
        config._initialize_deepseek_client()
        research_brief = run_async(run_research_cycle_async(config, args.gaps, args.context or ""))
        if research_brief:
            output_path = "research_brief.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(research_brief)
            logging.info(f"🎉 研究完成！研究简报已保存至: {output_path}")
        else:
            logging.warning("未能生成任何研究简报。")
    except Exception as e:
        logging.critical(f"执行研究时发生严重错误: {e}", exc_info=True)

def assess_quality(args):
    """妙用5：文章质量评估器"""
    logging.info("--- 开发者模式：启动文章质量评估器 ---")
    if not os.path.exists(args.file):
        logging.error(f"输入文件不存在: {args.file}")
        return
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config._initialize_deepseek_client()
        report = quality_check(config, content)
        logging.info("\n--- 质量评估报告 ---\n")
        print(report)
        logging.info("\n--- 报告结束 ---")
    except Exception as e:
        logging.critical(f"执行质量评估时发生严重错误: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Deep Research - 开发者工具箱")
    subparsers = parser.add_subparsers(dest="command", required=True, help="可用的命令")

    # 功能一：seed-db
    parser_seed = subparsers.add_parser("seed-db", help="将个人文件作为知识存入向量数据库。")
    parser_seed.add_argument("--files", required=True, type=str, help="要添加的文件路径，用逗号分隔。")
    parser_seed.add_argument("--doc_id", type=str, help="（可选）为这批文档指定一个自定义ID。")
    parser_seed.set_defaults(func=seed_knowledge_base)

    # 功能二：polish
    parser_polish = subparsers.add_parser("polish", help="对一个现有的Markdown文件进行修复和润色。")
    parser_polish.add_argument("--input", required=True, type=str, help="输入的Markdown文件路径。")
    parser_polish.add_argument("--output", type=str, help="（可选）输出的文件路径。")
    parser_polish.set_defaults(func=polish_document)
    
    # 妙用3：generate-outline
    parser_outline = subparsers.add_parser("generate-outline", help="根据一个主题快速生成结构化大纲。")
    parser_outline.add_argument("--prompt", required=True, type=str, help="生成大纲的主题或提示。")
    parser_outline.add_argument("--output", type=str, help="（可选）输出的JSON文件路径。")
    parser_outline.set_defaults(func=generate_outline)
    
    # 妙用4：research
    parser_research = subparsers.add_parser("research", help="针对一个或多个知识空白进行网络研究。")
    parser_research.add_argument("--gaps", required=True, nargs='+', help="需要研究的一个或多个知识空白点（问题）。")
    parser_research.add_argument("--context", type=str, help="（可选）提供一些上下文以帮助生成更精确的查询。")
    parser_research.set_defaults(func=research_gaps)

    # 妙用5：assess-quality
    parser_assess = subparsers.add_parser("assess-quality", help="对一个文本文件进行AI质量评估。")
    parser_assess.add_argument("--file", required=True, type=str, help="要评估的文本文件路径。")
    parser_assess.set_defaults(func=assess_quality)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()