# workflows/generation.py

import logging
from typing import Optional

from config import Config
from services.vector_db import VectorDBManager
from utils.file_handler import load_external_data
from utils.text_processor import calculate_checksum

# 从新的子模块导入核心功能
from .sub_workflows import (
    generate_style_guide,
    optimize_solution_with_two_ais
)

async def generate_extended_content_workflow(
    config: Config,
    vector_db_manager: VectorDBManager | None,
    log_handler: logging.Handler | None = None,
    initial_solution: Optional[str] = None
) -> str:
    """
    主工作流，编排整个内容生成过程。
    (已重构为高层协调器)
    """
    root_logger = logging.getLogger()
    if log_handler:
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)

    try:
        logging.info("\n--- 开始扩展内容生成工作流 ---")

        # 1. 生成风格指南
        style_guide = generate_style_guide(config)

        # 2. 加载和准备外部数据与历史经验
        loaded_ext_data = load_external_data(config, config.external_data_files or [])
        retrieved_experience_text = ""
        if vector_db_manager:
            retrieved_exps = vector_db_manager.retrieve_experience(config.user_problem)
            if retrieved_exps:
                exp_texts = [
                    f"---历史经验 {i+1} (相关度: {exp.get('distance', -1):.4f})---\n{exp.get('document')}"
                    for i, exp in enumerate(retrieved_exps)
                ]
                retrieved_experience_text = "\n\n===== 上下文检索到的相关历史经验 =====\n" + \
                                          "\n\n".join(exp_texts) + "\n===== 历史经验结束 =====\n\n"
                logging.info(f"成功检索并格式化了 {len(retrieved_exps)} 条经验。")

        final_external_data = retrieved_experience_text + loaded_ext_data
        ext_data_checksum = calculate_checksum(final_external_data)

        # 3. 运行核心的迭代优化流程
        final_answer, _, _, _, error_message = await optimize_solution_with_two_ais(
            config,
            config.user_problem,
            style_guide,
            final_external_data,
            ext_data_checksum,
            vector_db_manager,
            initial_solution
        )

        # 4. 处理结果
        if error_message:
            logging.error(f"优化过程因错误而终止: {error_message}")
            return f"错误: {error_message}"
        if not final_answer:
            logging.error("优化过程完成，但未生成任何有效答案。")
            return "错误：工作流未产生任何答案。"

        logging.info("\n--- 工作流成功完成 ---")
        return final_answer

    finally:
        if log_handler:
            root_logger.removeHandler(log_handler)