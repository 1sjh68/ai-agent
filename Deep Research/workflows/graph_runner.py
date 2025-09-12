# workflows/graph_runner.py

import logging
from typing import Optional

from config import Config
from services.vector_db import VectorDBManager
from workflows.graph_builder import build_graph
from utils.file_handler import load_external_data

async def run_graph_workflow(
    config: Config,
    vector_db_manager: Optional[VectorDBManager],
    log_handler: Optional[logging.Handler] = None, # 接收从Web UI传递的处理器
    initial_solution: Optional[str] = None
) -> str:
    """
    (V2 - 带动态日志处理器)
    使用LangGraph编排内容生成过程。
    """
    root_logger = logging.getLogger()
    if log_handler:
        # 如果有传递处理器（说明是从Web UI调用的），则将其添加到根日志记录器
        root_logger.addHandler(log_handler)

    try:
        logging.info("\n--- 开始基于LangGraph的内容生成工作流 ---")

        # 1. 准备来自外部文件和过去经验的初始上下文
        loaded_ext_data = load_external_data(config, config.external_data_files or [])
        retrieved_experience_text = ""
        if vector_db_manager:
            retrieved_exps = vector_db_manager.retrieve_experience(config.user_problem)
            if retrieved_exps:
                exp_texts = [
                    f"---历史经验 {i+1} (相关度: {exp.get('distance', -1):.4f})---\n{exp.get('document')}"
                    for i, exp in enumerate(retrieved_exps)
                ]
                retrieved_experience_text = "\n\n===== 检索到的相关历史经验 =====\n" + \
                                          "\n\n".join(exp_texts) + "\n===== 历史经验结束 =====\n\n"
                logging.info(f"成功检索并格式化了 {len(retrieved_exps)} 条经验。")

        initial_external_data = retrieved_experience_text + loaded_ext_data

        # 2. 构建图
        app = build_graph()

        # 3. 在config中设置共享对象，以便在状态中访问
        setattr(config, 'vector_db_manager', vector_db_manager)
        if vector_db_manager:
            setattr(config, 'embedding_model_instance', vector_db_manager.embedding_model)

        # 4. 准备图的初始状态
        initial_state = {
            "topic": config.user_problem,
            "config": config,
            "external_data": initial_external_data,
            "refinement_count": 0,
            "current_chapter_index": 0,
            "completed_chapters_content": {},
            "draft_content": initial_solution,
        }

        logging.info(f"为主题调用图: '{config.user_problem}'")

        # 5. 异步运行图
        final_state = await app.ainvoke(initial_state)

        final_answer = final_state.get("final_solution")

        if not final_answer:
            logging.error("工作流已完成，但未生成最终解决方案。")
            return "错误：工作流未产生最终答案。"

        logging.info("\n--- LangGraph工作流成功完成 ---")
        return final_answer

    except Exception as e:
        logging.critical(f"工作流中发生意外错误: {e}", exc_info=True)
        return f"发生意外错误: {e}"

    finally:
        if log_handler and root_logger:
            # 任务结束时，无论成功还是失败，都必须移除处理器
            # 以免对其他任务或下一次运行造成干扰
            root_logger.removeHandler(log_handler)