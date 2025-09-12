# web/ui_app.py

import os
import sys
import asyncio
import json
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
from datetime import datetime
import uuid

# --- 路径修正代码 ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# --- 核心模块导入 ---
from config import Config
from services.vector_db import VectorDBManager, EmbeddingModel
from workflows.graph_runner import run_graph_workflow
from workflows.sub_workflows import perform_final_polish, generate_style_guide
from utils.text_processor import final_post_processing
# [核心修改] 从 utils 导入统一的日志处理器
from utils.log_streamer import log_stream_handler
from utils.file_handler import load_external_data
from utils.text_processor import chunk_document_for_rag, consolidate_document_structure
from planning.outline import generate_document_outline_with_tools
from services.web_research import run_research_cycle_async

# --- 初始化 ---
app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# --- 配置与核心服务 ---
config = Config()

# [核心修改] 更新 setup_log_streaming 函数
def setup_log_streaming():
    # 只输出原始消息，因为JSON和普通文本都需要
    formatter = logging.Formatter('%(message)s')
    log_stream_handler.setFormatter(formatter)
    logging.getLogger().setLevel(logging.INFO)
    # 不再将Web UI的处理器添加到全局根日志记录器中

setup_log_streaming()

try:
    config._initialize_deepseek_client()
    embedding_model_instance = EmbeddingModel(config)
    vector_db_manager = VectorDBManager(config, embedding_model_instance)
except Exception as e:
    logging.critical(f"启动时初始化核心服务失败: {e}")
    embedding_model_instance = None
    vector_db_manager = None

tasks = {}
task_queue = asyncio.Queue()

# [核心修改] run_workflow_wrapper 现在设置和清除活动任务ID
async def run_workflow_wrapper(task_id: str, user_problem: str, external_files: list, target_length: int, max_iterations: int):
    logging.info(f"后台任务 {task_id} 已启动，主题: {user_problem[:50]}...")
    tasks[task_id] = {"status": "运行中", "result": None}

    # 为这个任务设置日志处理器
    log_stream_handler.set_active_task(task_id)
    # 获取根日志记录器并添加我们的处理器
    root_logger = logging.getLogger()
    root_logger.addHandler(log_stream_handler)
    
    task_config = Config()
    task_config.user_problem = user_problem
    task_config.external_data_files = external_files
    task_config.initial_solution_target_chars = target_length
    task_config.max_iterations = max_iterations
    
    try:
        logging.info(f"正在为任务 {task_id} 初始化专用的AI客户端...")
        task_config._initialize_deepseek_client()
        logging.info(f"任务 {task_id} 的AI客户端初始化成功。")
        
        # 不再传递处理器，因为它现在是全局管理的
        result = await run_graph_workflow(task_config, vector_db_manager, log_handler=None)
        
        tasks[task_id]["status"] = "已完成"
        tasks[task_id]["result"] = result
        logging.info(f"后台任务 {task_id} 成功完成。")

    except Exception as e:
        error_message = f"后台任务 {task_id} 失败: {e}"
        logging.error(error_message, exc_info=True)
        tasks[task_id]["status"] = "失败"
        tasks[task_id]["result"] = error_message
    finally:
        # [核心修改] 任务结束时，无论成功失败，都必须移除处理器并清理
        root_logger.removeHandler(log_stream_handler)
        log_stream_handler.clear_active_task()

async def worker():
    while True:
        task_id, user_problem, external_files, target_length, max_iterations = await task_queue.get()
        await run_workflow_wrapper(task_id, user_problem, external_files, target_length, max_iterations)
        task_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())
    logging.info("后台任务工作者已启动。")

# --- 路由 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_target_chars": config.initial_solution_target_chars,
        "default_iterations": config.max_iterations
    })

@app.get("/configure", response_class=HTMLResponse)
async def configure_page(request: Request):
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    return templates.TemplateResponse("configure.html", {"request": request, "current_config": config_dict})

@app.post("/update_config")
async def update_config(
    request: Request,
    main_ai_model: str = Form(...),
    main_ai_model_heavy: str = Form(...),
    embedding_model_name: str = Form(...),
    deepseek_api_key: str = Form(""),
    embedding_api_key: str = Form(""),
    google_api_keys: str = Form(""),
    google_cse_ids: str = Form(""),
    max_chunks_per_section: int = Form(...),
    num_retrieved_experiences: int = Form(...)
):
    config.main_ai_model = main_ai_model
    config.main_ai_model_heavy = main_ai_model_heavy
    config.embedding_model_name = embedding_model_name
    config.deepseek_api_key = deepseek_api_key
    config.embedding_api_key = embedding_api_key
    config.google_api_keys = [k.strip() for k in google_api_keys.split(',') if k.strip()]
    config.google_cse_ids = [i.strip() for i in google_cse_ids.split(',') if i.strip()]
    config.max_chunks_per_section = max_chunks_per_section
    config.num_retrieved_experiences = num_retrieved_experiences
    config.save_to_env()
    return RedirectResponse(url="/configure?status=success", status_code=303)

@app.post("/submit_task")
async def submit_task(
    user_problem: str = Form(...),
    external_files: list[UploadFile] = File([]),
    target_length: int = Form(...),
    max_iterations: int = Form(...)
):
    task_id = f"task_{uuid.uuid4().hex}"
    logging.info(f"收到新的写作任务，分配ID: {task_id}")

    try:
        file_paths = []
        temp_upload_dir = os.path.join("temp_uploads", task_id)
        os.makedirs(temp_upload_dir, exist_ok=True)

        for uploaded_file in external_files:
            if uploaded_file.filename:
                file_path = os.path.join(temp_upload_dir, uploaded_file.filename)
                content = await uploaded_file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
                file_paths.append(file_path)

        logging.info(f"正在将任务 {task_id} (目标长度: {target_length}, 迭代次数: {max_iterations}) 添加到处理队列...")
        await task_queue.put((task_id, user_problem, file_paths, target_length, max_iterations))
        tasks[task_id] = {"status": "排队中", "result": None}
        
        return RedirectResponse(url=f"/results?task_id={task_id}", status_code=303)
    except Exception as e:
        logging.critical(f"提交任务 {task_id} 的过程中发生严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器在处理您的任务提交时出错: {e}")

@app.get("/results", response_class=HTMLResponse)
async def show_results(request: Request, task_id: str):
    return templates.TemplateResponse("progress.html", {"request": request, "task_id": task_id})

@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    return task

# [核心修改] log_stream 现在使用新的生成器
@app.get("/api/log_stream")
async def log_stream(task_id: str):
    return StreamingResponse(log_stream_handler.log_generator(task_id), media_type="text/event-stream")

# ... (开发者工具箱路由保持不变) ...
@app.get("/toolkit", response_class=HTMLResponse)
async def toolkit_page(request: Request):
    return templates.TemplateResponse("toolkit.html", {"request": request})

@app.post("/toolkit/seed-db")
async def toolkit_seed_db(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    doc_id: str = Form(None)
):
    temp_upload_dir = os.path.join("temp_uploads", f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(temp_upload_dir, exist_ok=True)

    file_paths = []
    for uploaded_file in files:
        if uploaded_file.filename:
            file_path = os.path.join(temp_upload_dir, uploaded_file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    buffer.write(await uploaded_file.read())
                file_paths.append(file_path)
                logging.info(f"Seed file '{uploaded_file.filename}' saved to '{file_path}'")
            except Exception as e:
                logging.error(f"Failed to save seed file '{uploaded_file.filename}': {e}")

    def seed_task(paths):
        logging.info("--- [Web UI] 启动知识库构建任务 ---")
        valid_files = paths
        if valid_files and vector_db_manager:
            doc_identifier = doc_id or f"custom_seed_{datetime.now().strftime('%Y%m%d')}"
            full_content = load_external_data(config, valid_files)
            if full_content:
                chunks, metadatas = chunk_document_for_rag(config, full_content, doc_identifier)
                if chunks:
                    vector_db_manager.add_experience(texts=chunks, metadatas=metadatas)
                    logging.info(f"--- [Web UI] 知识库构建完成 ---")

    background_tasks.add_task(seed_task, file_paths)
    return JSONResponse(content={"message": "知识库构建任务已在后台启动，请在 session.log 中查看进度。"})

@app.post("/toolkit/polish")
async def toolkit_polish(file: UploadFile = File(...)):
    if not file.filename.endswith(('.md', '.txt')):
        raise HTTPException(status_code=400, detail="请上传 .md 或 .txt 文件。")
    content = await file.read()
    content_str = content.decode('utf-8')
    
    temp_config = Config()
    temp_config.user_problem = f"润色以下文档：\n{content_str[:500]}..."
    style_guide = generate_style_guide(temp_config)
    
    polished_content = perform_final_polish(config, content_str, style_guide)
    structured_content = consolidate_document_structure(polished_content)
    final_content = final_post_processing(structured_content)

    return Response(content=final_content, media_type="text/markdown", headers={
        'Content-Disposition': f'attachment; filename="polished_{file.filename}"'
    })

@app.post("/toolkit/generate-outline")
async def toolkit_generate_outline(prompt: str = Form(...)):
    outline_data = generate_document_outline_with_tools(config, prompt)
    if outline_data:
        return JSONResponse(content=outline_data)
    raise HTTPException(status_code=500, detail="生成大纲失败。")

@app.post("/toolkit/research")
async def toolkit_research(gaps: str = Form(...)):
    gaps_list = [gap.strip() for gap in gaps.splitlines() if gap.strip()]
    if not gaps_list:
        raise HTTPException(status_code=400, detail="请输入至少一个研究问题。")
    research_brief = await run_research_cycle_async(config, gaps_list, "")
    return Response(content=research_brief, media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)