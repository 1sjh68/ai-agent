# run_desktop_app.py

import sys
import os
import threading
import uvicorn
import webview

# --- [核心] 路径修正代码 ---
# 确保无论如何运行，总能找到项目中的模块
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# --- [核心] 路径修正结束 ---

from web.ui_app import app as fastapi_app

# 定义服务器运行的函数
def run_server():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

if __name__ == '__main__':
    # 在一个独立的线程中启动FastAPI服务器
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # 设置为守护线程，主程序退出时线程也退出
    server_thread.start()

    # 创建并启动PyWebview窗口
    # PyWebview会自动加载URL，并展示前端界面
    webview.create_window(
        '智能AI长文写作框架',  # 窗口标题
        'http://127.0.0.1:8000',  # FastAPI服务器的地址
        width=1280,
        height=800,
        resizable=True
    )
    
    # 这会阻塞主线程，直到窗口被关闭
    webview.start()
    
    # 程序退出
    sys.exit()