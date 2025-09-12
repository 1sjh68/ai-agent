# utils/log_streamer.py

import logging
import asyncio
from typing import Dict
import queue

class LogStreamHandler(logging.Handler):
    """
    一个自定义的日志处理器，用于将日志通过SSE发送给前端。
    (V3 - 任务特定队列版)
    """
    def __init__(self):
        super().__init__()
        self.queues: Dict[str, asyncio.Queue] = {}
        self.active_task_id: str | None = None

    def set_active_task(self, task_id: str):
        """设置当前活动的任务ID，以便日志能被正确路由。"""
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()
        self.active_task_id = task_id

    def clear_active_task(self):
        """清除活动的任务ID。"""
        self.active_task_id = None

    def emit(self, record: logging.LogRecord):
        """将日志记录放入当前活动任务的队列中。"""
        if self.active_task_id:
            try:
                msg = self.format(record)
                self.queues[self.active_task_id].put_nowait(msg)
            except asyncio.QueueFull:
                # 在队列满时可以决定是丢弃还是如何处理
                pass
            except Exception:
                self.handleError(record)

    async def log_generator(self, task_id: str):
        """为特定任务ID创建异步日志生成器。"""
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()

        q = self.queues[task_id]
        while True:
            try:
                log_entry = await q.get()
                yield f"data: {log_entry}\n\n"
                q.task_done()
            except asyncio.CancelledError:
                break

# 创建一个全局的日志流处理器实例
log_stream_handler = LogStreamHandler()