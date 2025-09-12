import logging
import threading
import asyncio
from typing import List, Optional

class LogStreamHandler(logging.Handler):
    """一个自定义的日志处理器，用于将日志发送到SSE客户端。"""
    def __init__(self, capacity: int = 100):
        super().__init__()
        self.capacity = capacity
        self.logs: List[str] = []
        self.condition = threading.Condition()
        self.new_log_event = asyncio.Event()

    def emit(self, record: logging.LogRecord):
        """重写emit方法，将日志记录添加到队列中。"""
        log_entry = self.format(record)
        with self.condition:
            if len(self.logs) >= self.capacity:
                self.logs.pop(0)
            self.logs.append(log_entry)
            self.condition.notify_all()
        # 通知异步生成器有新日志
        self.new_log_event.set()
        self.new_log_event.clear()

    async def log_generator(self):
        """生成日志流的异步生成器。"""
        # 发送已有的日志
        current_log_count = len(self.logs)
        for log in self.logs:
            yield f"data: {log}\n\n"

        # 等待新的日志
        while True:
            await self.new_log_event.wait()
            # 发送所有新的日志
            new_logs = self.logs[current_log_count:]
            current_log_count = len(self.logs)
            for log in new_logs:
                yield f"data: {log}\n\n"

# 创建一个全局的日志流处理器实例
log_stream_handler = LogStreamHandler()