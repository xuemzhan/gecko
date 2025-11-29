# gecko/core/memory/_executor.py
"""
内部模块：Token 计算线程池管理

本模块只负责：
- 全局线程池的懒加载（get_token_executor）
- 全局线程池的关闭（shutdown_token_executor）

注意：
- 这是内部实现细节模块，正常情况下不建议从业务代码直接导入使用
- 对外统一通过 `gecko.core.memory.shutdown_token_executor` 暴露关闭接口
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# 全局线程池实例（单进程单实例）
_TOKEN_EXECUTOR: Optional[ThreadPoolExecutor] = None
# 保护初始化过程的锁，避免多线程下重复创建线程池
_EXECUTOR_LOCK = threading.Lock()


def get_token_executor() -> ThreadPoolExecutor:
    """
    线程安全的懒加载全局 Token 计算线程池。

    设计说明：
    - 使用“双重检查 + 互斥锁”的方式保证只创建一个线程池实例
    - `max_workers` 不宜过大：
        * Token 计算属于 CPU 密集型
        * 少量线程即可充分利用多核
    - `thread_name_prefix` 用于调试和观察线程来源
    """
    global _TOKEN_EXECUTOR
    if _TOKEN_EXECUTOR is None:
        # 第一层检查：大多数情况下这里就直接通过，不会进锁
        with _EXECUTOR_LOCK:
            # 第二层检查：保证在锁保护下只初始化一次
            if _TOKEN_EXECUTOR is None:
                _TOKEN_EXECUTOR = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="gecko_token_",
                )
    return _TOKEN_EXECUTOR


def shutdown_token_executor() -> None:
    """
    关闭全局 Token 计算线程池。

    使用场景：
    - 单元测试结束后清理资源
    - 某些需要显式控制资源生命周期的场景

    注意：
    - 正常服务进程一般可以依赖解释器退出时自动回收线程池，
      无需主动调用本函数。
    - 这里使用 `wait=False`，以避免在关闭时长时间阻塞主线程。
    """
    global _TOKEN_EXECUTOR
    if _TOKEN_EXECUTOR is not None:
        _TOKEN_EXECUTOR.shutdown(wait=False)
        _TOKEN_EXECUTOR = None
