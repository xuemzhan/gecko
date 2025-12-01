# gecko/plugins/storage/mixins.py
"""
存储功能混入类 (Mixins)

包含解决异步阻塞、并发冲突和数据序列化的通用逻辑。

更新日志：
- [Arch] 引入 FileLock 实现跨进程文件锁，解决多进程环境下 SQLite/文件存储的并发安全问题。
- [Fix] 分离 asyncio.Lock (write_guard) 和 FileLock (file_lock_guard) 以避免死锁。
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from functools import partial
import os
from typing import Any, Callable, TypeVar, Optional

from anyio import to_thread

from gecko.core.exceptions import ConfigurationError
from gecko.core.logging import get_logger

logger = get_logger(__name__)

# 尝试导入 filelock，用于跨进程文件锁
try:
    from filelock import FileLock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False

T = TypeVar("T")


class ThreadOffloadMixin:
    """
    [核心] 线程卸载混入类
    
    将同步的 IO 操作（如 sqlite3, chromadb, pandas 操作）卸载到
    独立的线程池中执行，防止阻塞主线程的 Event Loop。
    
    原理: 使用 anyio.to_thread.run_sync
    """
    
    async def _run_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        在线程池中执行同步函数
        
        参数:
            func: 同步函数
            *args, **kwargs: 传递给函数的参数
            
        返回:
            函数执行结果
        """
        if kwargs:
            func = partial(func, **kwargs)
        
        return await to_thread.run_sync(func, *args)


class AtomicWriteMixin:
    """
    [核心] 原子写混入类 (增强版)
    
    提供双层锁机制：
    1. asyncio.Lock: 保证单进程内协程的互斥 (write_guard)。
    2. FileLock: 保证跨进程（如 Gunicorn Worker）对同一文件的互斥访问 (file_lock_guard)。
    
    使用方法：
        1. 在子类 __init__ 中调用 self.setup_multiprocess_lock(path) 启用文件锁。
        2. 在 async 方法中由于 await _run_sync 调用前使用 async with self.write_guard()。
        3. 在 _run_sync 调用的同步函数内部使用 with self.file_lock_guard()。
    """
    
    _write_lock: Optional[asyncio.Lock] = None
    _file_lock: Any = None

    @property
    def write_lock(self) -> asyncio.Lock:
        """懒加载获取协程锁"""
        if getattr(self, "_write_lock", None) is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock # type: ignore

    def setup_multiprocess_lock(self, lock_path: str) -> None:
        """
        配置跨进程文件锁
        
        参数:
            lock_path: 锁文件路径（通常是数据库文件路径）
        """
        # 获取当前运行环境，默认为 development
        env = os.getenv("GECKO_ENV", "development").lower()
        
        if not FILELOCK_AVAILABLE:
            msg = (
                "filelock module not installed. "
                "Install with: pip install filelock"
            )
            # 生产环境强制检查
            if env == "production":
                raise ConfigurationError(
                    f"[CRITICAL] Running in PRODUCTION mode without filelock! "
                    f"This will cause data corruption in multi-worker setups. {msg}"
                )
            
            logger.warning(
                f"filelock module not installed. Cross-process safety is NOT guaranteed. {msg}"
            )
            return

        try:
            self._file_lock = FileLock(f"{lock_path}.lock") # type: ignore
            logger.debug("FileLock initialized", path=f"{lock_path}.lock")
        except Exception as e:
            if env == "production":
                raise ConfigurationError(f"Failed to initialize FileLock in production: {e}") from e
            logger.error("Failed to initialize FileLock", error=str(e))

    @asynccontextmanager
    async def write_guard(self):
        """
        写操作保护上下文 (Async)
        
        只负责进程内协程互斥。
        """
        async with self.write_lock:
            yield

    @contextmanager
    def file_lock_guard(self):
        """
        跨进程文件锁 (Sync)
        
        必须在 Worker 线程（_run_sync 调用的函数）内部使用。
        确保 acquire/release 在同一个线程中执行，避免 Reentrancy 问题。
        """
        if self._file_lock:
            with self._file_lock:
                yield
        else:
            yield


class JSONSerializerMixin:
    """
    JSON 序列化混入类
    
    提供标准化的数据序列化/反序列化方法。
    """
    
    def _serialize(self, data: Any) -> str:
        """序列化为 JSON 字符串"""
        try:
            # ensure_ascii=False 减少体积并保持中文可读性
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error("Serialization failed", error=str(e))
            raise

    def _deserialize(self, data: str | bytes | None) -> Any:
        """从 JSON 字符串反序列化"""
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error("Deserialization failed", error=str(e))
            return None