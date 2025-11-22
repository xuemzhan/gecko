# gecko/plugins/storage/mixins.py
"""
存储功能混入类 (Mixins)

包含解决异步阻塞、并发冲突和数据序列化的通用逻辑。
这是本次重构的核心，用于修复 Event Loop 阻塞问题。
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Callable, TypeVar, Optional

from anyio import to_thread

from gecko.core.logging import get_logger

logger = get_logger(__name__)

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
    [核心] 原子写混入类
    
    提供应用层的异步写锁，防止文件型数据库（如 SQLite, LanceDB）
    在并发写入时发生 'database is locked' 错误。
    
    修复：使用 Lazy Loading 替代 __init__，防止多重继承时初始化链断裂。
    """
    
    _write_lock: Optional[asyncio.Lock] = None

    @property
    def write_lock(self) -> asyncio.Lock:
        """懒加载获取锁"""
        # 注意：这里需要处理 _write_lock 可能不存在的情况（虽然类属性已定义）
        if getattr(self, "_write_lock", None) is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock # type: ignore

    @asynccontextmanager
    async def write_guard(self):
        """
        写操作保护上下文
        
        示例:
            async with self.write_guard():
                await self._run_sync(sync_write_func)
        """
        # 使用 property 获取锁
        async with self.write_lock:
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