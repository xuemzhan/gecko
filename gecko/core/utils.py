# gecko/core/utils.py
import asyncio
from typing import Any, Callable

async def ensure_awaitable(func: Callable, *args, **kwargs) -> Any:
    """辅助函数：如果结果是协程则 await，否则直接返回"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result