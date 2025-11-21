# gecko/core/utils.py  
"""  
通用工具函数  
  
- ensure_awaitable：统一处理同步/异步调用  
"""  
  
from __future__ import annotations  
  
import asyncio  
from typing import Any, Awaitable, Callable, TypeVar  
  
T = TypeVar("T")  
  
  
async def ensure_awaitable(func: Callable[..., T | Awaitable[T]], *args, **kwargs) -> T:  
    if asyncio.iscoroutinefunction(func):  
        return await func(*args, **kwargs)  
  
    result = func(*args, **kwargs)  
    if asyncio.iscoroutine(result):  
        return await result  
    return result  
