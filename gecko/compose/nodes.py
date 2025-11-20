# gecko/compose/nodes.py
from __future__ import annotations
import asyncio
from typing import Any, Callable, List, Optional
from pydantic import BaseModel

# [新增] 控制流指令
class Next(BaseModel):
    """
    节点返回值指令：明确指示 Workflow 跳转到下一个节点
    """
    node: str
    input: Optional[Any] = None  # 可选：修改传递给下个节点的输入

# [保留] 辅助函数
async def ensure_awaitable(func: Callable, *args, **kwargs) -> Any:
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result

# [保留] 装饰器 (略微简化，适配新机制)
def step(name: Optional[str] = None):
    def decorator(func: Callable):
        func._is_step = True
        func._step_name = name or func.__name__
        return func
    return decorator

# Loop 和 Parallel 可以暂时保留，但在新引擎中可能不再作为特殊节点，
# 而是作为普通节点内部的逻辑，或者通过 Next 实现循环。
# 为了兼容性，我们暂且保留原定义，但建议后续通过 Next 实现循环。