# gecko/compose/nodes.py
"""
Workflow 节点定义与辅助工具

核心功能：
1. Next: 控制流指令，用于动态跳转节点
2. step: 节点装饰器，用于标记和增强函数元数据

优化日志：
- [Fix] 使用 functools.wraps 保留被装饰函数的元数据 (签名、文档等)
- [Refactor] 移除本地 ensure_awaitable，引用 gecko.core.utils
- [Feat] step 装饰器自动将同步函数转为异步，统一调用行为
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from gecko.core.utils import ensure_awaitable


class Next(BaseModel):
    """
    控制流指令：用于 Workflow 节点返回值中，指示引擎跳转到特定节点。
    
    示例:
        ```python
        def check_score(score: int):
            if score > 60:
                return Next(node="Pass", input="Good job")
            return Next(node="Fail", input="Try again")
        ```
    """
    node: str = Field(..., description="下一个节点的名称")
    input: Optional[Any] = Field(
        default=None, 
        description="传递给下一个节点的输入数据。如果为 None，则保持上下文中的 last_output 不变。"
    )


def step(name: Optional[str] = None):
    """
    节点装饰器
    
    功能：
    1. 标记函数为 Workflow 节点
    2. 允许自定义节点名称 (metadata)
    3. 统一将同步函数包装为异步函数
    
    参数:
        name: 自定义节点名称（可选，默认使用函数名）
        
    示例:
        ```python
        @step(name="DataFetcher")
        def fetch_data(url: str):
            return requests.get(url).text
            
        # 在 Workflow 中使用
        workflow.add_node("fetch", fetch_data)
        ```
    """
    def decorator(func: Callable):
        # 1. 保留原始函数的元数据 (name, doc, signature)
        # 这对于 Workflow 的智能参数注入 (Smart Binding) 至关重要
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 2. 统一转为异步执行
            return await ensure_awaitable(func, *args, **kwargs)
        
        # 3. 附加标识位和名称
        setattr(wrapper, "_is_step", True)
        setattr(wrapper, "_step_name", name or func.__name__)
        
        return wrapper
    return decorator