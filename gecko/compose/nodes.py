# gecko/compose/nodes.py
from __future__ import annotations
import asyncio
import anyio
from typing import TYPE_CHECKING, Callable, List, Optional, Any
from gecko.core.events import RunEvent
from gecko.core.utils import ensure_awaitable  # [修复] 导入辅助函数

if TYPE_CHECKING:
    from .workflow import Workflow

def step(name: Optional[str] = None):
    """
    节点装饰器：将普通函数或异步函数标记为 Workflow 节点。
    
    用法:
        @step("research")
        async def research(context): ...
    """
    def decorator(func: Callable):
        # 包装函数，确保在 Workflow 中调用时始终是异步的
        async def wrapper(context):
            return await ensure_awaitable(func, context)
        
        # 保留原函数的元数据（如 __name__），以便日志和可视化使用
        wrapper.__name__ = name or func.__name__
        return wrapper
    return decorator

class Condition:
    """条件节点：用于 Workflow 的边（Edge）判断条件"""
    def __init__(self, predicate: Callable, name: Optional[str] = None):
        self.predicate = predicate
        self.name = name

    async def __call__(self, context: Any) -> bool:
        # [修复] 使用 ensure_awaitable，兼容 lambda x: True 这种同步写法
        return await ensure_awaitable(self.predicate, context)

class Loop:
    """循环节点：包装一个子 Workflow 进行循环执行"""
    def __init__(
        self,
        body: Workflow,
        condition: Callable,
        max_iters: int = 5,
        event_bus = None
    ):
        self.body = body
        self.condition = condition
        self.max_iters = max_iters
        # 如果没有传入 event_bus，尝试从 body 中获取
        self.event_bus = event_bus or (body.event_bus if hasattr(body, "event_bus") else None)

    async def execute(self, context: Any) -> Any:
        iters = 0
        if self.event_bus:
            await self.event_bus.publish(RunEvent(type="loop_started", data={"max_iters": self.max_iters}))
        
        # [修复] 这里的 condition 可能是同步 lambda，使用 ensure_awaitable 包装
        # 只有当 condition 返回 True 且未达到最大迭代次数时继续
        while (await ensure_awaitable(self.condition, context)) and iters < self.max_iters:
            # 执行循环体（子 Workflow）
            # 注意：通常循环会将当前 context 作为输入，并将输出合并回 context
            # [修复] 调用子 Workflow 时要求返回完整 context，以便捕获副作用（如计数器更新）
            loop_context = await self.body.execute(context.get("input", context), return_context=True)
            
            # 如果结果是字典，更新当前上下文（模拟状态流转）
            if isinstance(loop_context, dict) and isinstance(context, dict):
                context.update(loop_context)
            iters += 1
            
        if self.event_bus:
            await self.event_bus.publish(RunEvent(type="loop_completed", data={"iters": iters}))
        return context

class Parallel:
    """并行节点：并发执行多个步骤"""
    def __init__(self, steps: List[Callable], event_bus: Optional = None):
        self.steps = steps
        self.event_bus = event_bus

    async def execute(self, context: Any) -> List[Any]:
        if self.event_bus:
            await self.event_bus.publish(RunEvent(type="parallel_started", data={"steps": len(self.steps)}))
        
        # 初始化结果列表，保持顺序
        results = [None] * len(self.steps)
        
        async def _run_step(idx: int, step: Callable):
            # [修复] 确保每个子步骤都被正确 await
            try:
                res = await ensure_awaitable(step, context)
                results[idx] = res
            except Exception as e:
                results[idx] = f"Error: {str(e)}"

        # 使用 AnyIO TaskGroup 进行真·异步并发
        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)
        
        if self.event_bus:
            await self.event_bus.publish(RunEvent(type="parallel_completed", data={"results": results}))
        return results