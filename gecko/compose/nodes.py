# gecko/compose/nodes.py
from __future__ import annotations
import asyncio
import anyio
from typing import TYPE_CHECKING, Callable, List, Optional
if TYPE_CHECKING:
    from .workflow import Workflow
from gecko.core.events import RunEvent

def step(name: Optional[str] = None):
    """
    节点装饰器：自动注册 callable / Agent / Team 为 Workflow 节点
    用法：
    @step("research")
    async def research_func(context): ...
    """
    def decorator(func: Callable):
        step_name = name or func.__name__
        # 返回包装器，便于 Workflow.add_node(step_name, func)
        async def wrapper(context):
            return await func(context) if asyncio.iscoroutinefunction(func) else func(context)
        return wrapper
    return decorator

class Condition:
    """条件节点：作为 edge condition 使用"""
    def __init__(self, predicate: Callable):
        self.predicate = predicate

    async def __call__(self, context):
        return await self.predicate(context) if asyncio.iscoroutinefunction(self.predicate) else self.predicate(context)

class Loop:
    """循环节点：包装子 Workflow，支持 max_iters"""
    def __init__(
        self,
        body: Workflow,  # 类型注解使用 Workflow（TYPE_CHECKING 下安全）
        condition: Callable,
        max_iters: int = 5,
        event_bus = None
        ):
        self.body = body
        self.condition = condition
        self.max_iters = max_iters
        self.event_bus = event_bus or (body.event_bus if hasattr(body, "event_bus") else None)

    async def execute(self, context):
        iters = 0
        await self.event_bus.publish(RunEvent(type="loop_started", data={"max_iters": self.max_iters}))
        while await self.condition(context) and iters < self.max_iters:
            # 关键修复：递归调用后继承 output
            loop_context = await self.body.execute(context.get("input", context))  # 输入继承
            context.update(loop_context)  # 输出合并
            iters += 1
        await self.event_bus.publish(RunEvent(type="loop_completed", data={"iters": iters}))
        return context

class Parallel:
    """并行节点：包装多个子节点，默认 TaskGroup 并行（补：事件广播）"""
    def __init__(self, steps: List[Callable], event_bus: Optional = None): # type: ignore
        self.steps = steps
        self.event_bus = event_bus

    async def execute(self, context):
        await self.event_bus.publish(RunEvent(type="parallel_started", data={"steps": len(self.steps)}))
        results = []
        async def _run_step(step):
            results.append(await step(context) if asyncio.iscoroutinefunction(step) else step(context))

        async with anyio.create_task_group() as tg:
            for step in self.steps:
                tg.start_soon(_run_step, step)
        await self.event_bus.publish(RunEvent(type="parallel_completed", data={"results": results}))
        return results
    
class Condition:
    """条件节点：支持自定义显示名称（增强版）"""
    def __init__(self, predicate: Callable, name: Optional[str] = None):
        self.predicate = predicate
        self.name = name  # 新增：用户可指定标签名

    async def __call__(self, context):
        return await self.predicate(context) if asyncio.iscoroutinefunction(self.predicate) else self.predicate(context)