# gecko/compose/team.py
from __future__ import annotations
import asyncio
import anyio
from typing import List, Any, Dict, Callable
from gecko.core.agent import Agent  # 集成 Day1 Agent
from gecko.core.message import Message  # 补导入

class Team:
    """
    Team 节点：多 Agent 并行执行，作为 Workflow 的特殊节点
    - 支持异步并行：AnyIO TaskGroup
    - 结果聚合：默认合并输出，可自定义 aggregator（补：强制 await）
    - 集成 Workflow：可作为 @step 节点嵌入 DAG
    """
    def __init__(self, members: List[Agent], aggregator: Callable = lambda results: results):
        self.members = members
        self.aggregator = aggregator

    async def execute(self, context: Dict) -> Any:
        """并行执行所有成员 Agent"""
        results = [None] * len(self.members)
        async def _run_member(idx: int, agent: Agent):
            messages = [Message(role="user", content=context.get("input"))]
            output = await agent.run(messages)
            results[idx] = output.content  # 简化，实际可取 full output

        async with anyio.create_task_group() as tg:
            for idx, member in enumerate(self.members):
                tg.start_soon(_run_member, idx, member)

        # 补：异步 aggregator
        return await self.aggregator(results) if asyncio.iscoroutinefunction(self.aggregator) else self.aggregator(results)