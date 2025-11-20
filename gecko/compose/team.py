# gecko/compose/team.py
from __future__ import annotations
import asyncio
import anyio
from typing import List, Any, Dict, Union
from gecko.core.agent import Agent
# 假设 WorkflowContext 在运行时作为 Any 传入，这里做 duck typing

class Team:
    """
    并行执行多个 Agent，聚合结果
    """
    def __init__(self, members: List[Agent]):
        self.members = members

    # [新增] __call__ 方法，使 Team 实例可被直接调用，满足 Workflow 节点协议
    async def __call__(self, context_or_input: Any) -> List[Any]:
        """
        允许 Team 实例像函数一样被调用：await team(context)
        """
        return await self.execute(context_or_input)

    async def execute(self, context_or_input: Any) -> List[Any]:
        """
        执行 Team 逻辑
        """
        # 解析输入：兼容直接传值或 WorkflowContext 对象
        # 使用 getattr 避免循环导入 WorkflowContext 定义
        if hasattr(context_or_input, "history") and hasattr(context_or_input, "input"):
            # 是 WorkflowContext，尝试获取上一步输出，否则取全局 input
            history = getattr(context_or_input, "history", {})
            inp = history.get("last_output", getattr(context_or_input, "input", None))
        else:
            inp = context_or_input

        results = [None] * len(self.members)

        async def _run_one(idx, agent):
            # Agent.run 现在健壮地处理 str 或 Message
            res = await agent.run(inp)
            results[idx] = res.content

        # 并发执行
        async with anyio.create_task_group() as tg:
            for idx, agent in enumerate(self.members):
                tg.start_soon(_run_one, idx, agent)

        return results