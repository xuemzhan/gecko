# gecko/compose/team.py
"""
Team 多智能体并行引擎

提供 Map-Reduce 模式的并行执行能力，将单一任务分发给多个 Agent/Function 执行，
并聚合结果。适用于 "专家评审团"、"多路赛马"、"并发搜索" 等场景。

核心功能：
1. 高效并行：基于 AnyIO TaskGroup 实现异步并发。
2. 流量整形：支持 max_concurrent 限制，防止触发 LLM Rate Limit。
3. 容错机制：单个成员失败不熔断整体任务 (Partial Success)。
4. 智能绑定：自动解析 WorkflowContext，支持数据流转。

优化日志：
- [Fix] 增加 max_concurrent 信号量控制，防止 API 速率超限
- [Fix] 完善异常捕获边界，确保 TaskGroup 稳定性
- [Refactor] 统一输入解析与结果标准化逻辑
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Union, TYPE_CHECKING

import anyio
from pydantic import BaseModel

from gecko.core.agent import Agent
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable

if TYPE_CHECKING:
    # 避免运行时循环导入，仅用于类型检查
    from gecko.compose.workflow import WorkflowContext

logger = get_logger(__name__)


class Team:
    """
    多智能体协作组 (Parallel Execution Engine)
    
    将同一个输入广播给多个成员并行执行，并收集结果。
    
    示例:
        ```python
        # 创建一个包含 3 个 Agent 的团队，最大并发数为 2
        team = Team(
            members=[researcher, reviewer, coder],
            max_concurrent=2
        )
        
        # 执行
        results = await team.run("Design a snake game")
        # results = ["Research doc...", "Review notes...", "Python code..."]
        ```
    """

    def __init__(
        self, 
        members: List[Union[Agent, Callable]],
        name: str = "Team",
        max_concurrent: int = 0,
        return_full_output: bool = False
    ):
        """
        初始化 Team
        
        参数:
            members: 成员列表 (Agent 实例或可调用的函数)
            name: Team 名称 (用于日志追踪)
            max_concurrent: 最大并发数 (0 表示不限制，默认不限制)
            return_full_output: 是否返回完整对象 (如 AgentOutput)。
                                False (默认): 仅提取文本内容 (content)
                                True: 返回原始执行结果对象
        """
        self.members = members
        self.name = name
        self.max_concurrent = max_concurrent
        self.return_full_output = return_full_output

    # ========================= 接口协议 =========================

    async def __call__(self, context_or_input: Any) -> List[Any]:
        """
        实现 Callable 协议，使 Team 实例可直接作为 Workflow 节点使用
        """
        return await self.run(context_or_input)

    async def run(self, context_or_input: Any) -> List[Any]:
        """
        执行 Team 逻辑
        
        参数:
            context_or_input: 输入数据，支持原始值或 WorkflowContext
            
        返回:
            成员执行结果列表 (顺序与 members 一致)
        """
        # 1. 解析输入 (Context -> Data)
        inp = self._resolve_input(context_or_input)
        
        member_count = len(self.members)
        logger.info(
            "Team execution started", 
            team=self.name, 
            member_count=member_count,
            max_concurrent=self.max_concurrent or "unlimited"
        )

        # 2. 初始化容器
        results: List[Any] = [None] * member_count
        # 用于统计
        errors: List[Optional[Exception]] = [None] * member_count

        # 3. 准备并发控制
        # 如果设置了 max_concurrent，创建信号量；否则为 None
        semaphore = anyio.Semaphore(self.max_concurrent) if self.max_concurrent > 0 else None

        # 4. 定义 Worker
        async def _worker(idx: int, member: Any):
            # 如果有信号量，先获取许可
            if semaphore:
                await semaphore.acquire()
            
            try:
                # 执行成员逻辑
                raw_result = await self._execute_member(member, inp)
                # 结果标准化
                results[idx] = self._process_result(raw_result)
            except Exception as e:
                # 捕获异常，保证其他成员继续执行
                logger.error(
                    "Team member execution failed",
                    team=self.name,
                    member_index=idx,
                    error=str(e)
                )
                errors[idx] = e
                # 优雅降级：返回错误字符串，而不是抛出异常中断流程
                results[idx] = f"Error: {str(e)}"
            finally:
                # 释放信号量
                if semaphore:
                    semaphore.release()

        # 5. 启动并发任务组
        async with anyio.create_task_group() as tg:
            for idx, member in enumerate(self.members):
                tg.start_soon(_worker, idx, member)

        # 6. 执行摘要
        fail_count = sum(1 for e in errors if e is not None)
        logger.info(
            "Team execution completed",
            team=self.name,
            success=member_count - fail_count,
            failed=fail_count
        )

        return results

    # ========================= 内部逻辑 =========================

    def _resolve_input(self, context_or_input: Any) -> Any:
        """
        智能输入解析
        
        从 WorkflowContext 中提取真正需要传递给 Agent 的 Prompt 数据，
        同时处理 Data Handover（上一步输出是复杂对象的情况）。
        """
        # 1. 检查是否为 WorkflowContext (Duck Typing)
        # 判断依据: 具有 input, history 属性，且 history 是字典
        if (
            hasattr(context_or_input, "history") 
            and hasattr(context_or_input, "input")
            and isinstance(getattr(context_or_input, "history", None), dict)
        ):
            ctx = context_or_input
            history = getattr(ctx, "history", {})
            state = getattr(ctx, "state", {})
            
            # 优先级: 
            # 1. 显式传递的 _next_input (Next 指令)
            # 2. 上一步输出 (last_output)
            # 3. 全局初始输入 (input)
            val = state.pop("_next_input", None) or history.get("last_output", getattr(ctx, "input"))
            
            # 2. Data Handover 清洗
            # 如果上一步输出是 AgentOutput 的字典形式 (含 content, role, tool_calls 等)
            # 我们通常只需要 content 传给下一个 Agent，避免 Prompt 污染
            if isinstance(val, dict) and "content" in val and "role" not in val:
                return val["content"]
            
            return val
            
        # 3. 普通输入直接返回
        return context_or_input

    async def _execute_member(self, member: Any, inp: Any) -> Any:
        """执行单个成员 (支持 Agent 和 Async/Sync Function)"""
        # Case A: Agent (具备 run 方法)
        if hasattr(member, "run"):
            return await member.run(inp)
            
        # Case B: Callable
        if callable(member):
            return await ensure_awaitable(member, inp)
            
        raise TypeError(f"Member {member} is not executable (must be Agent or Callable)")

    def _process_result(self, result: Any) -> Any:
        """结果标准化处理"""
        if self.return_full_output:
            # 返回完整对象 (Pydantic 序列化)
            if isinstance(result, (BaseModel, AgentOutput, Message)):
                return result.model_dump()
            return result
            
        # 默认模式：仅提取核心文本内容
        if isinstance(result, AgentOutput):
            return result.content
        if isinstance(result, Message):
            return result.content
        if isinstance(result, dict) and "content" in result:
            return result["content"]
            
        return result

    def __repr__(self) -> str:
        return f"Team(name='{self.name}', members={len(self.members)}, concurrency={self.max_concurrent})"