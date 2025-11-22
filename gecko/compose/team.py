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

优化日志：
- [Refactor] 引入 MemberResult 标准化返回结果
- [Refactor] 移除输入处理中的隐式拆包逻辑 (Magic Unpacking)
- [Fix] 异常处理改为对象封装，不再返回错误字符串
"""
from __future__ import annotations

from typing import Any, Callable, Generic, List, Optional, TypeVar, Union, TYPE_CHECKING

import anyio
from pydantic import BaseModel, Field

from gecko.core.agent import Agent
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable

if TYPE_CHECKING:
    from gecko.compose.workflow import WorkflowContext

logger = get_logger(__name__)

R = TypeVar("R")


class MemberResult(BaseModel, Generic[R]):
    """
    标准化成员执行结果
    """
    result: Optional[R] = Field(default=None, description="执行成功时的返回值")
    error: Optional[str] = Field(default=None, description="执行失败时的错误信息")
    member_index: int = Field(..., description="成员在 Team 中的索引")
    is_success: bool = Field(default=True, description="是否执行成功")

    @property
    def value(self) -> R:
        """
        获取结果，如果是错误则抛出异常 (便捷方法)
        """
        if not self.is_success:
            raise RuntimeError(f"Member execution failed: {self.error}")
        return self.result # type: ignore


class Team:
    """
    多智能体协作组 (Parallel Execution Engine)
    """

    def __init__(
        self, 
        members: List[Union[Agent, Callable]],
        name: str = "Team",
        max_concurrent: int = 0,
        return_full_output: bool = False
    ):
        self.members = members
        self.name = name
        self.max_concurrent = max_concurrent
        self.return_full_output = return_full_output

    # ========================= 接口协议 =========================

    async def __call__(self, context_or_input: Any) -> List[MemberResult]:
        """
        实现 Callable 协议
        """
        return await self.run(context_or_input)

    async def run(self, context_or_input: Any) -> List[MemberResult]:
        """
        执行 Team 逻辑
        
        返回:
            MemberResult 列表，包含每个成员的执行状态和结果
        """
        # 1. 解析输入 (不再进行隐式内容提取)
        inp = self._resolve_input(context_or_input)
        
        member_count = len(self.members)
        logger.info(
            "Team execution started", 
            team=self.name, 
            member_count=member_count,
            max_concurrent=self.max_concurrent or "unlimited"
        )

        # 2. 初始化容器
        # 使用 MemberResult 占位，初始状态设为失败，防止未执行的情况
        results: List[Optional[MemberResult]] = [None] * member_count
        
        # 3. 准备并发控制
        semaphore = anyio.Semaphore(self.max_concurrent) if self.max_concurrent > 0 else None

        # 4. 定义 Worker
        async def _worker(idx: int, member: Any):
            if semaphore:
                await semaphore.acquire()
            
            try:
                # 执行成员逻辑
                raw_result = await self._execute_member(member, inp)
                # 结果标准化
                processed = self._process_result(raw_result)
                
                results[idx] = MemberResult(
                    member_index=idx,
                    result=processed,
                    is_success=True
                )
            except Exception as e:
                logger.error(
                    "Team member execution failed",
                    team=self.name,
                    member_index=idx,
                    error=str(e)
                )
                results[idx] = MemberResult(
                    member_index=idx,
                    error=str(e),
                    is_success=False
                )
            finally:
                if semaphore:
                    semaphore.release()

        # 5. 启动并发任务组
        async with anyio.create_task_group() as tg:
            for idx, member in enumerate(self.members):
                tg.start_soon(_worker, idx, member)

        # 6. 结果整理
        # 理论上 task_group 结束时所有 results 都已被赋值，这里做一次非空断言过滤
        final_results = [r for r in results if r is not None]
        
        # 统计
        success_count = sum(1 for r in final_results if r.is_success)
        fail_count = member_count - success_count
        
        logger.info(
            "Team execution completed",
            team=self.name,
            success=success_count,
            failed=fail_count
        )

        return final_results

    # ========================= 内部逻辑 =========================

    def _resolve_input(self, context_or_input: Any) -> Any:
        """
        智能输入解析 (Refactored: Removed Magic Unpacking)
        
        仅保留从 WorkflowContext 中提取数据的逻辑，移除针对 dict 内容的自动拆包。
        """
        # 1. 检查是否为 WorkflowContext (Duck Typing)
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
            
            # [Removed] Data Handover 清洗逻辑
            # 以前这里会检查 dict["content"]，现在移除，保持原始数据完整性
            return val
            
        # 2. 普通输入直接返回
        return context_or_input

    async def _execute_member(self, member: Any, inp: Any) -> Any:
        """执行单个成员"""
        if hasattr(member, "run"):
            return await member.run(inp)
        
        if callable(member):
            return await ensure_awaitable(member, inp)
            
        raise TypeError(f"Member {member} is not executable (must be Agent or Callable)")

    def _process_result(self, result: Any) -> Any:
        """结果标准化处理"""
        if self.return_full_output:
            if isinstance(result, (BaseModel, AgentOutput, Message)):
                return result.model_dump()
            return result
            
        # 默认模式：仅提取核心文本内容
        if isinstance(result, AgentOutput):
            return result.content
        if isinstance(result, Message):
            return result.content
        # [Modified] 这里的 dict 检查仅针对 AgentOutput.model_dump() 后的结构
        # 普通的 dict 不会被提取 content，除非它是 AgentOutput 的序列化形式
        # 但为了安全起见，我们仅处理明确的对象类型，对于 dict 保持原样，
        # 或者我们可以保留这里的逻辑如果确信 dict 是由 Agent 产生的。
        # 为了贯彻 "No Magic"，建议此处如果用户返回的是 dict，就返回 dict。
        # AgentOutput/Message 对象在上一步已经被识别处理了。
        
        return result

    def __repr__(self) -> str:
        return f"Team(name='{self.name}', members={len(self.members)}, concurrency={self.max_concurrent})"