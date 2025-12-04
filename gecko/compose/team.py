# gecko/compose/team.py
"""
Team 多智能体并行引擎 (Enhanced)

优化日志：
- [Feat] 增加 ExecutionStrategy 枚举 (ALL, RACE)
- [Feat] 增加 input_mapper 支持输入分片 (Sharding)
- [Perf] 实现 Race 模式下的快速返回与任务取消
- [Refactor] 保持 MemberResult 标准化结构
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Generic, List, Optional, TypeVar, Union, TYPE_CHECKING

import anyio
from pydantic import BaseModel, Field

from gecko.core.agent import Agent
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable

if TYPE_CHECKING:
    from gecko.compose.workflow.models import WorkflowContext

logger = get_logger(__name__)

R = TypeVar("R")


class ExecutionStrategy(str, Enum):
    """
    执行策略枚举
    """
    ALL = "all"       # 等待所有成员执行完毕 (Map-Reduce 默认)
    RACE = "race"     # 赛马模式：取第一个成功的结果，取消其他任务 (降低延迟)
    # Future: MAJORITY (投票), BEST_OF_N (采样)


class MemberResult(BaseModel, Generic[R]):
    """标准化成员执行结果"""
    result: Optional[R] = Field(default=None, description="执行成功时的返回值")
    error: Optional[str] = Field(default=None, description="执行失败时的错误信息")
    member_index: int = Field(..., description="成员在 Team 中的索引")
    is_success: bool = Field(default=True, description="是否执行成功")

    @property
    def value(self) -> R:
        if not self.is_success:
            raise RuntimeError(f"Member execution failed: {self.error}")
        return self.result  # type: ignore


class Team:
    """
    多智能体协作组 (Parallel Execution Engine)
    """

    def __init__(
        self,
        members: List[Union[Agent, Callable]],
        name: str = "Team",
        max_concurrent: int = 0,
        return_full_output: bool = False,
        strategy: ExecutionStrategy = ExecutionStrategy.ALL,  # [新增] 策略
        input_mapper: Optional[Callable[[Any, int], Any]] = None, # [新增] 输入分片
    ):
        self.members = members
        self.name = name
        self.max_concurrent = max_concurrent
        self.return_full_output = return_full_output
        self.strategy = strategy
        self.input_mapper = input_mapper
        # [P0-1 修复] Race 模式下的线程安全 Lock
        self._winner_lock: Optional[anyio.Lock] = None

    # ========================= 接口协议 =========================

    async def __call__(self, context_or_input: Any) -> List[MemberResult]:
        return await self.run(context_or_input)

    async def run(self, context_or_input: Any) -> List[MemberResult]:
        """执行 Team 逻辑"""
        # 1. 解析原始输入
        raw_input = self._resolve_input(context_or_input)
        member_count = len(self.members)

        logger.info(
            "Team execution started",
            team=self.name,
            members=member_count,
            strategy=self.strategy
        )

        # 2. 准备分片输入 (Sharding Logic)
        inputs = []
        for i in range(member_count):
            if self.input_mapper:
                try:
                    # input_mapper(raw_input, member_index)
                    # 允许根据索引切分数据
                    val = self.input_mapper(raw_input, i)
                    inputs.append(val)
                except Exception as e:
                    logger.error(f"Input mapping failed for member {i}", error=str(e))
                    inputs.append(None) # 映射失败视为 None 或抛出，这里选择防御性 None
            else:
                inputs.append(raw_input)

        # 3. 根据策略分发
        if self.strategy == ExecutionStrategy.RACE:
            return await self._execute_race(inputs)
        else:
            return await self._execute_all(inputs)

    # ========================= 核心执行模式 =========================

    async def _execute_all(self, inputs: List[Any]) -> List[MemberResult]:
        """
        [默认模式] 等待所有成员完成
        """
        results: List[Optional[MemberResult]] = [None] * len(self.members)
        semaphore = anyio.Semaphore(self.max_concurrent) if self.max_concurrent > 0 else None

        async def _worker(idx: int, member: Any, inp: Any):
            if semaphore:
                await semaphore.acquire()
            try:
                res = await self._safe_execute_member(idx, member, inp)
                results[idx] = res
            finally:
                if semaphore:
                    semaphore.release()

        async with anyio.create_task_group() as tg:
            for i, member in enumerate(self.members):
                tg.start_soon(_worker, i, member, inputs[i])

        # 整理结果
        final_results = [r for r in results if r is not None]
        self._log_completion(final_results)
        return final_results

    async def _execute_race(self, inputs: List[Any]) -> List[MemberResult]:
        """
        [赛马模式] 返回最快成功的那个，取消其他
        """
        # 用于存储获胜者的容器 (list 是可变的，闭包可写)
        winner: List[MemberResult] = []

        # 初始化 winner lock（确保在并发场景中可用）
        # [P0-1 修复] 在调用之前创建 Lock，避免 None 导致的异常
        self._winner_lock = anyio.Lock()
        
        # 使用 CancelScope 实现“一人成功，全员取消”
        # 注意：这里需要在外部包裹 try-except 处理取消异常
        try:
            async with anyio.create_task_group() as tg:
                for i, member in enumerate(self.members):
                    
                    async def _racer(idx: int, mem: Any, inp: Any):
                        # 执行成员逻辑
                        res = await self._safe_execute_member(idx, mem, inp)
                        
                        # [P0-1 修复] 使用原子 Lock 保护
                        # 防止多个协程同时通过 "if not winner" 的检查
                        async with self._winner_lock:
                            if res.is_success and not winner:
                                winner.append(res)
                                # 核心：取消整个 TaskGroup 的 Scope
                                tg.cancel_scope.cancel()
                        
                    tg.start_soon(_racer, i, member, inputs[i])
                    
        except anyio.get_cancelled_exc_class():
            # 捕获取消异常是预期行为（因为我们主动 cancel 了）
            pass
        except Exception as e:
            logger.error("Race execution crashed", error=str(e))
        finally:
            # [P0-1 修复] 清理 Lock
            self._winner_lock = None

        if winner:
            logger.info(f"Team {self.name} Race won by member {winner[0].member_index}")
            return winner
        
        # [P0-2 修复] 失败时返回有意义的结果，而不是空列表
        logger.warning(f"Team {self.name} Race failed: no winner")
        return [
            MemberResult(
                member_index=i,
                error="Race failed - no successful member",
                is_success=False
            )
            for i in range(len(self.members))
        ]

    # ========================= 内部逻辑 =========================

    async def _safe_execute_member(self, idx: int, member: Any, inp: Any) -> MemberResult:
        """单成员执行封装（含异常处理与结果标准化）"""
        try:
            raw = await self._execute_member(member, inp)
            processed = self._process_result(raw)
            return MemberResult(member_index=idx, result=processed, is_success=True)
        except Exception as e:
            logger.error(f"Member {idx} failed", error=str(e))
            return MemberResult(member_index=idx, error=str(e), is_success=False)

    def _resolve_input(self, context_or_input: Any) -> Any:
        """从 Context 提取输入 (保留 WorkflowContext 的 Duck Typing)"""
        if (
            hasattr(context_or_input, "history")
            and hasattr(context_or_input, "input")
            and isinstance(getattr(context_or_input, "history", None), dict)
        ):
            ctx = context_or_input
            history = getattr(ctx, "history", {})
            state = getattr(ctx, "state", {})

            if "_next_input" in state:
                return state.pop("_next_input")
            elif "last_output" in history:
                return history["last_output"]
            else:
                return getattr(ctx, "input")
        return context_or_input

    async def _execute_member(self, member: Any, inp: Any) -> Any:
        if hasattr(member, "run"):
            return await member.run(inp)
        if callable(member):
            return await ensure_awaitable(member, inp)
        raise TypeError(f"Member {member} is not executable")

    def _process_result(self, result: Any) -> Any:
        if self.return_full_output:
            if isinstance(result, (BaseModel, AgentOutput, Message)):
                return result.model_dump()
            return result
        # 默认提取内容
        if isinstance(result, AgentOutput):
            return result.content
        if isinstance(result, Message):
            return result.content
        return result

    def _log_completion(self, results: List[MemberResult]):
        success = sum(1 for r in results if r.is_success)
        logger.info(
            "Team execution completed",
            team=self.name,
            success=success,
            total=len(results)
        )