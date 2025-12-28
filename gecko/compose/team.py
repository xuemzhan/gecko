# gecko/compose/team.py
"""
Team 多智能体并行引擎 (Enhanced v0.5)

优化日志：
- [Fix P0-1] Race winner atomicity：并发下只能产生唯一 winner（原子性）
- [Fix P0-2] Race 全失败时返回结构化 MemberResult 列表（而不是空列表）
- [Fix P1-4] 输入分片 (Sharding) 容错：mapper 失败时隔离故障，不影响其他成员
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Generic, List, Optional, TypeVar, Union, TYPE_CHECKING, Set

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
    """Team 执行策略"""
    ALL = "all"     # 全员执行，聚合结果
    RACE = "race"   # 赛马：最快成功者获胜，其他取消


class MemberResult(BaseModel, Generic[R]):
    """
    单个成员的执行结果（标准化结构）
    """
    member_index: int = Field(..., description="成员在 Team 中的索引")
    result: Optional[R] = Field(default=None, description="成功时的结果")
    error: Optional[str] = Field(default=None, description="失败时的错误信息")
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
        members: List[Union[Agent, Callable[..., Any]]],
        name: str = "Team",
        max_concurrent: int = 0,
        return_full_output: bool = False,
        strategy: ExecutionStrategy = ExecutionStrategy.ALL,
        input_mapper: Optional[Callable[[Any, int], Any]] = None,
    ):
        self.members = members
        self.name = name
        self.max_concurrent = max_concurrent
        self.return_full_output = return_full_output
        self.strategy = strategy
        self.input_mapper = input_mapper

    # ========================= 接口协议 =========================

    async def __call__(self, context_or_input: Any, timeout: Optional[float] = None) -> List[MemberResult]:
        """允许 Team(...) 直接被 await 调用"""
        return await self.run(context_or_input, timeout=timeout)

    async def run(self, context_or_input: Any, timeout: Optional[float] = None) -> List[MemberResult]:
        """
        执行 Team
        
        [Fix P1-4] 增强了输入分片的鲁棒性，记录 failed_indices
        """
        raw_input = self._resolve_input(context_or_input)

        # 1) 输入分片：为每个成员准备独立输入
        inputs: List[Any] = []
        failed_indices: Set[int] = set()

        for i in range(len(self.members)):
            if self.input_mapper:
                try:
                    val = self.input_mapper(raw_input, i)
                    inputs.append(val)
                except Exception as e:
                    # [Fix P1-4] 防御性：分片失败记录索引，后续跳过执行
                    logger.error(f"Input mapping failed for member {i}", error=str(e))
                    inputs.append(None) # 占位
                    failed_indices.add(i)
            else:
                inputs.append(raw_input)

        # 2) 根据策略分发
        if self.strategy == ExecutionStrategy.RACE:
            return await self._execute_race(inputs, timeout, failed_indices)
        else:
            return await self._execute_all(inputs, timeout, failed_indices)

    # ========================= 核心执行模式 =========================

    async def _execute_all(
        self, 
        inputs: List[Any], 
        timeout: Optional[float] = None,
        failed_indices: Set[int] = set()
    ) -> List[MemberResult]:
        """
        [全员模式] 并行执行所有成员
        """
        from contextlib import nullcontext

        results: List[Optional[MemberResult]] = [None] * len(self.members)
        semaphore = anyio.Semaphore(self.max_concurrent) if self.max_concurrent and self.max_concurrent > 0 else None
        timeout_cm = anyio.move_on_after(timeout) if timeout is not None else nullcontext()

        async def _worker(idx: int, member: Any, inp: Any):
            # [Fix P1-4] 如果输入映射阶段已失败，直接返回失败结果，不执行成员
            if idx in failed_indices:
                results[idx] = MemberResult(
                    member_index=idx, 
                    error="Input mapping failed", 
                    is_success=False
                )
                return

            if semaphore:
                await semaphore.acquire()
            try:
                res = await self._safe_execute_member(idx, member, inp)
                results[idx] = res
            finally:
                if semaphore:
                    semaphore.release()

        with timeout_cm:
            async with anyio.create_task_group() as tg:
                for i, member in enumerate(self.members):
                    tg.start_soon(_worker, i, member, inputs[i])

        # 超时处理
        if timeout is not None and getattr(timeout_cm, "cancel_called", False):  # type: ignore[attr-defined]
            logger.warning(f"Team {self.name} ALL timed out after {timeout}s")
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = MemberResult(member_index=i, error=f"Timed out after {timeout}s", is_success=False)

        return [r or MemberResult(member_index=i, error="Unknown error", is_success=False) for i, r in enumerate(results)]

    async def _execute_race(
        self, 
        inputs: List[Any], 
        timeout: Optional[float] = None,
        failed_indices: Set[int] = set()
    ) -> List[MemberResult]:
        """
        [赛马模式 / Race]
        """
        from contextlib import nullcontext

        winner: List[MemberResult] = []
        winner_lock = anyio.Lock()
        timeout_cm = anyio.move_on_after(timeout) if timeout is not None else nullcontext()

        try:
            with timeout_cm:
                async with anyio.create_task_group() as tg:
                    for i, member in enumerate(self.members):

                        async def _racer(idx: int, mem: Any, inp: Any):
                            # [Fix P1-4] 映射失败不参与竞争
                            if idx in failed_indices:
                                return

                            # 1) 执行成员逻辑
                            res = await self._safe_execute_member(idx, mem, inp)

                            # 2) 原子性竞争
                            if not res.is_success:
                                return

                            async with winner_lock:
                                if winner:
                                    return
                                winner.append(res)
                                tg.cancel_scope.cancel()

                        tg.start_soon(_racer, i, member, inputs[i])

        except anyio.get_cancelled_exc_class():
            pass
        except Exception as e:
            logger.exception("Race execution crashed", team=self.name, error=str(e))

        # 1) 有 winner
        if winner:
            logger.info(f"Team {self.name} Race won by member {winner[0].member_index}")
            return winner

        # 2) 超时
        if timeout is not None and getattr(timeout_cm, "cancel_called", False):  # type: ignore[attr-defined]
            raise RuntimeError(f"Team {self.name} race timed out after {timeout}s")

        # 3) 全部失败
        logger.warning(f"Team {self.name} Race failed: no winner")
        return [
            MemberResult(
                member_index=i,
                error="Input mapping failed" if i in failed_indices else "Race failed - no successful member",
                is_success=False
            )
            for i in range(len(self.members))
        ]

    # ========================= 内部逻辑 =========================

    async def _safe_execute_member(self, idx: int, member: Any, inp: Any) -> MemberResult:
        """单成员执行封装"""
        try:
            raw = await self._execute_member(member, inp)
            processed = self._process_result(raw)
            return MemberResult(member_index=idx, result=processed, is_success=True)
        except Exception as e:
            logger.error(f"Member {idx} failed", error=str(e))
            logger.exception(f"Member {idx} execution failed", member_index=idx, error=str(e))
            return MemberResult(member_index=idx, error=str(e), is_success=False)

    def _resolve_input(self, context_or_input: Any) -> Any:
        """从 Context 提取输入"""
        if (
            hasattr(context_or_input, "history")
            and hasattr(context_or_input, "input")
            and isinstance(getattr(context_or_input, "history", None), dict)
        ):
            ctx = context_or_input
            history = getattr(ctx, "history", {}) or {}
            state = getattr(ctx, "state", {}) or {}

            if "_next_input" in state:
                return state.pop("_next_input")

            if "last_output" in history:
                return history["last_output"]

            return getattr(ctx, "input")

        return context_or_input
    
    async def _execute_member(self, member: Any, inp: Any) -> Any:
        """执行单个成员"""
        run_attr = getattr(member, "run", None)
        if callable(run_attr):
            return await ensure_awaitable(run_attr, inp)

        if callable(member):
            return await ensure_awaitable(member, inp)

        raise TypeError(f"Member {member} is not executable")


    def _process_result(self, raw: Any) -> Any:
        """结果归一化"""
        if self.return_full_output:
            return raw

        if isinstance(raw, AgentOutput):
            return raw.content

        return raw