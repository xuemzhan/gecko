# gecko/compose/team.py
"""
Team 多智能体并行引擎 (Enhanced)

优化日志：
- [Feat] 增加 ExecutionStrategy 枚举 (ALL, RACE)
- [Feat] 增加 input_mapper 支持输入分片 (Sharding)
- [Perf] 实现 Race 模式下的快速返回与任务取消
- [Refactor] 保持 MemberResult 标准化结构

P0 修复聚焦：
- [P0-1] Race winner atomicity：并发下只能产生唯一 winner（原子性）
- [P0-2] Race 全失败时返回结构化 MemberResult 列表（而不是空列表）
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
    """Team 执行策略"""
    ALL = "all"     # 全员执行，聚合结果
    RACE = "race"   # 赛马：最快成功者获胜，其他取消


class MemberResult(BaseModel, Generic[R]):
    """
    单个成员的执行结果（标准化结构）

    设计要点：
    - 无论成员执行成功/失败，都返回 MemberResult，便于上层统一聚合/观测
    - value 属性在失败时抛错，方便调用方用“显式错误”方式处理
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

    典型用法：
    - ALL：并行跑所有成员，返回所有结果（部分失败也会被结构化保留）
    - RACE：并行赛跑，返回最快成功者（节省 token/时间），其余取消
    """

    def __init__(
        self,
        members: List[Union[Agent, Callable[..., Any]]],
        name: str = "Team",
        max_concurrent: int = 0,
        return_full_output: bool = False,
        strategy: ExecutionStrategy = ExecutionStrategy.ALL,
        input_mapper: Optional[Callable[[Any, int], Any]] = None,  # [新增] 输入分片
    ):
        self.members = members
        self.name = name
        self.max_concurrent = max_concurrent
        self.return_full_output = return_full_output
        self.strategy = strategy
        self.input_mapper = input_mapper
        # [P0-1] Race 模式的获胜者竞争需要原子性保护：这里不在实例上持久化 Lock，
        # 而是在每次 _execute_race() 内部创建局部 Lock（避免同一个 Team 实例并发 run 时互相干扰）。

    # ========================= 接口协议 =========================

    async def __call__(self, context_or_input: Any, timeout: Optional[float] = None) -> List[MemberResult]:
        """允许 Team(...) 直接被 await 调用"""
        return await self.run(context_or_input, timeout=timeout)

    async def run(self, context_or_input: Any, timeout: Optional[float] = None) -> List[MemberResult]:
        """
        执行 Team

        输入可以是：
        - 任意对象（作为原始 input）
        - WorkflowContext（Duck typing：从 context.input 或 last_output 提取）
        """
        raw_input = self._resolve_input(context_or_input)

        # 1) 输入分片：为每个成员准备独立输入，避免成员间相互污染
        inputs: List[Any] = []
        for i in range(len(self.members)):
            if self.input_mapper:
                try:
                    val = self.input_mapper(raw_input, i)
                    inputs.append(val)
                except Exception as e:
                    # 防御性：分片失败不应直接炸掉整个 Team
                    logger.error(f"Input mapping failed for member {i}", error=str(e))
                    inputs.append(None)
            else:
                inputs.append(raw_input)

        # 2) 根据策略分发
        if self.strategy == ExecutionStrategy.RACE:
            return await self._execute_race(inputs, timeout)
        else:
            return await self._execute_all(inputs, timeout)

    # ========================= 核心执行模式 =========================

    async def _execute_all(self, inputs: List[Any], timeout: Optional[float] = None) -> List[MemberResult]:
        """
        [全员模式] 并行执行所有成员，返回每个成员的 MemberResult

        关键工程点：
        - max_concurrent > 0 时启用信号量限流，避免成员过多导致资源争用
        - 成员异常被 _safe_execute_member 捕获并转为 MemberResult
        """
        from contextlib import nullcontext

        results: List[Optional[MemberResult]] = [None] * len(self.members)
        semaphore = anyio.Semaphore(self.max_concurrent) if self.max_concurrent and self.max_concurrent > 0 else None
        timeout_cm = anyio.move_on_after(timeout) if timeout is not None else nullcontext()

        async def _worker(idx: int, member: Any, inp: Any):
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

        # 超时处理：move_on_after 取消后 results 可能仍有 None，用失败结果补齐
        if timeout is not None and getattr(timeout_cm, "cancel_called", False):  # type: ignore[attr-defined]
            logger.warning(f"Team {self.name} ALL timed out after {timeout}s")
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = MemberResult(member_index=i, error=f"Timed out after {timeout}s", is_success=False)

        return [r or MemberResult(member_index=i, error="Unknown error", is_success=False) for i, r in enumerate(results)]

    async def _execute_race(self, inputs: List[Any], timeout: Optional[float] = None) -> List[MemberResult]:
        """
        [赛马模式 / Race]
        - 目标：**返回最快成功**的成员结果，并尽快取消其他成员的执行，节省成本与时间。
        - 语义：
          1) 任意成员成功 -> 立刻“宣告获胜”，取消整个 TaskGroup（其余协程会收到取消信号）。
          2) 全部失败 -> 返回每个成员的失败 MemberResult（而不是空列表，便于上层定位问题）。
          3) 超时 -> 触发取消，并抛出超时异常（保持行为清晰）。

        [P0-1 修复点：获胜者原子性]
        - 并发下可能出现多个协程同时满足 `if res.is_success and not winner` 的竞态。
        - 必须引入 Lock 做“检查+写入+取消”的原子区段，确保只会选出**一个** winner。
        """
        from contextlib import nullcontext

        # winner 容器（list 可变，便于闭包写入；约定最多只写入 1 个元素）
        winner: List[MemberResult] = []

        # [关键] 局部 Lock：只保护本次 run 的竞争，不污染实例状态
        winner_lock = anyio.Lock()

        # 超时管理器：anyio.move_on_after 超时后不会抛异常，而是 cancel_scope.cancel_called=True
        timeout_cm = anyio.move_on_after(timeout) if timeout is not None else nullcontext()

        # 使用 TaskGroup + cancel_scope 实现“一人成功，全员取消”
        try:
            with timeout_cm:
                async with anyio.create_task_group() as tg:
                    for i, member in enumerate(self.members):

                        async def _racer(idx: int, mem: Any, inp: Any):
                            # 1) 执行成员逻辑（内部已做异常捕获，保证返回 MemberResult）
                            res = await self._safe_execute_member(idx, mem, inp)

                            # 2) 原子性竞争：确保只会写入一次 winner，并且只 cancel 一次
                            if not res.is_success:
                                return

                            async with winner_lock:
                                # 再次检查（双重校验）——只有首个成功者可写入 winner
                                if winner:
                                    return
                                winner.append(res)
                                # 取消整个 TaskGroup：其余协程会被尽快取消
                                tg.cancel_scope.cancel()

                        # 注意：start_soon 参数必须显式传入，避免闭包捕获循环变量
                        tg.start_soon(_racer, i, member, inputs[i])

        except anyio.get_cancelled_exc_class():
            # 取消是预期行为：可能来自“winner 触发取消”或“超时触发取消”
            pass
        except Exception as e:
            # 兜底：不应因个别未捕获异常导致整个 Team 无结果
            logger.exception("Race execution crashed", team=self.name, error=str(e))

        # 1) 有 winner：直接返回（与历史版本兼容：返回 List[MemberResult]，长度=1）
        if winner:
            logger.info(f"Team {self.name} Race won by member {winner[0].member_index}")
            return winner

        # 2) 超时：move_on_after 的语义是取消 scope，不抛 TimeoutError；因此用 cancel_called 判断
        if timeout is not None and getattr(timeout_cm, "cancel_called", False):  # type: ignore[attr-defined]
            raise RuntimeError(f"Team {self.name} race timed out after {timeout}s")

        # 3) 全部失败：返回每个成员的失败结果（P0-2）
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
        """
        单成员执行封装（含异常处理与结果标准化）

        原则：
        - 成员异常不外抛，统一收敛为 MemberResult(is_success=False)
        - 保留 error 文本与 logger.exception，便于线上排障
        """
        try:
            raw = await self._execute_member(member, inp)
            processed = self._process_result(raw)
            return MemberResult(member_index=idx, result=processed, is_success=True)
        except Exception as e:
            logger.error(f"Member {idx} failed", error=str(e))
            logger.exception(f"Member {idx} execution failed", member_index=idx, error=str(e))
            return MemberResult(member_index=idx, error=str(e), is_success=False)

    def _resolve_input(self, context_or_input: Any) -> Any:
        """
        从 Context 提取输入（保留 WorkflowContext 的 Duck Typing）

        关键点：
        1) 只要 state 中存在 "_next_input" 这个 key，就必须优先使用它
        - 即使值是 0 / False / "" 这样的“假值”，也要保留
        - 不能用 state.get("_next_input") 这种真值判断
        2) 如果没有 _next_input，则使用 history["last_output"]
        - 同理，last_output 也可能是 False，必须用 key 是否存在判断
        3) 都没有则回退到 ctx.input
        """
        if (
            hasattr(context_or_input, "history")
            and hasattr(context_or_input, "input")
            and isinstance(getattr(context_or_input, "history", None), dict)
        ):
            ctx = context_or_input
            history = getattr(ctx, "history", {}) or {}
            state = getattr(ctx, "state", {}) or {}

            # 修复：用 `in` 判断 key 存在性，保留 0/False 等假值
            if "_next_input" in state:
                return state.pop("_next_input")

            # 修复：同样用 `in`，确保 last_output=False 仍然被使用
            if "last_output" in history:
                return history["last_output"]

            return getattr(ctx, "input")

        return context_or_input
    
    async def _execute_member(self, member: Any, inp: Any) -> Any:
        """
        执行单个成员。

        兼容三类成员：
        1) Agent-like（duck typing）：只要有 run 方法即可（MockAgent / Agent / 自定义 Agent）
        2) 普通可调用：sync/async function 或实现了 __call__ 的对象
        3) 非法成员：既没有 run，也不可调用 -> 抛出明确的 not executable
        """

        # 1) ✅ Agent-like：优先识别 run()（修复 MockAgent 被当 callable 的回归）
        run_attr = getattr(member, "run", None)
        if callable(run_attr):
            # 用 ensure_awaitable 做兜底：兼容 sync run / async run
            return await ensure_awaitable(run_attr, inp)

        # 2) 普通可调用：sync/async 均可
        if callable(member):
            return await ensure_awaitable(member, inp)

        # 3) 非法类型：给出稳定、可断言的错误信息
        raise TypeError(f"Member {member} is not executable")


    def _process_result(self, raw: Any) -> Any:
        """
        结果归一化

        - return_full_output=True：尽量保留 AgentOutput（或其等价结构）
        - 否则：优先提取 AgentOutput.content，给上层更“干净”的文本/结构
        """
        if self.return_full_output:
            return raw

        # AgentOutput：默认抽取 content
        if isinstance(raw, AgentOutput):
            return raw.content

        return raw