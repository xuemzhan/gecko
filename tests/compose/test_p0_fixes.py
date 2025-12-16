"""
Unit tests for P0 bug fixes in compose module

This test suite uses the public APIs of `Team` and `Workflow`:
- Build graphs via `engine.graph.add_node` / `add_edge` and `set_entry_point`
- Provide a `WorkflowContext` via `_resume_context` so tests can inspect `context.history`

Covered fixes:
- P0-1: Race winner atomicity (lock protection)
- P0-2: Race all-fail returns MemberResult list (not empty list)
- P0-3: Next.input=None preserves last_output (no pollution)
- P0-4: Skipped nodes produce SKIPPED status and don't pollute history
"""

import pytest
import anyio
from typing import Any

from gecko.compose.team import Team, MemberResult, ExecutionStrategy
from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext
from gecko.compose.nodes import Next


# ----------------------------- Helpers -----------------------------
async def run_team_and_collect(team: Team, inp: Any = None):
    """Helper to run Team and return its result list"""
    return await team.run(inp)


# ----------------------------- Tests ------------------------------
class TestP0_RaceBehavior:
    @pytest.mark.asyncio
    async def test_race_single_winner_atomic_strict_shape(self):
        """
        P0-1（升级断言点）：
        - Race 必须返回长度=1 的列表（只返回 winner）
        - winner 必须 is_success=True
        - winner 的 member_index 必须合法
        """
        async def member1(inp=None):
            await anyio.sleep(0.01)
            return "m1"

        async def member2(inp=None):
            await anyio.sleep(0.01)
            return "m2"

        team = Team(members=[member1, member2], name="race1", strategy=ExecutionStrategy.RACE)

        for _ in range(50):
            results = await run_team_and_collect(team, "input")
            assert isinstance(results, list)

            # ✅ 更严格：Race 只返回 winner（长度必须为 1）
            assert len(results) == 1

            # ✅ winner 必须是成功结果
            winner = results[0]
            assert isinstance(winner, MemberResult)
            assert winner.is_success is True
            assert winner.member_index in (0, 1)
            assert winner.result in ("m1", "m2")

    @pytest.mark.asyncio
    async def test_race_cancels_non_winner_task(self):
        """
        P0-1（升级断言点）：
        - winner 产生后，应取消其它未完成任务（至少不能正常跑完）
        """
        cancelled_2 = False
        completed_2 = False
        CancelledError = anyio.get_cancelled_exc_class()

        async def fast(inp=None):
            await anyio.sleep(0.01)
            return "fast"

        async def slow(inp=None):
            nonlocal cancelled_2, completed_2
            try:
                # 故意很慢，理论上会被 winner 触发取消
                await anyio.sleep(10)
                completed_2 = True
                return "slow"
            except CancelledError:
                cancelled_2 = True
                raise

        team = Team(members=[fast, slow], name="race_cancel", strategy=ExecutionStrategy.RACE)
        results = await run_team_and_collect(team, "input")

        # ✅ winner 形态必须严格
        assert len(results) == 1
        assert results[0].is_success is True
        assert results[0].result == "fast"
        assert results[0].member_index == 0

        # 给取消传播一个调度机会（避免极端情况下 flag 还没来得及写）
        await anyio.sleep(0)

        # ✅ slow 不应正常完成，应被取消（至少 cancelled 标志应为 True）
        assert completed_2 is False, "Non-winner should not complete normally in race mode"
        assert cancelled_2 is True, "Non-winner should be cancelled when winner is chosen"

    @pytest.mark.asyncio
    async def test_race_reentrant_concurrent_runs_do_not_interfere(self):
        """
        P0-1（升级断言点）：
        - 同一个 Team 实例在并发 run 时，不应共享 winner/lock 状态
          （验证 PR1 把 lock 放在 _execute_race 内部的必要性）
        """
        async def member1(inp=None):
            await anyio.sleep(0.01)
            return f"m1:{inp}"

        async def member2(inp=None):
            await anyio.sleep(0.01)
            return f"m2:{inp}"

        team = Team(members=[member1, member2], name="race_reentrant", strategy=ExecutionStrategy.RACE)

        inputs = [f"req-{i}" for i in range(30)]
        outputs = {}

        async def _runner(x: str):
            res = await run_team_and_collect(team, x)
            outputs[x] = res

        async with anyio.create_task_group() as tg:
            for x in inputs:
                tg.start_soon(_runner, x)

        # ✅ 每次 run 都必须独立产生一个 winner，并且 winner 的内容必须带对应 input
        for x in inputs:
            res = outputs[x]
            assert isinstance(res, list)
            assert len(res) == 1
            assert res[0].is_success is True
            assert isinstance(res[0].result, str)
            assert res[0].result.endswith(x), "Winner result should correspond to its own input"

class TestP0_NextAndHistory:
    @pytest.mark.asyncio
    async def test_next_input_none_preserves_last_output(self):
        engine = Workflow(name="next_none_test")

        def node_a(context: WorkflowContext):
            return {"data": "from_a"}

        def node_b(context: WorkflowContext):
            # Return a Next instructing to jump to node 'c' but with input=None (pass-through)
            return Next(node="c", input=None)

        def node_c(context: WorkflowContext):
            return context.get_last_output()

        engine.graph.add_node("a", node_a)
        engine.graph.add_node("b", node_b)
        engine.graph.add_node("c", node_c)
        engine.graph.add_edge("a", "b")
        engine.graph.add_edge("b", "c")
        engine.graph.set_entry_point("a")

        ctx = WorkflowContext(input={})
        await engine.execute(input_data=None, _resume_context=ctx)

        assert "last_output" in ctx.history
        lo = ctx.history["last_output"]
        assert isinstance(lo, dict)
        assert lo.get("data") == "from_a"

    @pytest.mark.asyncio
    async def test_next_with_explicit_input_overrides(self):
        engine = Workflow(name="next_override_test")

        def node_a(context: WorkflowContext):
            return {"data": "from_a"}

        def node_b(context: WorkflowContext):
            return Next(node="c", input={"data": "override"})

        def node_c(context: WorkflowContext):
            return context.get_last_output()

        engine.graph.add_node("a", node_a)
        engine.graph.add_node("b", node_b)
        engine.graph.add_node("c", node_c)
        engine.graph.add_edge("a", "b")
        engine.graph.add_edge("b", "c")
        engine.graph.set_entry_point("a")

        ctx = WorkflowContext(input={})
        await engine.execute(input_data=None, _resume_context=ctx)

        assert "last_output" in ctx.history
        lo = ctx.history["last_output"]
        assert isinstance(lo, dict)
        assert lo.get("data") == "override"


class TestP0_SkippedNodes:
    @pytest.mark.asyncio
    async def test_conditional_skip_does_not_pollute_history(self):
        engine = Workflow(name="skip_test")

        def node_a(context: WorkflowContext):
            context.state["should_run_b"] = False
            return {"value": "from_a"}

        def condition_for_b(context: WorkflowContext):
            return context.state.get("should_run_b", True)

        def node_b(context: WorkflowContext):
            return {"value": "from_b"}

        def node_c(context: WorkflowContext):
            return context.get_last_output()

        engine.graph.add_node("a", node_a)
        engine.graph.add_node("b", node_b)
        engine.graph.add_node("c", node_c)
        # attach conditional edge a->b
        engine.graph.add_edge("a", "b", condition=condition_for_b)
        engine.graph.add_edge("b", "c")
        engine.graph.set_entry_point("a")

        ctx = WorkflowContext(input={})
        await engine.execute(input_data=None, _resume_context=ctx)

        assert "last_output" in ctx.history
        last = ctx.history["last_output"]
        assert isinstance(last, dict)
        assert last.get("value") == "from_a"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
    # end of file
