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
    async def test_race_single_winner_atomic(self):
        async def member1(inp=None):
            await anyio.sleep(0.01)
            return "m1"

        async def member2(inp=None):
            await anyio.sleep(0.01)
            return "m2"

        team = Team(members=[member1, member2], name="race1", strategy=ExecutionStrategy.RACE)

        for _ in range(5):
            results = await run_team_and_collect(team, "input")
            assert isinstance(results, list)
            success = [r for r in results if r.is_success]
            assert len(success) == 1

    @pytest.mark.asyncio
    async def test_race_all_fail_returns_memberresults(self):
        async def fail1(inp=None):
            raise RuntimeError("fail1")

        async def fail2(inp=None):
            raise ValueError("fail2")

        team = Team(members=[fail1, fail2], name="race_fail", strategy=ExecutionStrategy.RACE)
        results = await run_team_and_collect(team, "input")

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MemberResult)
            assert not r.is_success
            assert r.error is not None


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
