"""
Tests for Copy-On-Write (COW) behavior in Workflow engine

Goals:
- Verify each node receives a shallow-copied `state` (different id than main context.state)
- Verify `history` is shared (same id) to avoid deep-copy overhead
- Verify node state changes are merged back into main `context.state` after the layer
"""

import pytest
import anyio

from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext


@pytest.mark.asyncio
async def test_cow_state_is_per_node_and_history_shared():
    engine = Workflow(name="cow_test")

    observed_state_ids = []
    observed_history_ids = []

    def start(ctx: WorkflowContext):
        # initial start node
        return {"init": True}

    def node_a(ctx: WorkflowContext):
        # record ids
        observed_state_ids.append(id(ctx.state))
        observed_history_ids.append(id(ctx.history))
        # mutate node-local state
        ctx.state["a"] = 1
        return {"a": 1}

    async def node_b(ctx: WorkflowContext):
        # small sleep to increase chance of parallel overlap
        await anyio.sleep(0)
        observed_state_ids.append(id(ctx.state))
        observed_history_ids.append(id(ctx.history))
        ctx.state["b"] = 2
        return {"b": 2}

    # wire graph so a and b are in same parallel layer after start
    engine.graph.add_node("start", start)
    engine.graph.add_node("a", node_a)
    engine.graph.add_node("b", node_b)
    engine.graph.add_edge("start", "a")
    engine.graph.add_edge("start", "b")
    engine.graph.set_entry_point("start")

    ctx = WorkflowContext(input={})
    await engine.execute(input_data=None, _resume_context=ctx)

    # Each node should have a distinct state object (COW)
    main_state_id = id(ctx.state)
    assert all(sid != main_state_id for sid in observed_state_ids), "Node state should be a shallow copy (COW)"

    # History should be shared to avoid deep copy overhead
    main_history_id = id(ctx.history)
    assert all(hid == main_history_id for hid in observed_history_ids), "History should be shared (no deep copy)"

    # After merge, main context.state should contain merged keys from nodes
    assert ctx.state.get("a") == 1
    assert ctx.state.get("b") == 2
