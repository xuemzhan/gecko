# tests/unit/test_workflow.py
import pytest
import asyncio
from gecko.compose import Workflow, step, Loop, Condition

@step("double")
async def double(context):
    return context["input"] * 2

@step("add_one")
def add_one(context):
    return context["double"] + 1

def test_simple_dag(workflow):
    workflow.add_node("double", double)
    workflow.add_node("add_one", add_one)
    workflow.add_edge("start", "double")
    workflow.add_edge("double", "add_one")
    workflow.add_edge("add_one", "end")

    result = asyncio.run(workflow.execute(10))
    assert result == 21

def test_loop_node(workflow):
    counter = 0
    async def increment(context):
        nonlocal counter
        counter += 1
        return counter

    @step("inc")
    async def inc_node(context):
        return await increment(context)

    loop = Loop(
        body=Workflow().add_node("inc", inc_node).add_edge("start", "inc").add_edge("inc", "end"),
        condition=lambda ctx: ctx.get("inc", 0) < 3
    )

    workflow.add_node("loop", loop)
    workflow.add_edge("start", "loop")
    workflow.add_edge("loop", "end")

    result = asyncio.run(workflow.execute({}))
    assert result["inc"] == 3