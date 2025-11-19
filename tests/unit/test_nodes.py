# tests/unit/test_nodes.py
import asyncio
from gecko.compose.nodes import Condition

def test_condition_callable():
    cond = Condition(lambda ctx: ctx["flag"])
    assert asyncio.run(cond({"flag": True})) is True