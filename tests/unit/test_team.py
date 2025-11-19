# tests/unit/test_team.py
import pytest
import asyncio
from gecko.compose import Team
from gecko.core.message import Message

@pytest.mark.asyncio
async def test_team_parallel(simple_agent):
    team = Team(members=[simple_agent, simple_agent])
    context = {"input": "test team"}
    result = await team.execute(context)
    assert len(result) == 2
    assert all(isinstance(r, str) for r in result)  # 修复断言