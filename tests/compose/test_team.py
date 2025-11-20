# tests/compose/test_team.py
import pytest
from unittest.mock import AsyncMock
from gecko.compose.team import Team
from gecko.core.agent import Agent
from gecko.core.output import AgentOutput

@pytest.mark.asyncio
async def test_team_execution():
    # Mock Agents
    agent1 = AsyncMock(spec=Agent)
    agent1.run.return_value = AgentOutput(content="Agent 1")
    
    agent2 = AsyncMock(spec=Agent)
    agent2.run.return_value = AgentOutput(content="Agent 2")
    
    team = Team(members=[agent1, agent2])
    
    # Test __call__ interface
    results = await team("start")
    
    assert len(results) == 2
    assert "Agent 1" in results
    assert "Agent 2" in results
    
    # Verify parallel execution implies both were called
    assert agent1.run.called
    assert agent2.run.called