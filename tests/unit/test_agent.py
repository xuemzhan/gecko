# tests/unit/test_agent.py
from unittest.mock import AsyncMock
import pytest
import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.exceptions import AgentError
from gecko.core.message import Message
from gecko.core.output import AgentOutput

@pytest.fixture
def simple_agent(mock_model):
    # 注入工具避免 fallback
    from gecko.plugins.tools.calculator import CalculatorTool
    return AgentBuilder()\
        .with_model(mock_model)\
        .with_tools([CalculatorTool()])\
        .build()

@pytest.mark.asyncio
async def test_agent_run(simple_agent):
    messages = [Message(role="user", content="hello")]
    output = await simple_agent.run(messages)
    assert isinstance(output, AgentOutput)
    assert "hello" in output.content

@pytest.mark.asyncio
async def test_agent_error_handling(simple_agent, mocker):
    simple_agent.model.acompletion = AsyncMock(side_effect=Exception("API down"))
    messages = [Message(role="user", content="test")]
    with pytest.raises(AgentError):
        await simple_agent.run(messages)