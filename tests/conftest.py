# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.core.output import AgentOutput

class MockModel:
    async def acompletion(self, messages, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = f"Mock response for: {messages[-1]['content']}"
        return mock_response

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def simple_agent(mock_model):
    return AgentBuilder().with_model(mock_model).build()

@pytest.fixture
def workflow():
    from gecko.compose import Workflow
    return Workflow()