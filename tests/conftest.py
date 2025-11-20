# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from gecko.core.events import EventBus
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox

@pytest.fixture
def event_loop():
    """创建测试用的 EventLoop"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm():
    """Mock LLM 对象"""
    llm = MagicMock()
    llm.acompletion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test Response", tool_calls=None))]
    ))
    return llm

@pytest.fixture
def memory():
    return TokenMemory(session_id="test_session", model_name="gpt-3.5-turbo")

@pytest.fixture
def toolbox():
    return ToolBox()

@pytest.fixture
def event_bus():
    return EventBus()