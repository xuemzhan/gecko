# tests/conftest.py
import os
import warnings
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from gecko.core.events import EventBus
from gecko.core.memory import TokenMemory
from gecko.core.protocols import ModelProtocol
from gecko.core.toolbox import ToolBox

def pytest_configure(config):
    """
    Pytest 配置钩子
    """
    # 1. 屏蔽 Pydantic 序列化警告
    # 原因: LiteLLM 内部处理智谱/Ollama 等非标准 OpenAI 响应时，会触发 Pydantic v2 的序列化警告。
    # 既然 Gecko 使用 Adapter 手动提取数据，这些上游警告是无关噪音，可以安全忽略。
    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        message="Pydantic serializer warnings"
    )
    
    # 2. 屏蔽 LiteLLM 的一些 verbose 输出
    os.environ["LITELLM_LOG"] = "ERROR"

@pytest.fixture(autouse=True)
def env_setup():
    """自动设置测试环境"""
    # 确保测试期间不会意外读取 .env 造成依赖
    # 可以在这里 mock 环境变量
    pass

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
def model():
    model = MagicMock(spec=ModelProtocol) 
    model.acompletion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test Response", tool_calls=None))]
    ))
    return model

@pytest.fixture
def memory():
    return TokenMemory(session_id="test_session", model_name="gpt-3.5-turbo")

@pytest.fixture
def toolbox():
    return ToolBox()

@pytest.fixture
def event_bus():
    return EventBus()