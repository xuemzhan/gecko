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

from gecko.plugins.tools.registry import ToolRegistry

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
def clean_tool_registry():
    """
    [新增] 自动清理工具注册表
    确保每个测试用例都在一个干净的注册表状态下运行，
    避免 Test A 注册的工具影响 Test B。
    """
    # 备份当前状态
    original_registry = ToolRegistry._registry.copy()
    yield
    # 还原状态
    ToolRegistry._registry = original_registry

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
    """
    [Critical Fix] Mock LLM 对象
    必须返回对象，且实现 count_tokens 以通过 ModelProtocol 检查
    """
    # 1. 使用 spec 自动模拟协议特征
    llm = MagicMock(spec=ModelProtocol)
    
    # 2. 模拟 acompletion (异步推理)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test Response", tool_calls=None))
    ]
    # 确保 model_dump 可调用 (Agent 内部会调用)
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant", 
        "content": "Test Response"
    }
    
    llm.acompletion = AsyncMock(return_value=mock_response)
    
    # 3. [关键] 模拟 count_tokens (同步计数)
    # 这是修复 "model 必须实现 ModelProtocol" 错误的核心
    llm.count_tokens = MagicMock(return_value=10)
    
    # 4. [关键] 必须返回对象，否则测试中收到 None
    return llm

@pytest.fixture
def model(mock_llm):
    """
    model fixture 是 mock_llm 的别名，用于某些特定测试
    """
    return mock_llm

@pytest.fixture
def memory(mock_llm):
    """
    Memory Fixture
    [Fix] 注入 model_driver 以支持新的计数逻辑
    """
    return TokenMemory(
        session_id="test_session", 
        max_tokens=4000, 
        model_name="gpt-3.5-turbo",
        model_driver=mock_llm  # 注入 Mock 驱动
    )

# @pytest.fixture
# def memory():
#     return TokenMemory(session_id="test_session", model_name="gpt-3.5-turbo")

@pytest.fixture
def toolbox():
    return ToolBox()

@pytest.fixture
def event_bus():
    return EventBus()