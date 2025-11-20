# tests/integration/test_phase1_integration.py
"""
Phase 1 改进版集成测试
验证所有组件协同工作
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from gecko.config import GeckoSettings
from gecko.core.agent import Agent
from gecko.core.message import Message
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.logging import get_logger

@pytest.fixture
def test_config():
    """测试配置（独立实例）"""
    return GeckoSettings(
        log_level="DEBUG",
        max_turns=3,
        enable_cache=False
    )

@pytest.fixture
def mock_model_complete():
    """完整的 Mock 模型"""
    model = AsyncMock()
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Hello from mock LLM",
        "tool_calls": None
    }
    
    model.acompletion.return_value = mock_response
    return model

@pytest.mark.asyncio
async def test_full_agent_flow(mock_model_complete, test_config):
    """测试完整的 Agent 流程"""
    
    # 1. 创建组件
    toolbox = ToolBox([])
    memory = TokenMemory(
        session_id="test_session",
        max_tokens=test_config.max_context_tokens
    )
    
    agent = Agent(
        model=mock_model_complete,
        toolbox=toolbox,
        memory=memory
    )
    
    # 2. 执行 Agent
    user_msg = Message.user("Hello, AI!")
    output = await agent.run([user_msg])
    
    # 3. 验证输出
    assert output.content == "Hello from mock LLM"
    
    # 4. 验证 Message 序列化
    api_payload = user_msg.to_openai_format()
    assert api_payload["role"] == "user"
    assert api_payload["content"] == "Hello, AI!"

@pytest.mark.asyncio
async def test_error_handling_integration(mock_model_complete):
    """测试错误处理集成"""
    from gecko.core.exceptions import AgentError, ModelError
    
    # 让模型抛出异常
    mock_model_complete.acompletion.side_effect = RuntimeError("API Error")
    
    toolbox = ToolBox([])
    memory = TokenMemory(session_id="test")
    
    agent = Agent(
        model=mock_model_complete,
        toolbox=toolbox,
        memory=memory
    )
    
    # 应该抛出 ModelError（被包装）
    with pytest.raises(ModelError) as exc_info:
        await agent.run([Message.user("test")])
    
    # 验证异常链
    assert exc_info.value.__cause__ is not None

def test_logging_integration(caplog):
    """测试日志集成"""
    import logging
    caplog.set_level(logging.INFO)
    
    logger = get_logger("test.integration")
    logger.info("Integration test started", phase=1, status="success")
    
    # 验证日志被记录
    assert len(caplog.records) > 0

def test_config_integration(test_config):
    """测试配置系统集成"""
    assert test_config.log_level == "DEBUG"
    assert test_config.max_turns == 3
    assert test_config.enable_cache is False

@pytest.mark.asyncio
async def test_message_multimodal_integration(mock_model_complete, tmp_path):
    """测试多模态消息集成"""
    # 创建临时图片
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")
    
    # 创建多模态消息
    msg = Message.user(
        "What's in this image?",
        images=[str(image_file)]
    )
    
    # 序列化
    api_payload = msg.to_openai_format()
    
    # 验证格式
    assert isinstance(api_payload["content"], list)
    assert len(api_payload["content"]) == 2
    assert api_payload["content"][0]["type"] == "text"
    assert api_payload["content"][1]["type"] == "image_url"
    assert "base64" in api_payload["content"][1]["image_url"]["url"]

@pytest.mark.asyncio
async def test_memory_persistence(mock_model_complete, tmp_path):
    """测试记忆持久化"""
    from gecko.plugins.storage.sqlite import SQLiteSessionStorage
    
    # 创建存储
    storage = SQLiteSessionStorage(f"sqlite:///{tmp_path}/test.db")
    
    # 第一次对话
    memory1 = TokenMemory(session_id="user_123", storage=storage)
    agent1 = Agent(
        model=mock_model_complete,
        toolbox=ToolBox([]),
        memory=memory1
    )
    
    await agent1.run([Message.user("My name is Alice")])
    
    # 第二次对话（新实例）
    memory2 = TokenMemory(session_id="user_123", storage=storage)
    agent2 = Agent(
        model=mock_model_complete,
        toolbox=ToolBox([]),
        memory=memory2
    )
    
    # 应该能加载历史
    # （实际验证需要检查传给模型的消息）
    # 这里简化：只验证不报错
    await agent2.run([Message.user("What's my name?")])