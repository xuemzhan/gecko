# tests/core/test_builder.py
import pytest
from unittest.mock import MagicMock
from gecko.core.builder import AgentBuilder
from gecko.core.agent import Agent
from gecko.core.engine.base import CognitiveEngine
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.core.exceptions import ConfigurationError

class MockEngine(CognitiveEngine):
    async def step(self, *args, **kwargs): pass # type: ignore

def test_builder_storage_validation(mock_llm):
    """[New] 测试 Storage 接口类型检查"""
    builder = AgentBuilder().with_model(mock_llm)
    
    # 1. 传入无效对象
    class NotStorage: pass
    with pytest.raises(TypeError, match="SessionInterface"):
        builder.with_storage(NotStorage()) # type: ignore
        
    # 2. 传入有效对象 (Mock)
    valid_storage = MagicMock(spec=SessionInterface)
    builder.with_storage(valid_storage)
    assert builder._storage is valid_storage

def test_builder_engine_kwargs_passthrough(mock_llm):
    """[New] 测试 Engine 参数透传"""
    builder = AgentBuilder().with_model(mock_llm)
    
    # 设置 Engine 参数
    builder.with_engine(
        MockEngine,
        custom_param="value",
        max_iterations=99
    )
    
    agent = builder.build()
    
    assert isinstance(agent.engine, MockEngine)
    assert agent.engine.max_iterations == 99
    # 验证 custom_param 被传入 _config (基类行为)
    assert agent.engine.get_config("custom_param") == "value"

def test_builder_system_prompt_handling(mock_llm):
    """[New] 测试 System Prompt 统一处理"""
    builder = AgentBuilder().with_model(mock_llm)
    builder.with_system_prompt("You are a bot")
    
    # system_prompt 应该被放入 engine_kwargs
    assert builder._engine_kwargs["system_prompt"] == "You are a bot"
    
    agent = builder.build()
    # 验证 ReActEngine (默认) 接收到了
    # 注意：ReActEngine 会将 str 转换为 PromptTemplate
    if hasattr(agent.engine, "prompt_template"):
        assert "You are a bot" in agent.engine.prompt_template.template # type: ignore