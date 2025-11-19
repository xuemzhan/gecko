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
        
        # [核心修复] 显式将 tool_calls 设为 None 或空列表
        # 否则 MagicMock 属性在 boolean context 下默认为 True，
        # 导致 Runner 误以为有工具调用，进而引发后续的 Mock 对象泄露到消息历史中。
        mock_response.choices[0].message.tool_calls = None 
        
        # 安全获取 content，兼容 dict 或对象
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            last_content = last_msg.get('content', '')
        else:
            last_content = getattr(last_msg, 'content', '')

        # 确保内容是字符串
        if not isinstance(last_content, str):
             last_content = str(last_content)
             
        mock_response.choices[0].message.content = f"Mock response for: {last_content}"
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