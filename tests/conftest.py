# tests/conftest.py
import asyncio
import pytest
import litellm
from unittest.mock import AsyncMock, MagicMock
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.core.output import AgentOutput

# [新增] 自动清理 fixture
@pytest.fixture(autouse=True)
async def cleanup_resources():
    """在每个测试后清理全局资源，防止 ResourceWarning"""
    yield
    # 清理 litellm 客户端
    if hasattr(litellm, "async_http_handler") and litellm.async_http_handler:
        await litellm.async_http_handler.client.close()
    if hasattr(litellm, "http_client") and litellm.http_client:
        # httpx client
        await litellm.http_client.aclose()
        
    # 给予一点时间让底层 socket 关闭
    await asyncio.sleep(0.01)
class MockModel:
    async def acompletion(self, messages, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        
        # [核心修复] 显式将 tool_calls 设为 None 或空列表
        mock_response.choices[0].message.tool_calls = None 
        
        # 安全获取 content
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            last_content = last_msg.get('content', '')
        else:
            last_content = getattr(last_msg, 'content', '')

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