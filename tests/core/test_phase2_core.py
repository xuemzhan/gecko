# tests/unit/test_phase2_core.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.core.agent import Agent
from gecko.core.message import Message
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.react import ReActEngine
from gecko.plugins.tools.base import BaseTool

# --- Mocks ---
class MockModel:
    async def acompletion(self, messages, **kwargs):
        # 模拟一个简单的响应
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello from Mock LLM"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].message.model_dump = lambda: {
            "role": "assistant", 
            "content": "Hello from Mock LLM"
        }
        return mock_resp

class Calculator(BaseTool):
    name: str = "calc"
    description: str = "calc"
    async def execute(self, args):
        return "42"

# --- Tests ---

@pytest.mark.asyncio
async def test_agent_run_basic_flow():
    # Setup
    model = MockModel()
    toolbox = ToolBox([Calculator()])
    memory = TokenMemory("test_session")
    
    agent = Agent(model, toolbox, memory)
    
    # Execute
    response = await agent.run("Hi")
    
    # Verify
    assert response.content == "Hello from Mock LLM"
    
@pytest.mark.asyncio
async def test_react_engine_steps():
    # 这是一个更深度的测试，验证 Engine 内部是否正确构造了上下文
    model = MockModel()
    toolbox = ToolBox()
    memory = TokenMemory("test_session")
    
    engine = ReActEngine(model, toolbox, memory)
    
    msgs = [Message(role="user", content="Test")]
    output = await engine.step(msgs)
    
    assert output.content == "Hello from Mock LLM"