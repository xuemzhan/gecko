# tests/core/engine/test_react.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.core.engine.react import ReActEngine
from gecko.core.message import Message
from gecko.core.output import AgentOutput

@pytest.mark.asyncio
async def test_react_max_turns(toolbox, memory):
    """测试最大轮次限制"""
    mock_model = MagicMock()
    
    # [修复] 显式创建一个 mock message 并配置 model_dump
    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.tool_calls = [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]
    
    # 关键修复：让 model_dump 返回字典，而不是默认的 MagicMock 对象
    mock_message.model_dump.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]
    }
    
    # 模拟一直返回 tool call，导致死循环
    mock_model.acompletion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=mock_message)]
    ))
    
    engine = ReActEngine(mock_model, toolbox, memory, max_turns=2)
    
    # Mock toolbox to avoid actual execution error
    toolbox.execute = AsyncMock(return_value="tool_res")
    
    output = await engine.step([Message.user("hi")])
    
    assert output.content == "Max iterations reached."
    # 应该调用了2次 LLM (turn 1, turn 2)
    assert mock_model.acompletion.call_count == 2