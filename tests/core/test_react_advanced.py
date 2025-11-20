# tests/core/test_react_advanced.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from gecko.core.engine.react import ReActEngine
from gecko.core.message import Message
from gecko.core.output import AgentOutput

class MockResponse(BaseModel):
    result: str

@pytest.mark.asyncio
async def test_react_hooks(toolbox, memory):
    """测试生命周期钩子"""
    mock_model = MagicMock()
    
    # [修复] 配置 message.model_dump 返回字典
    mock_message = MagicMock(content="done", tool_calls=None)
    mock_message.model_dump.return_value = {
        "role": "assistant",
        "content": "done",
        "tool_calls": None
    }
    
    mock_model.acompletion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=mock_message)]
    ))
    
    # 定义钩子
    start_hook = AsyncMock()
    end_hook = AsyncMock()
    tool_hook = MagicMock()
    
    engine = ReActEngine(
        mock_model, toolbox, memory,
        on_turn_start=start_hook,
        on_turn_end=end_hook,
        on_tool_execute=tool_hook
    )
    
    await engine.step([Message.user("hi")])
    
    assert start_hook.called
    assert end_hook.called

@pytest.mark.asyncio
async def test_structure_retry_logic(toolbox, memory):
    """测试结构化输出的错误反馈重试机制"""
    mock_model = MagicMock()

    # 模拟 LLM 行为：
    # 第1次调用：返回错误的 JSON
    bad_message = MagicMock(content='{"result": 123}', tool_calls=None)
    bad_message.model_dump.return_value = {
        "role": "assistant",
        "content": '{"result": 123}',
        "tool_calls": None
    }
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])

    # 第2次调用：返回正确的 JSON (在收到反馈后)
    good_message = MagicMock(content='{"result": "fixed"}', tool_calls=None)
    good_message.model_dump.return_value = {
        "role": "assistant",
        "content": '{"result": "fixed"}',
        "tool_calls": None
    }
    good_response = MagicMock(choices=[MagicMock(message=good_message)])

    mock_model.acompletion = AsyncMock(side_effect=[bad_response, good_response])

    engine = ReActEngine(mock_model, toolbox, memory, max_turns=3)

    # 执行
    result = await engine.step(
        [Message.user("extract")],
        response_model=MockResponse,
        max_retries=1
    )

    assert isinstance(result, MockResponse)
    assert result.result == "fixed"
    # 验证是否发起了2次调用
    assert mock_model.acompletion.call_count == 2

    # 验证是否将错误反馈到了上下文
    call_args_list = mock_model.acompletion.call_args_list
    second_call_msgs = call_args_list[1][1]['messages']
    
    # [修复] 修改索引：反馈消息是输入列表的最后一条 (-1)，而不是倒数第二条 (-2)
    # Context sequence: [System, User, Assistant(BadOutput), User(Feedback)]
    assert "Parsing error" in second_call_msgs[-1]['content']