import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

from pydantic import BaseModel

from gecko.core.engine.react import ReActEngine, ExecutionContext
from gecko.core.engine.base import AgentOutput
from gecko.core.message import Message
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox, ToolExecutionResult
from gecko.core.protocols import ModelProtocol, StreamableModelProtocol
from gecko.core.exceptions import AgentError, ModelError
from gecko.core.structure import StructureParseError

# ==========================================
# Helpers & Fixtures
# ==========================================

class MockResponseModel(BaseModel):
    """用于测试结构化输出的模型"""
    answer: str
    confidence: float

def create_mock_response(
    content: Optional[str] = None, 
    tool_calls: Optional[List[Dict]] = None,
    usage: Optional[Dict] = None
):
    """辅助函数：创建 Mock 的 LLM 响应"""
    msg_dict = {"role": "assistant"}
    if content is not None:
        msg_dict["content"] = content
    if tool_calls is not None:
        msg_dict["tool_calls"] = tool_calls
    
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.role = "assistant"
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.message.model_dump.return_value = msg_dict
    
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage or {"total_tokens": 10}
    return response

@pytest.fixture
def mock_model():
    model = MagicMock(spec=StreamableModelProtocol)
    model.acompletion = AsyncMock()
    model.astream = AsyncMock() 
    model._supports_function_calling = True
    return model

@pytest.fixture
def mock_toolbox():
    toolbox = MagicMock(spec=ToolBox)
    toolbox.to_openai_schema.return_value = [
        {
            "type": "function", 
            "function": {
                "name": "test_tool", 
                "description": "A test tool"
            }
        }
    ]
    toolbox.execute_many = AsyncMock()
    return toolbox

@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get = AsyncMock(return_value=None)
    storage.set = AsyncMock()
    return storage

@pytest.fixture
def mock_memory(mock_storage):
    memory = TokenMemory(session_id="test_session", storage=mock_storage)
    memory.get_history = AsyncMock(return_value=[])
    return memory

@pytest.fixture
def engine(mock_model, mock_toolbox, mock_memory):
    return ReActEngine(
        model=mock_model,
        toolbox=mock_toolbox,
        memory=mock_memory,
        max_turns=3
    )

# ==========================================
# Tests: Initialization & Config
# ==========================================

def test_init_config(mock_model, mock_toolbox, mock_memory):
    engine = ReActEngine(mock_model, mock_toolbox, mock_memory, system_prompt="Custom Prompt")
    assert engine.prompt_template.template == "Custom Prompt"

    engine_def = ReActEngine(mock_model, mock_toolbox, mock_memory)
    assert "Available Tools" in engine_def.prompt_template.template

    del mock_model._supports_function_calling
    engine_cap = ReActEngine(mock_model, mock_toolbox, mock_memory)
    assert engine_cap._supports_functions is True

@pytest.mark.asyncio
async def test_build_execution_context(engine):
    engine.memory.storage.get.return_value = {"messages": [{"role": "user", "content": "History"}]}
    history_msg = Message.user("History")
    engine.memory.get_history.return_value = [history_msg]
    
    input_msgs = [Message.user("Current")]
    ctx = await engine._build_execution_context(input_msgs)
    
    assert len(ctx.messages) == 3 # System + History + Current
    assert ctx.messages[0].role == "system"
    assert ctx.messages[1].content == "History"

    sys_msg = Message.system("User defined system")
    input_msgs_with_sys = [sys_msg, Message.user("Current")]
    ctx2 = await engine._build_execution_context(input_msgs_with_sys)
    
    assert len(ctx2.messages) == 3
    system_msgs = [m for m in ctx2.messages if m.role == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0].content == "User defined system"

# ==========================================
# Tests: Step (Reasoning Loop)
# ==========================================

@pytest.mark.asyncio
async def test_step_basic_text(engine, mock_model):
    mock_model.acompletion.return_value = create_mock_response(content="Hello World")
    
    output = await engine.step([Message.user("Hi")])
    
    assert output.content == "Hello World"
    assert engine.stats.total_steps == 1
    mock_model.acompletion.assert_called_once()

@pytest.mark.asyncio
async def test_step_with_tool_execution(engine, mock_model, mock_toolbox):
    resp1 = create_mock_response(
        tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]
    )
    resp2 = create_mock_response(content="Final Answer")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    
    mock_toolbox.execute_many.return_value = [
        ToolExecutionResult(tool_name="test_tool", call_id="call_1", result="Tool Result", is_error=False)
    ]
    
    engine.on_turn_start = AsyncMock()
    engine.on_turn_end = AsyncMock()
    engine.on_tool_execute = AsyncMock()

    output = await engine.step([Message.user("Run tool")])
    
    assert output.content == "Final Answer"
    assert mock_toolbox.execute_many.call_count == 1
    assert mock_model.acompletion.call_count == 2
    
    engine.on_turn_start.assert_called()
    engine.on_turn_end.assert_called()
    engine.on_tool_execute.assert_called_with("test_tool", {})

@pytest.mark.asyncio
async def test_step_max_turns_reached(engine, mock_model, mock_toolbox):
    resp_tool = create_mock_response(
        tool_calls=[{"id": "call_1", "function": {"name": "loop", "arguments": "{}"}}]
    )
    mock_model.acompletion.return_value = resp_tool
    mock_toolbox.execute_many.return_value = [
        ToolExecutionResult(tool_name="loop", call_id="call_1", result="...", is_error=False)
    ]
    
    engine.max_turns = 2
    output = await engine.step([Message.user("Loop")])
    
    assert mock_model.acompletion.call_count == 2
    assert output.content == "..."

@pytest.mark.asyncio
async def test_step_tool_error_feedback(engine, mock_model, mock_toolbox):
    resp1 = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "err", "arguments": "{}"}}])
    resp2 = create_mock_response(content="Fixed")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    
    mock_toolbox.execute_many.return_value = [
        ToolExecutionResult(tool_name="err", call_id="1", result="Error occurred", is_error=True)
    ]
    
    await engine.step([Message.user("Try")])
    
    call_args = mock_model.acompletion.call_args_list[1]
    messages_sent = call_args[1]['messages']
    
    assert messages_sent[-1]['role'] == 'user'
    assert "failed" in messages_sent[-1]['content']

# ==========================================
# Tests: Structured Output
# ==========================================

@pytest.mark.asyncio
async def test_step_structured_success(engine, mock_model):
    target_json = json.dumps({"answer": "42", "confidence": 0.99})
    resp = create_mock_response(
        tool_calls=[{
            "function": {
                "name": "mock_response_model",
                "arguments": target_json
            }
        }]
    )
    mock_model.acompletion.return_value = resp
    
    result = await engine.step(
        [Message.user("Q")], 
        response_model=MockResponseModel,
        strategy="function_calling"
    )
    
    assert isinstance(result, MockResponseModel)
    assert result.answer == "42"

@pytest.mark.asyncio
async def test_step_structured_retry_success(engine, mock_model):
    resp1 = create_mock_response(content="Not JSON")
    resp2 = create_mock_response(content=json.dumps({"answer": "OK", "confidence": 1.0}))
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    
    with patch("gecko.core.engine.react.StructureEngine") as MockStructEngine:
        MockStructEngine.to_openai_tool.return_value = {"function": {"name": "extract"}}
        MockStructEngine.parse = AsyncMock(side_effect=[
            StructureParseError("Invalid"),
            MockResponseModel(answer="OK", confidence=1.0)
        ])
        
        result = await engine.step([Message.user("Q")], response_model=MockResponseModel)
        
        assert result.answer == "OK"
        assert mock_model.acompletion.call_count == 2
        assert MockStructEngine.parse.call_count == 2

@pytest.mark.asyncio
async def test_step_structured_fail_max_retries(engine, mock_model):
    mock_model.acompletion.return_value = create_mock_response(content="Bad")
    
    with patch("gecko.core.engine.react.StructureEngine") as MockStructEngine:
        MockStructEngine.to_openai_tool.return_value = {"function": {"name": "extract"}}
        MockStructEngine.parse = AsyncMock(side_effect=StructureParseError("Fail"))
        
        with pytest.raises(AgentError) as exc:
            await engine.step(
                [Message.user("Q")], 
                response_model=MockResponseModel,
                max_retries=1
            )
        
        assert "Failed to parse structured output" in str(exc.value)
        assert mock_model.acompletion.call_count == 2

# ==========================================
# Tests: Streaming
# ==========================================

@pytest.mark.asyncio
async def test_step_stream_basic(engine, mock_model):
    mock_model.acompletion.return_value = create_mock_response(content="Start")
    
    # ✅ 修复：使用 SimpleNamespace 避免 MagicMock 自动创建 content 属性
    async def stream_gen(*args, **kwargs):
        chunk1 = SimpleNamespace(choices=[{"delta": {"content": "Hello"}}])
        chunk2 = SimpleNamespace(choices=[{"delta": {"content": " World"}}])
        yield chunk1
        yield chunk2
    
    mock_model.astream = MagicMock(side_effect=stream_gen)
    
    chunks = []
    async for chunk in engine.step_stream([Message.user("Hi")]):
        chunks.append(chunk)
    
    assert "".join(chunks) == "Start"

@pytest.mark.asyncio
async def test_step_stream_actual_streaming(engine, mock_model):
    """测试真正的流式"""
    mock_model.acompletion.side_effect = Exception("Peek Error")
    
    async def stream_gen(*args, **kwargs):
        chunk = SimpleNamespace(choices=[{"delta": {"content": "Streamed"}}])
        yield chunk
    
    mock_model.astream = MagicMock(side_effect=stream_gen)
    
    chunks = []
    async for chunk in engine.step_stream([Message.user("Hi")]):
        chunks.append(chunk)
    
    assert "Streamed" in "".join(chunks)


@pytest.mark.asyncio
async def test_step_stream_with_tool_peek(engine, mock_model, mock_toolbox):
    peek_resp = create_mock_response(
        tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}]
    )
    turn_resp = create_mock_response(
        tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}]
    )
    
    mock_model.acompletion.side_effect = [peek_resp, turn_resp]
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "Res", False)]
    
    async def stream_gen(*args, **kwargs):
        chunk = SimpleNamespace(choices=[{"delta": {"content": "Final"}}])
        yield chunk
    
    mock_model.astream = MagicMock(side_effect=stream_gen)
    
    chunks = []
    async for chunk in engine.step_stream([Message.user("Hi")]):
        chunks.append(chunk)
    
    assert "Final" in "".join(chunks)
    mock_toolbox.execute_many.assert_called()

@pytest.mark.asyncio
async def test_step_stream_not_supported(engine, mock_model):
    engine._supports_stream = False
    
    with pytest.raises(AgentError, match="不支持流式"):
        async for _ in engine.step_stream([Message.user("Hi")]):
            pass

# ==========================================
# Tests: Internals & Edge Cases
# ==========================================

@pytest.mark.asyncio
async def test_parse_llm_response_formats(engine):
    obj1 = MagicMock()
    obj1.choices = [MagicMock(message=MagicMock())]
    obj1.choices[0].message.model_dump.return_value = {"role": "assistant", "content": "A"}
    assert engine._parse_llm_response(obj1).content == "A"

    obj2 = MagicMock()
    obj2.choices = [MagicMock(message=MagicMock())]
    del obj2.choices[0].message.model_dump
    obj2.choices[0].message.to_dict.return_value = {"role": "assistant", "content": "B"}
    assert engine._parse_llm_response(obj2).content == "B"
    
    obj3 = MagicMock()
    obj3.choices = [MagicMock(message=MagicMock())]
    del obj3.choices[0].message.model_dump
    del obj3.choices[0].message.to_dict
    obj3.choices[0].message.role = "assistant"
    obj3.choices[0].message.content = "C"
    obj3.choices[0].message.tool_calls = None
    assert engine._parse_llm_response(obj3).content == "C"

    with pytest.raises(ModelError):
        engine._parse_llm_response(MagicMock(choices=[]))

@pytest.mark.asyncio
async def test_error_handling_in_loop(engine, mock_model):
    mock_model.acompletion.side_effect = Exception("API Crash")
    engine.on_error = AsyncMock()
    
    with pytest.raises(Exception, match="API Crash"):
        await engine.step([Message.user("Hi")])
    
    engine.on_error.assert_called_once()
    assert engine.stats.errors == 1

@pytest.mark.asyncio
async def test_save_context_failure(engine, mock_storage):
    engine.memory.storage = mock_storage
    mock_storage.set.side_effect = Exception("DB Error")
    
    ctx = ExecutionContext([Message.user("Hi")])
    await engine._save_context(ctx) # Should not raise

@pytest.mark.asyncio
async def test_peek_failure(engine, mock_model):
    mock_model.acompletion.side_effect = Exception("Peek Fail")
    
    ctx = ExecutionContext([Message.user("Hi")])
    needs_tools, resp = await engine._check_needs_tools(ctx, {})
    
    assert needs_tools is False
    assert resp is None