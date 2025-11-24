# tests/core/test_react.py
"""
ReAct Engine 单元测试 (完整版)

覆盖率目标：100%
包含：
1. 基础推理 (Text/Tool)
2. 流式推理 (Stream with Tool Chunks)
3. 错误处理与反馈
4. 新特性 (死循环检测、输出截断)
5. 生命周期 Hooks
6. 结构化输出与重试
7. 上下文与记忆管理
"""
import pytest
import json
import time
from unittest.mock import MagicMock, AsyncMock, call
from types import SimpleNamespace
from typing import Any, List

from pydantic import BaseModel

from gecko.core.engine.react import ReActEngine, ExecutionContext
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.toolbox import ToolBox, ToolExecutionResult
from gecko.core.protocols import StreamableModelProtocol

# ========================= Helpers =========================

def create_mock_response(content: str = None, tool_calls: list = None): # type: ignore
    """创建模拟的 CompletionResponse"""
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    
    choice = MagicMock()
    choice.message = SimpleNamespace(**msg)
    
    response = MagicMock()
    response.choices = [choice]
    return response

class MockModel(MagicMock):
    """同时支持同步和流式的 Mock 模型"""
    def __init__(self, *args, **kwargs):
        super().__init__(spec=StreamableModelProtocol, *args, **kwargs)
        self.acompletion = AsyncMock()
        self.astream = MagicMock()

# ========================= Fixtures =========================

@pytest.fixture
def mock_toolbox():
    tb = MagicMock(spec=ToolBox)
    tb.to_openai_schema.return_value = [{"type": "function", "function": {"name": "t1"}}]
    tb.execute_many = AsyncMock(return_value=[])
    return tb

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.storage = MagicMock() # 模拟有存储
    mem.session_id = "test_session"
    mem.storage.get = AsyncMock(return_value=None)
    mem.storage.set = AsyncMock()
    mem.get_history = AsyncMock(return_value=[])
    return mem

@pytest.fixture
def engine(mock_model, mock_toolbox, mock_memory):
    return ReActEngine(
        model=mock_model,
        toolbox=mock_toolbox,
        memory=mock_memory,
        max_turns=5,
        max_observation_length=100  # 设置较小的截断阈值方便测试
    )

# ========================= 1. 基础推理测试 =========================

@pytest.mark.asyncio
async def test_step_basic_text(engine, mock_model):
    """测试基本的文本回复"""
    mock_model.acompletion.return_value = create_mock_response(content="Hello")
    output = await engine.step([Message.user("Hi")])
    assert output.content == "Hello"
    mock_model.acompletion.assert_called_once()

@pytest.mark.asyncio
async def test_step_with_tool_execution(engine, mock_model, mock_toolbox):
    """测试工具调用流程"""
    # Round 1: Tool Call
    resp1 = create_mock_response(tool_calls=[
        {"id": "call_1", "function": {"name": "t1", "arguments": "{}"}}
    ])
    # Round 2: Final Answer
    resp2 = create_mock_response(content="Final")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    mock_toolbox.execute_many.return_value = [
        ToolExecutionResult("t1", "call_1", "Result", False)
    ]
    
    output = await engine.step([Message.user("Run")])
    
    assert output.content == "Final"
    # 验证中间消息包含 tool role
    call_args = mock_model.acompletion.call_args_list[1]
    messages = call_args[1]['messages']
    assert messages[-1]['role'] == 'tool'
    assert messages[-1]['content'] == "Result"

@pytest.mark.asyncio
async def test_step_max_turns(engine, mock_model):
    """测试最大轮数限制"""
    resp = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}])
    mock_model.acompletion.return_value = resp
    engine.max_turns = 2
    
    output = await engine.step([Message.user("Loop")])
    
    # 超过轮数强制结束
    assert "No response" in output.content or output.tool_calls

# ========================= 2. 流式推理测试 =========================

@pytest.mark.asyncio
async def test_step_stream_basic(engine, mock_model):
    """测试纯文本流式"""
    async def stream_gen(*args, **kwargs):
        yield SimpleNamespace(choices=[{"delta": {"content": "A"}}])
        yield SimpleNamespace(choices=[{"delta": {"content": "B"}}])

    mock_model.astream = MagicMock(side_effect=stream_gen)
    chunks = [c async for c in engine.step_stream([Message.user("Hi")])]
    assert "".join(chunks) == "AB"

@pytest.mark.asyncio
async def test_step_stream_with_tool_call(engine, mock_model, mock_toolbox):
    """测试流式工具调用 (模拟 chunk 拼接)"""
    # Round 1: Tool Call Chunks
    async def stream_r1(*args, **kwargs):
        yield SimpleNamespace(choices=[{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t1"}}]}}])
        yield SimpleNamespace(choices=[{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}])

    # Round 2: Text
    async def stream_r2(*args, **kwargs):
        yield SimpleNamespace(choices=[{"delta": {"content": "Done"}}])

    mock_model.astream.side_effect = [stream_r1(), stream_r2()]
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "c1", "Res", False)]

    chunks = [c async for c in engine.step_stream([Message.user("Run")])]
    
    assert "".join(chunks) == "Done"
    mock_toolbox.execute_many.assert_called_once()

# ========================= 3. 错误处理与反馈 =========================

@pytest.mark.asyncio
async def test_step_tool_error_feedback(engine, mock_model, mock_toolbox):
    """测试工具错误反馈"""
    resp1 = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "err", "arguments": "{}"}}])
    resp2 = create_mock_response(content="Fixed")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    mock_toolbox.execute_many.return_value = [
        ToolExecutionResult("err", "1", "Error!!", True)
    ]
    
    await engine.step([Message.user("Try")])
    
    # 验证即使错误，也以 tool role 返回
    messages = mock_model.acompletion.call_args_list[1][1]['messages']
    assert messages[-1]['role'] == 'tool'
    assert "Error!!" in messages[-1]['content']

@pytest.mark.asyncio
async def test_consecutive_error_warning(engine, mock_model, mock_toolbox):
    """测试连续 3 次错误触发系统警告"""
    # [Fix] 为了避开死循环检测，让每次 Tool Call 的参数略有不同
    # 模拟连续 3 次调用工具都报错，但参数不同 (i=1, i=2, i=3)
    resp1 = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": '{"i": 1}'}}])
    resp2 = create_mock_response(tool_calls=[{"id": "2", "function": {"name": "t1", "arguments": '{"i": 2}'}}])
    resp3 = create_mock_response(tool_calls=[{"id": "3", "function": {"name": "t1", "arguments": '{"i": 3}'}}])
    resp_stop = create_mock_response("Stop")

    mock_model.acompletion.side_effect = [resp1, resp2, resp3, resp_stop]

    # 工具每次都返回错误
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "Err", True)]

    engine.max_turns = 10
    await engine.step([Message.user("Try")])

    # 第 4 次调用时 (下标为3)，上下文中应该包含 User 的警告信息
    call_4 = mock_model.acompletion.call_args_list[3]
    msgs = call_4[1]['messages']

    # 验证消息结构: ... -> [Assistant] -> [Tool Result (Err)] -> [User Warning]
    
    # 1. 最后一条应该是 System Warning (User role)
    assert msgs[-1]['role'] == 'user'
    assert "Too many tool errors" in msgs[-1]['content']
    
    # 2. 倒数第二条是 Tool Result
    assert msgs[-2]['role'] == 'tool'

# ========================= 4. 新特性测试 =========================

@pytest.mark.asyncio
async def test_infinite_loop_detection(engine, mock_model, mock_toolbox):
    """测试死循环检测"""
    # 模拟 LLM 总是返回相同的 Tool Call
    same_call = create_mock_response(tool_calls=[
        {"id": "1", "function": {"name": "t1", "arguments": '{"a":1}'}}
    ])
    
    mock_model.acompletion.return_value = same_call
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "Res", False)]
    
    engine.max_turns = 5
    await engine.step([Message.user("Loop")])
    
    # 应该只调用了 2 次 (第一次正常，第二次发现 hash 相同被中断)
    assert mock_model.acompletion.call_count == 2
    # 验证 warning 日志 (实际通过 mock logger 验证，这里简略)

@pytest.mark.asyncio
async def test_observation_truncation(engine, mock_model, mock_toolbox):
    """测试工具输出截断"""
    resp1 = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}])
    resp2 = create_mock_response(content="Done")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    
    # 返回超长结果 (limit=100)
    long_text = "A" * 200
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", long_text, False)]
    
    await engine.step([Message.user("BigData")])
    
    # 验证上下文中的 Tool Message 被截断
    messages = mock_model.acompletion.call_args_list[1][1]['messages']
    tool_content = messages[-1]['content']
    
    assert len(tool_content) < 200
    assert "truncated" in tool_content

# ========================= 5. Hooks 测试 =========================

@pytest.mark.asyncio
async def test_hooks_execution(mock_model, mock_toolbox, mock_memory):
    """测试生命周期 Hooks"""
    start_hook = AsyncMock()
    end_hook = AsyncMock()
    tool_hook = AsyncMock()
    
    engine = ReActEngine(
        mock_model, mock_toolbox, mock_memory,
        on_turn_start=start_hook,
        on_turn_end=end_hook,
        on_tool_execute=tool_hook
    )
    
    # 模拟一次工具调用流程
    mock_model.acompletion.side_effect = [
        create_mock_response(tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}]),
        create_mock_response(content="Done")
    ]
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "Res", False)]
    
    await engine.step([Message.user("Hook")])
    
    # 验证 Hooks 调用次数
    assert start_hook.call_count == 2 # 2 turns
    assert end_hook.call_count == 2
    assert tool_hook.call_count == 1

# ========================= 6. 结构化输出与重试 =========================

@pytest.mark.asyncio
async def test_structure_output_retry(engine, mock_model):
    """测试结构化输出解析失败后的重试"""
    class User(BaseModel):
        name: str
    
    # 1. 错误格式 -> 2. 正确格式
    mock_model.acompletion.side_effect = [
        create_mock_response(content="Not JSON"), 
        create_mock_response(content='{"name": "Alice"}')
    ]
    
    user = await engine.step([Message.user("Parse")], response_model=User, max_retries=1)
    
    assert isinstance(user, User)
    assert user.name == "Alice"
    # 验证发生了重试 (2次 LLM 调用)
    assert mock_model.acompletion.call_count == 2
    # 验证第二次调用包含了反馈信息
    msg_2 = mock_model.acompletion.call_args_list[1][1]['messages']
    assert "Error parsing response" in msg_2[-1]['content']

# ========================= 7. 上下文与记忆 =========================

@pytest.mark.asyncio
async def test_context_building_with_history(engine, mock_memory, mock_model):
    """测试历史记录加载和 System Prompt"""
    # [Fix] 必须 Mock storage.get 返回包含 "messages" 的数据，
    # 否则 _load_history 会直接返回 []，不调用 get_history
    mock_memory.storage.get.return_value = {"messages": ["some_raw_data"]}
    
    # 模拟内存转换后的历史对象
    mock_memory.get_history.return_value = [Message.user("Old")]
    mock_model.acompletion.return_value = create_mock_response(content="Hi")

    await engine.step([Message.user("New")])

    call_args = mock_model.acompletion.call_args[1]
    sent_msgs = call_args['messages']

    # 验证顺序: System -> History(Old) -> Input(New)
    assert sent_msgs[0]['role'] == 'system' # 自动注入的 System Prompt
    assert sent_msgs[1]['role'] == 'user' and sent_msgs[1]['content'] == "Old"
    assert sent_msgs[2]['role'] == 'user' and sent_msgs[2]['content'] == "New"

@pytest.mark.asyncio
async def test_system_prompt_rendering(mock_model, mock_toolbox, mock_memory):
    """测试 System Prompt 模板渲染"""
    # 使用自定义模板
    tmpl = "Time: {{ current_time }}"
    engine = ReActEngine(mock_model, mock_toolbox, mock_memory, system_prompt=tmpl)
    mock_model.acompletion.return_value = create_mock_response("Hi")
    
    await engine.step([Message.user("A")])
    
    sent_msgs = mock_model.acompletion.call_args[1]['messages']
    sys_content = sent_msgs[0]['content']
    
    # 验证时间被注入 (当前年份)
    import datetime
    current_year = str(datetime.datetime.now().year)
    assert "Time:" in sys_content
    assert current_year in sys_content

@pytest.mark.asyncio
async def test_step_stream_recursion_depth_safety(engine, mock_model, mock_toolbox):
    """
    [New] 测试流式推理在高轮次下不会触发 RecursionError
    验证优化点：ReActEngine._run_streaming_loop 改为迭代式
    """
    engine.max_turns = 50

    # [修复] 使用计数器生成唯一的参数，避开 ReAct 引擎的 Hash 死循环检测
    counter = 0
    async def endless_tool_stream(*args, **kwargs):
        nonlocal counter
        counter += 1
        # 构造动态参数 {"i": 1}, {"i": 2}...
        unique_args = f'{{"i": {counter}}}'
        
        yield SimpleNamespace(choices=[{
            "delta": {
                "tool_calls": [{
                    "index": 0, 
                    "function": {
                        "name": "t1", 
                        "arguments": unique_args
                    }
                }]
            }
        }])

    mock_model.astream.side_effect = endless_tool_stream
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "res", False)]

    # 执行流式推理
    count = 0
    async for _ in engine.step_stream([Message.user("Start")]):
        count += 1

    # 验证确实执行了多次 LLM 调用 (达到 max_turns 上限)
    assert mock_model.astream.call_count >= 50