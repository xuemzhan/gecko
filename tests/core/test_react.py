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
from unittest.mock import MagicMock, AsyncMock
from types import SimpleNamespace

from pydantic import BaseModel

from gecko.core.engine.react import ReActEngine, ExecutionContext
from gecko.core.events.bus import EventBus
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

# [修复] 显式定义 mock_event_bus fixture
@pytest.fixture
def mock_event_bus():
    bus = MagicMock(spec=EventBus)
    bus.publish = AsyncMock()
    return bus

# [修复] 确保 engine fixture 接收 mock_event_bus 参数
@pytest.fixture
def engine(mock_model, mock_toolbox, mock_memory, mock_event_bus):
    return ReActEngine(
        model=mock_model,
        toolbox=mock_toolbox,
        memory=mock_memory,
        event_bus=mock_event_bus, # 注入
        max_turns=5,
        max_observation_length=100
    )
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
    # 构造导致死循环的相同调用
    resp = create_mock_response(tool_calls=[{"id": "1", "function": {"name": "t1", "arguments": "{}"}}])
    mock_model.acompletion.return_value = resp
    engine.max_turns = 2
    
    output = await engine.step([Message.user("Loop")])
    
    # [Fix] 更新断言：新的死循环检测逻辑会注入 System Alert
    # 检查内容中是否包含 "Execution stopped" 或 "System Alert"
    assert "Execution stopped" in output.content or "System Alert" in output.content

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

# ========================= 8. 修复验证与增强测试 =========================

@pytest.mark.asyncio
async def test_structured_output_with_intermediate_tool(engine, mock_model):
    """
    [Fix Verification] 测试非目标工具拦截逻辑
    
    场景：用户请求结构化输出 (TargetModel)，但模型因死循环或逻辑错误，
    返回了一个中间工具 (e.g., search)，而不是最终结果工具。
    
    期望：
    1. 系统不应尝试用 search 的参数去解析 TargetModel (防止 ValidationError)。
    2. 系统应抛出 StructureParseError 并触发重试。
    3. 重试后若模型返回正确结果，应成功。
    """
    class TargetModel(BaseModel):
        reason: str
        score: int

    target_tool_name = "targetmodel"
    
    resp_wrong = create_mock_response(tool_calls=[
        {"id": "1", "function": {"name": "calculator", "arguments": '{"expr": "1+1"}'}}
    ])
    
    resp_correct = create_mock_response(tool_calls=[
        {"id": "2", "function": {
            "name": target_tool_name, 
            "arguments": '{"reason": "ok", "score": 100}'
        }}
    ])
    
    mock_model.acompletion.side_effect = [resp_wrong, resp_correct]
    
    # [Fix] 关键修改：将 max_turns 设置为 1
    # 这强制引擎在第一轮（错误工具）后停止，将其作为最终结果返回，
    # 从而触发 _handle_structured_output 中的校验和重试逻辑。
    engine.max_turns = 1
    
    # 执行
    result = await engine.step(
        [Message.user("Evaluate")], 
        response_model=TargetModel,
        max_retries=1
    )
    
    # 验证
    assert isinstance(result, TargetModel)
    assert result.score == 100
    
    # 验证调用次数 (1次 ReAct + 1次 Retry)
    assert mock_model.acompletion.call_count == 2
    
    # 检查重试时的反馈消息
    # call_args_list[1] 是第二次调用 (Retry)
    # [1]['messages'] 是参数中的 messages 列表
    retry_input_msgs = mock_model.acompletion.call_args_list[1][1]['messages']
    last_msg = retry_input_msgs[-1]
    
    # 验证反馈内容
    assert last_msg['role'] == 'user'
    assert "Incorrect tool used" in last_msg['content']

@pytest.mark.asyncio
async def test_structured_output_with_parallel_tools(engine, mock_model):
    """
    [Enhancement] 测试并行工具调用筛选
    
    场景：模型很聪明，在一次返回中同时调用了 'save_log' (副作用) 和 'final_result' (目标)。
    期望：系统能遍历 tool_calls，忽略 log，精准找到 final_result 进行解析。
    """
    class FinalResult(BaseModel):
        answer: str

    target_name = "finalresult"
    
    # 模拟并行调用：列表里有两个工具
    parallel_resp = create_mock_response(tool_calls=[
        # 干扰项：排在第一个
        {"id": "1", "function": {"name": "save_log", "arguments": '{"msg": "thinking"}'}},
        # 目标项
        {"id": "2", "function": {"name": target_name, "arguments": '{"answer": "42"}'}}
    ])
    
    mock_model.acompletion.return_value = parallel_resp
    
    # 执行
    result = await engine.step([Message.user("Run")], response_model=FinalResult)
    
    # 验证
    assert isinstance(result, FinalResult)
    assert result.answer == "42"
    # 确保只调用了一次 LLM (没有因为解析错误而重试)
    assert mock_model.acompletion.call_count == 1

@pytest.mark.asyncio
async def test_infinite_loop_feedback_injection(engine, mock_model, mock_toolbox):
    """
    [Fix Verification] 测试死循环时的 System Alert 注入
    
    场景：模型陷入死循环，被 detect_infinite_loop 拦截。
    期望：
    1. 循环中断。
    2. 上下文中被注入了一条 System Alert (User Role)。
    """
    # 构造完全相同的工具调用
    repeat_call = create_mock_response(tool_calls=[
        {"id": "x", "function": {"name": "t1", "arguments": "{}"}}
    ])
    
    # 设置为每次都返回相同内容
    mock_model.acompletion.return_value = repeat_call
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "x", "res", False)]
    
    engine.max_turns = 5
    
    # 执行 (这会触发死循环保护)
    # 我们不关心返回值，只关心上下文状态
    await engine.step([Message.user("Loop")])
    
    # 获取最后一次调用时的上下文消息历史
    last_call_args = mock_model.acompletion.call_args
    messages_sent = last_call_args[1]['messages']
    
    # 验证最后一条消息是否包含 System Alert
    # 注意：Engine 内部逻辑是：检测到循环 -> 添加 AssitantMsg(ToolCall) -> 添加 UserMsg(Alert) -> Break
    # 但因为 Break 了，最后一条 Alert 实际上不会发给 LLM (因为没有下一轮了)。
    # 但是 step 方法最后会保存 context。我们需要检查 engine 内部的 context 或者通过 Side Effect 验证。
    
    # 这里我们通过检查 engine 运行过程中的行为来验证。
    # 由于 step 方法结束后 context 就销毁了（局部变量），我们可以 Mock on_turn_end 来捕获 Context。
    
    captured_context = None
    async def capture_ctx(ctx):
        nonlocal captured_context
        captured_context = ctx
    
    engine.on_turn_end = capture_ctx
    
    # 重新运行
    await engine.step([Message.user("Loop2")])
    
    assert captured_context is not None
    last_msg = captured_context.messages[-1]
    
    # 验证注入了警告
    assert last_msg.role == "user"
    assert "System Alert" in str(last_msg.content)
    assert "Stop looping" in str(last_msg.content)

# [新增] 测试流式工具事件
@pytest.mark.asyncio
async def test_step_stream_events(engine, mock_model, mock_toolbox, mock_event_bus):
    """
    测试流式模式下是否发送工具开始/结束事件
    解决前端'静默'问题
    """
    # 1. 模拟第一轮：LLM 调用工具
    async def stream_gen_tool(*args, **kwargs):
        # 模拟返回一个 Tool Call
        yield SimpleNamespace(choices=[{
            "delta": {
                "tool_calls": [{
                    "index": 0, 
                    "id": "c1", 
                    "function": {"name": "t1", "arguments": "{}"}
                }]
            }
        }])
    
    # 2. [新增] 模拟第二轮：LLM 在工具执行后返回最终结果
    # 如果不提供这个，ReAct 循环再次调用 astream 时会因为 side_effect 耗尽而崩溃
    async def stream_gen_final(*args, **kwargs):
        yield SimpleNamespace(choices=[{
            "delta": {
                "content": "Done"
            }
        }])
    
    # 配置 side_effect 包含两轮响应
    mock_model.astream.side_effect = [stream_gen_tool(), stream_gen_final()]
    
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "c1", "Res", False)]
    
    # 执行流式
    async for _ in engine.step_stream([Message.user("Run")]):
        pass
        
    # 验证 EventBus 调用
    # 至少调用2次 (Start, End)
    assert mock_event_bus.publish.call_count >= 2
    
    # 验证具体事件类型
    calls = mock_event_bus.publish.call_args_list
    event_types = [c[0][0].type for c in calls] # c[0][0] is the Event object
    
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types

# [新增] 测试 JSON 容错与反馈
@pytest.mark.asyncio
async def test_react_json_fault_tolerance(engine, mock_model, mock_toolbox):
    """
    测试 LLM 返回非法 JSON 时的处理流程
    """
    # 1. LLM 返回非法 JSON (少个引号)
    bad_json_args = '{"arg": "value"' 
    
    resp1 = create_mock_response(tool_calls=[
        {"id": "1", "function": {"name": "t1", "arguments": bad_json_args}}
    ])
    # 2. 第二轮 LLM 收到错误反馈后，修正并返回正确结果
    resp2 = create_mock_response(content="Sorry, here is the result.")
    
    mock_model.acompletion.side_effect = [resp1, resp2]
    
    # 3. ToolBox 应该收到带有错误标记的参数
    # 我们需要模拟 ToolBox 的 execute_many 行为，这里它会识别标记并返回系统错误提示
    # 实际集成中由 ToolBox 代码处理，但在单元测试中我们需要 Mock 它的返回值或者使用真实 ToolBox
    # 为了测试 Engine 的逻辑，我们验证 Engine 传递给 ToolBox 的参数是否包含标记
    
    await engine.step([Message.user("Break JSON")])
    
    # 验证 Engine 调用 ToolBox 时传递了错误标记
    execute_call = mock_toolbox.execute_many.call_args
    passed_tool_calls = execute_call[0][0] # 第一个参数
    
    assert len(passed_tool_calls) == 1
    args = passed_tool_calls[0]["arguments"]
    
    # 关键断言：Engine 捕获了 JSON 错误并转换为了特殊 Key
    assert "__gecko_parse_error__" in args
    assert "JSON format error" in args["__gecko_parse_error__"]

@pytest.mark.asyncio
async def test_step_stream_infinite_loop_notification(engine, mock_model, mock_toolbox):
    """
    [New] 测试流式模式下死循环检测的通知机制
    """
    # 模拟死循环工具调用
    same_call_chunk = SimpleNamespace(choices=[{
        "delta": {
            "tool_calls": [{
                "index": 0, 
                "function": {"name": "t1", "arguments": '{"a":1}'}
            }]
        }
    }])
    
    mock_model.astream.return_value = (c for c in [same_call_chunk]) # 模拟生成器
    # 必须让 Mock Model 表现出每次都返回一样
    mock_model.astream.side_effect = None
    mock_model.astream.return_value = None
    
    async def endless_stream(*args, **kwargs):
        yield same_call_chunk

    mock_model.astream.side_effect = endless_stream
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "1", "Res", False)]
    
    engine.max_turns = 5
    
    collected_output = []
    async for chunk in engine.step_stream([Message.user("Loop")]):
        collected_output.append(chunk)
        
    full_text = "".join(collected_output)
    
    # 验证输出中包含系统注入的警告
    assert "[System: Execution stopped" in full_text
    assert "infinite tool loop" in full_text

@pytest.mark.asyncio
async def test_stream_recursion_and_exit_reason(engine, mock_model, mock_toolbox):
    """
    [New] 测试流式递归循环和退出原因反馈
    模拟场景：模型不断调用同一工具，触发死循环保护
    """
    # 1. 构造一个总是返回相同 Tool Call 的流式响应
    tool_chunk = MagicMock()
    tool_chunk.choices = [{
        "delta": {
            "tool_calls": [{
                "index": 0, 
                "function": {"name": "t1", "arguments": '{"a": 1}'}
            }]
        }
    }]
    
    # 模拟 astream 每次被调用都返回这个 chunk
    async def infinite_stream(*args, **kwargs):
        yield tool_chunk
        
    mock_model.astream.side_effect = infinite_stream
    mock_toolbox.execute_many.return_value = [ToolExecutionResult("t1", "id", "result", False)]
    
    engine.max_turns = 3
    
    chunks = []
    async for token in engine.step_stream([Message.user("Start")]):
        chunks.append(token)
    
    full_output = "".join(chunks)
    
    # 验证系统注入了退出原因
    assert "System: Execution stopped" in full_output
    assert "infinite tool loop" in full_output

@pytest.mark.asyncio
async def test_observation_truncation_logic(engine):
    """[New] 测试观测值截断逻辑 (v0.2.1)"""
    engine.max_observation_length = 50
    
    # 模拟超长输出
    long_output = "A" * 100
    tool_calls = [{"name": "t1", "id": "1", "arguments": {}}]
    
    # Mock toolbox return
    engine.toolbox.execute_many = AsyncMock(return_value=[
        ToolExecutionResult("t1", "1", long_output, False)
    ])
    
    ctx = ExecutionContext([])
    await engine._execute_tool_calls(tool_calls, ctx)
    
    # 检查上下文中的消息
    tool_msg = ctx.messages[-1]
    assert len(tool_msg.content) < 100
    assert "truncated" in tool_msg.content