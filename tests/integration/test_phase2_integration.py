# tests/integration/test_phase2_integration.py
"""
Phase 2 综合集成测试

验证所有优化组件协同工作
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.react import ReActEngine
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.plugins.tools.base import BaseTool
from gecko.plugins.storage.sqlite import SQLiteSessionStorage
from gecko.config import GeckoSettings

# ========== 测试工具 ==========

class CalculatorTool(BaseTool):
    """简单计算器工具"""
    name: str = "calculator"
    description: str = "Execute math calculations"
    parameters: dict = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
    
    async def execute(self, arguments: dict) -> str:
        expr = arguments.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

class SlowTool(BaseTool):
    """慢速工具（测试超时）"""
    name: str = "slow_tool"
    description: str = "A slow tool"
    parameters: dict = {"type": "object", "properties": {}}
    
    async def execute(self, arguments: dict) -> str:
        await asyncio.sleep(0.5)
        return "Completed after delay"

# ========== 集成测试 ==========

@pytest.fixture
def mock_model():
    """Mock LLM 模型"""
    model = AsyncMock()
    
    # 默认返回不调用工具的响应
    def make_response(content="Mocked response"):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": content,
            "tool_calls": None
        }
        return mock_response
    
    model.acompletion.return_value = make_response()
    return model

@pytest.fixture
def test_config():
    """测试配置"""
    return GeckoSettings(
        log_level="DEBUG",
        max_turns=3,
        max_context_tokens=2000,
        tool_execution_timeout=2.0
    )

# ========== ReActEngine 集成测试 ==========

@pytest.mark.asyncio
async def test_react_engine_with_hooks(mock_model, test_config):
    """测试 ReActEngine 钩子功能"""
    hook_calls = []
    
    def on_turn_start(context):
        hook_calls.append(("start", context.turn))
    
    def on_turn_end(context):
        hook_calls.append(("end", context.turn))
    
    toolbox = ToolBox([])
    memory = TokenMemory(session_id="test", max_tokens=test_config.max_context_tokens)
    
    engine = ReActEngine(
        model=mock_model,
        toolbox=toolbox,
        memory=memory,
        max_turns=test_config.max_turns,
        on_turn_start=on_turn_start,
        on_turn_end=on_turn_end
    )
    
    await engine.step([Message.user("Hello")])
    
    # 验证钩子被调用
    assert len(hook_calls) >= 2
    assert ("start", 1) in hook_calls
    assert ("end", 1) in hook_calls

@pytest.mark.asyncio
async def test_react_engine_execution_context(mock_model):
    """测试执行上下文传递"""
    toolbox = ToolBox([])
    memory = TokenMemory(session_id="test")
    
    engine = ReActEngine(
        model=mock_model,
        toolbox=toolbox,
        memory=memory
    )
    
    # 执行多轮对话
    await engine.step([Message.user("First question")])
    await engine.step([Message.user("Second question")])
    
    # 验证记忆保存
    assert mock_model.acompletion.call_count >= 2

# ========== Memory 集成测试 ==========

@pytest.mark.asyncio
async def test_memory_with_caching(tmp_path):
    """测试 Memory 缓存功能"""
    storage = SQLiteSessionStorage(f"sqlite:///{tmp_path}/test.db")
    memory = TokenMemory(
        session_id="user_123",
        storage=storage,
        max_tokens=1000,
        cache_size=50
    )
    
    # 创建一些消息
    messages = [
        Message.user(f"Question {i}")
        for i in range(20)
    ]
    
    # 第一次计数（无缓存）
    counts1 = memory.count_messages_batch(messages)
    
    # 第二次计数（有缓存）
    counts2 = memory.count_messages_batch(messages)
    
    # 结果应该一致
    assert counts1 == counts2
    
    # 验证缓存命中
    stats = memory.get_cache_stats()
    assert stats["hits"] > 0
    assert stats["hit_rate"] > 0

@pytest.mark.asyncio
async def test_memory_trimming_with_storage(tmp_path):
    """测试 Memory 修剪与持久化"""
    storage = SQLiteSessionStorage(f"sqlite:///{tmp_path}/test.db")
    memory = TokenMemory(
        session_id="user_456",
        storage=storage,
        max_tokens=500
    )
    
    # 保存大量消息
    long_messages = [
        Message.user("This is a long message " * 20)
        for _ in range(10)
    ]
    
    await storage.set("user_456", {"messages": [m.model_dump() for m in long_messages]})
    
    # 加载并修剪
    trimmed = await memory.get_history([m.model_dump() for m in long_messages])
    
    # 应该被修剪
    assert len(trimmed) < len(long_messages)
    
    # 总 Token 数应该不超过上限
    total_tokens = memory.count_total_tokens(trimmed)
    assert total_tokens <= memory.max_tokens

# ========== Workflow 集成测试 ==========

@pytest.mark.asyncio
async def test_workflow_with_agents(mock_model):
    """测试 Workflow 与 Agent 集成"""
    # 创建两个 Agent
    agent1 = Agent(
        model=mock_model,
        toolbox=ToolBox([]),
        memory=TokenMemory(session_id="agent1")
    )
    
    agent2 = Agent(
        model=mock_model,
        toolbox=ToolBox([]),
        memory=TokenMemory(session_id="agent2")
    )
    
    # 创建 Workflow
    wf = Workflow(name="MultiAgentFlow")
    
    wf.add_node("step1", agent1)
    wf.add_node("step2", agent2)
    wf.add_edge("step1", "step2")
    wf.set_entry_point("step1")
    
    # 执行
    result = await wf.execute("Test input")
    
    # 验证两个 Agent 都被调用
    assert mock_model.acompletion.call_count >= 2

@pytest.mark.asyncio
async def test_workflow_execution_tracking():
    """测试 Workflow 执行追踪"""
    wf = Workflow(name="TrackedFlow")
    
    call_order = []
    
    def node_a(ctx):
        call_order.append("a")
        return "result_a"
    
    def node_b(ctx):
        call_order.append("b")
        return "result_b"
    
    def node_c(ctx):
        call_order.append("c")
        return "result_c"
    
    wf.add_node("a", node_a)
    wf.add_node("b", node_b)
    wf.add_node("c", node_c)
    
    wf.add_edge("a", "b")
    wf.add_edge("b", "c")
    wf.set_entry_point("a")
    
    # 执行并捕获上下文
    class ContextCapture:
        def __init__(self):
            self.context = None
    
    capture = ContextCapture()
    
    # 包装执行以捕获上下文
    original_execute_loop = wf._execute_loop
    
    async def wrapped_execute_loop(ctx):
        result = await original_execute_loop(ctx)
        capture.context = ctx
        return result
    
    wf._execute_loop = wrapped_execute_loop
    
    result = await wf.execute("test")
    
    # 验证执行顺序
    assert call_order == ["a", "b", "c"]
    
    # 验证执行追踪
    assert capture.context is not None
    assert len(capture.context.executions) == 3
    
    summary = capture.context.get_execution_summary()
    assert summary["total_nodes"] == 3
    assert summary["status_counts"]["success"] == 3

@pytest.mark.asyncio
async def test_workflow_with_retry():
    """测试 Workflow 重试机制"""
    wf = Workflow(enable_retry=True, max_retries=3)
    
    attempt_count = 0
    
    def flaky_node(ctx):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise Exception("Temporary failure")
        return "success"
    
    wf.add_node("flaky", flaky_node)
    wf.set_entry_point("flaky")
    
    result = await wf.execute("test")
    
    assert result == "success"
    assert attempt_count == 2

# ========== ToolBox 集成测试 ==========

@pytest.mark.asyncio
async def test_toolbox_concurrent_execution():
    """测试 ToolBox 并发执行"""
    toolbox = ToolBox(
        [CalculatorTool(), SlowTool()],
        max_concurrent=3,
        default_timeout=2.0
    )
    
    tool_calls = [
        {"name": "calculator", "arguments": {"expression": "1+1"}, "id": "call_1"},
        {"name": "calculator", "arguments": {"expression": "2*3"}, "id": "call_2"},
        {"name": "slow_tool", "arguments": {}, "id": "call_3"},
    ]
    
    import time
    start = time.time()
    results = await toolbox.execute_many(tool_calls)
    duration = time.time() - start
    
    # 验证结果
    assert len(results) == 3
    assert results[0]["result"] == "Result: 2"
    assert results[1]["result"] == "Result: 6"
    
    # 并发执行应该快于顺序执行
    # 顺序执行需要 >0.5s，并发应该接近 0.5s
    assert duration < 1.0  # 并发执行

@pytest.mark.asyncio
async def test_toolbox_timeout_handling():
    """测试 ToolBox 超时处理"""
    from gecko.core.exceptions import ToolTimeoutError
    
    class VerySlowTool(BaseTool):
        name: str = "very_slow"
        description: str = "Very slow"
        parameters: dict = {}
        
        async def execute(self, arguments):
            await asyncio.sleep(10)
            return "done"
    
    toolbox = ToolBox([VerySlowTool()], default_timeout=0.5)
    
    with pytest.raises(ToolTimeoutError):
        await toolbox.execute("very_slow", {})

@pytest.mark.asyncio
async def test_toolbox_statistics():
    """测试 ToolBox 统计功能"""
    toolbox = ToolBox([CalculatorTool()])
    
    # 执行一些工具调用
    await toolbox.execute("calculator", {"expression": "1+1"})
    await toolbox.execute("calculator", {"expression": "2+2"})
    await toolbox.execute("calculator", {"expression": "3+3"})
    
    stats = toolbox.get_stats()
    
    assert stats["calculator"]["executions"] == 3
    assert stats["calculator"]["errors"] == 0
    assert stats["calculator"]["success_rate"] == 1.0
    assert stats["calculator"]["avg_time"] > 0

# ========== 完整流程测试 ==========

@pytest.mark.asyncio
async def test_full_agent_with_tools_and_memory(tmp_path):
    """测试完整的 Agent + Tools + Memory 流程"""
    # 准备组件
    storage = SQLiteSessionStorage(f"sqlite:///{tmp_path}/full_test.db")
    toolbox = ToolBox([CalculatorTool()])
    memory = TokenMemory(
        session_id="full_test",
        storage=storage,
        cache_size=100
    )
    
    # Mock 模型返回工具调用
    model = AsyncMock()
    
    # 第一次调用：返回工具调用
    tool_response = MagicMock()
    tool_response.choices = [MagicMock()]
    tool_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "10 + 20"}'
            }
        }]
    }
    
    # 第二次调用：返回最终答案
    final_response = MagicMock()
    final_response.choices = [MagicMock()]
    final_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "The result is 30",
        "tool_calls": None
    }
    
    model.acompletion.side_effect = [tool_response, final_response]
    
    # 创建 Agent
    engine = ReActEngine(
        model=model,
        toolbox=toolbox,
        memory=memory
    )
    
    agent = Agent(
        model=model,
        toolbox=toolbox,
        memory=memory,
        engine_cls=lambda **kwargs: engine
    )
    
    # 执行
    output = await agent.run([Message.user("Calculate 10 + 20")])
    
    # 验证结果
    assert "30" in output.content
    
    # 验证工具被调用
    assert model.acompletion.call_count == 2
    
    # 验证缓存统计
    cache_stats = memory.get_cache_stats()
    assert cache_stats["total_requests"] > 0

@pytest.mark.asyncio
async def test_agent_builder_integration(tmp_path):
    """测试 AgentBuilder 集成"""
    from gecko.plugins.models.zhipu import glm_4_5_air
    
    # 创建存储
    storage = SQLiteSessionStorage(f"sqlite:///{tmp_path}/builder_test.db")
    
    # 使用 Builder 构建 Agent
    agent = (
        AgentBuilder()
        .with_model(glm_4_5_air(temperature=0.3))
        .with_tools([CalculatorTool()])
        .with_session_id("builder_test")
        .with_storage(storage)
        .with_system_prompt("You are a helpful calculator assistant")
        .build()
    )
    
    # 验证组件
    assert agent.toolbox.has_tool("calculator")
    assert agent.memory.session_id == "builder_test"
    assert agent.memory.storage is storage

# ========== 错误恢复测试 ==========

@pytest.mark.asyncio
async def test_error_isolation_in_workflow():
    """测试 Workflow 中的错误隔离"""
    from gecko.core.exceptions import WorkflowError
    
    wf = Workflow(name="ErrorTest")
    
    def good_node(ctx):
        return "success"
    
    def bad_node(ctx):
        raise ValueError("Node failed")
    
    wf.add_node("good", good_node)
    wf.add_node("bad", bad_node)
    wf.add_edge("good", "bad")
    wf.set_entry_point("good")
    
    # 应该抛出 WorkflowError
    with pytest.raises(WorkflowError, match="bad"):
        await wf.execute("test")

@pytest.mark.asyncio
async def test_tool_error_isolation():
    """测试工具错误隔离"""
    class ErrorTool(BaseTool):
        name: str = "error_tool"
        description: str = "Error tool"
        parameters: dict = {}
        
        async def execute(self, arguments):
            raise RuntimeError("Tool error")
    
    toolbox = ToolBox([CalculatorTool(), ErrorTool()])
    
    tool_calls = [
        {"name": "calculator", "arguments": {"expression": "1+1"}},
        {"name": "error_tool", "arguments": {}},
        {"name": "calculator", "arguments": {"expression": "2+2"}},
    ]
    
    results = await toolbox.execute_many(tool_calls)
    
    # 第一个和第三个应该成功
    assert results[0]["is_error"] is False
    assert results[2]["is_error"] is False
    
    # 第二个应该失败
    assert results[1]["is_error"] is True