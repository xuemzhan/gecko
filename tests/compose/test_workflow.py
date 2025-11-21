# tests/compose/test_workflow.py
"""
Workflow 模块单元测试

覆盖率目标：100%
修复内容：
- [Fix] test_node_binding_error: 适配最新的参数注入逻辑，断言运行时的 TypeError 而非签名检查错误
- [Fix] 保持其他测试用例的稳定性
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from pydantic import BaseModel

from gecko.compose.workflow import Workflow, WorkflowContext, NodeStatus, WorkflowError, WorkflowCycleError
from gecko.compose.nodes import Next
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface

# ========================= Fixtures =========================

@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=SessionInterface)
    storage.set = AsyncMock()
    return storage

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus

@pytest.fixture
def simple_workflow(mock_event_bus):
    return Workflow(name="TestWorkflow", event_bus=mock_event_bus)

# ========================= 1. DAG 构建与验证测试 =========================

def test_add_duplicate_node(simple_workflow):
    """测试添加重复节点抛出异常"""
    simple_workflow.add_node("A", lambda: None)
    with pytest.raises(ValueError, match="already exists"):
        simple_workflow.add_node("A", lambda: None)

def test_add_edge_missing_nodes(simple_workflow):
    """测试添加边时节点不存在"""
    simple_workflow.add_node("A", lambda: None)
    with pytest.raises(ValueError, match="Source node 'B' not found"):
        simple_workflow.add_edge("B", "A")
    with pytest.raises(ValueError, match="Target node 'B' not found"):
        simple_workflow.add_edge("A", "B")

def test_set_invalid_entry_point(simple_workflow):
    """测试设置不存在的入口节点"""
    with pytest.raises(ValueError, match="Node 'X' not found"):
        simple_workflow.set_entry_point("X")

def test_validate_no_entry_point(simple_workflow):
    """测试验证：无入口点"""
    simple_workflow.add_node("A", lambda: None)
    assert simple_workflow.validate() is False
    assert "No entry point defined" in simple_workflow._validation_errors

def test_validate_cycle_detection(simple_workflow):
    """测试验证：环检测"""
    simple_workflow.add_node("A", lambda: None)
    simple_workflow.add_node("B", lambda: None)
    simple_workflow.set_entry_point("A")
    
    simple_workflow.add_edge("A", "B")
    simple_workflow.add_edge("B", "A")  # A->B->A 环
    
    assert simple_workflow.validate() is False
    assert any("Cycle detected" in err for err in simple_workflow._validation_errors)

def test_validate_ambiguous_edges(simple_workflow):
    """测试验证：静态歧义检测（同一个节点多个无条件出边）"""
    simple_workflow.add_node("A", lambda: None)
    simple_workflow.add_node("B", lambda: None)
    simple_workflow.add_node("C", lambda: None)
    simple_workflow.set_entry_point("A")
    
    simple_workflow.add_edge("A", "B")
    simple_workflow.add_edge("A", "C")  # 第二个无条件出边
    
    assert simple_workflow.validate() is False
    assert any("ambiguous edges" in err for err in simple_workflow._validation_errors)

def test_check_connectivity_warning(simple_workflow):
    """测试验证：不可达节点警告 (使用 Mock Logger)"""
    with patch("gecko.compose.workflow.logger") as mock_logger:
        simple_workflow.add_node("A", lambda: None)
        simple_workflow.add_node("B", lambda: None) # B 不可达
        simple_workflow.set_entry_point("A")
        
        assert simple_workflow.validate() is True # 只是警告，验证通过
        
        # 验证 logger.warning 被调用
        mock_logger.warning.assert_called()
        args, kwargs = mock_logger.warning.call_args
        assert "Unreachable nodes detected" in args[0] or "nodes" in kwargs

# ========================= 2. 节点执行策略测试 (Smart Binding) =========================

@pytest.mark.asyncio
async def test_node_binding_strategies(simple_workflow):
    """测试不同签名的函数能否被正确调用"""
    
    # 1. 接收 context
    async def node_context(context: WorkflowContext):
        return f"ctx:{context.input}"
    
    # 2. 接收 input
    def node_input(text: str):
        return f"in:{text}"
    
    # 3. 无参
    async def node_empty():
        return "empty"
    
    # 4. 接收 workflow_context
    def node_wf_ctx(workflow_context: WorkflowContext):
        return f"wf:{workflow_context.input}"

    simple_workflow.add_node("Start", node_context)
    simple_workflow.add_node("Input", node_input)
    simple_workflow.add_node("Empty", node_empty)
    simple_workflow.add_node("WfCtx", node_wf_ctx)
    
    simple_workflow.add_edge("Start", "Input")
    simple_workflow.add_edge("Input", "Empty")
    simple_workflow.add_edge("Empty", "WfCtx")
    
    simple_workflow.set_entry_point("Start")
    
    res = await simple_workflow.execute("start_val")
    
    assert res == "wf:start_val"

@pytest.mark.asyncio
async def test_node_binding_error(simple_workflow):
    """测试函数签名参数不足导致的运行时错误"""
    # 定义一个需要两个参数的函数
    def bad_node(a, b):
        return a + b
        
    simple_workflow.add_node("Bad", bad_node)
    simple_workflow.set_entry_point("Bad")
    
    # 修正点：现在的逻辑是尝试执行，Python 会抛出 TypeError (missing argument)，
    # 最终被 Workflow 捕获并包装为 "Node 'Bad' failed"
    with pytest.raises(WorkflowError, match="Node 'Bad' failed"):
        await simple_workflow.execute("init")

@pytest.mark.asyncio
async def test_agent_node_execution(simple_workflow):
    """测试 Agent 对象（具备 run 方法）的执行与数据流转"""
    
    class MockAgent:
        async def run(self, message):
            # 模拟 Agent 接收输入
            return {"content": f"Agent processed: {message}", "role": "assistant"}

    agent = MockAgent()
    simple_workflow.add_node("Agent", agent)
    simple_workflow.set_entry_point("Agent")
    
    res = await simple_workflow.execute("hello")
    assert res["content"] == "Agent processed: hello"

@pytest.mark.asyncio
async def test_data_handover_extraction(simple_workflow):
    """测试数据交接：从上一个节点的字典输出中提取 content 给下一个 Agent"""
    
    # 节点1返回字典
    def node1():
        return {"content": "pure text", "metadata": 123}
    
    # 节点2是 Agent，应该只收到 "pure text"
    class MockAgent:
        async def run(self, text):
            assert text == "pure text" # 关键断言
            return "done"
            
    simple_workflow.add_node("N1", node1)
    simple_workflow.add_node("Agent", MockAgent())
    simple_workflow.add_edge("N1", "Agent")
    simple_workflow.set_entry_point("N1")
    
    await simple_workflow.execute("init")

# ========================= 3. 控制流逻辑测试 =========================

@pytest.mark.asyncio
async def test_conditional_branching(simple_workflow):
    """测试条件分支"""
    simple_workflow.add_node("Start", lambda x: x)
    simple_workflow.add_node("PathA", lambda: "A")
    simple_workflow.add_node("PathB", lambda: "B")
    
    # 路由逻辑：输入 > 5 走 A，否则走 B
    simple_workflow.add_edge("Start", "PathA", lambda ctx: ctx.get_last_output() > 5)
    simple_workflow.add_edge("Start", "PathB", lambda ctx: ctx.get_last_output() <= 5)
    
    simple_workflow.set_entry_point("Start")
    
    # Case 1: > 5
    res_a = await simple_workflow.execute(10)
    assert res_a == "A"
    
    # Case 2: <= 5
    res_b = await simple_workflow.execute(3)
    assert res_b == "B"

@pytest.mark.asyncio
async def test_next_instruction(simple_workflow):
    """测试 Next 指令：跳转与参数注入"""
    
    def start():
        # 跳转到 End，并注入新输入
        return Next(node="End", input="jumped input")
    
    def skipped():
        return "skipped"
        
    def end(inp):
        return f"Received: {inp}"
        
    simple_workflow.add_node("Start", start)
    simple_workflow.add_node("Skipped", skipped)
    simple_workflow.add_node("End", end)
    
    simple_workflow.add_edge("Start", "Skipped") # 正常 DAG 边
    simple_workflow.add_edge("Skipped", "End")
    
    simple_workflow.set_entry_point("Start")
    
    res = await simple_workflow.execute("init")
    assert res == "Received: jumped input"

@pytest.mark.asyncio
async def test_runtime_ambiguity_error(simple_workflow):
    """测试运行时歧义：多个条件同时满足"""
    simple_workflow.add_node("Start", lambda: 10)
    simple_workflow.add_node("A", lambda: "A")
    simple_workflow.add_node("B", lambda: "B")
    
    # 两个条件都为真
    simple_workflow.add_edge("Start", "A", lambda ctx: True)
    simple_workflow.add_edge("Start", "B", lambda ctx: True)
    
    simple_workflow.set_entry_point("Start")
    
    with pytest.raises(WorkflowError, match="Ambiguous branching"):
        await simple_workflow.execute(None)

@pytest.mark.asyncio
async def test_condition_evaluation_error(simple_workflow):
    """测试条件评估抛出异常 (使用 Mock Logger)"""
    def bad_condition(ctx):
        raise ValueError("Eval failed")
        
    with patch("gecko.compose.workflow.logger") as mock_logger:
        simple_workflow.add_node("Start", lambda: 1)
        simple_workflow.add_node("A", lambda: "A")
    
        simple_workflow.add_edge("Start", "A", bad_condition)
        simple_workflow.set_entry_point("Start")
    
        # 因为没有可行的路径（条件报错视为不满足），流程会在 Start 后结束
        res = await simple_workflow.execute(None)
        
        assert res == 1 # 停留在 Start 的输出
        
        # 验证错误日志记录
        mock_logger.error.assert_called()
        args, kwargs = mock_logger.error.call_args
        assert "Condition evaluation failed" in args[0]

# ========================= 4. 异常处理与重试机制 =========================

@pytest.mark.asyncio
async def test_node_execution_failure(simple_workflow):
    """测试节点执行失败抛出 WorkflowError"""
    def failing_node():
        raise RuntimeError("Node crashed")
        
    simple_workflow.add_node("Fail", failing_node)
    simple_workflow.set_entry_point("Fail")
    
    with pytest.raises(WorkflowError, match="Node 'Fail' failed"):
        await simple_workflow.execute(None)

@pytest.mark.asyncio
async def test_max_steps_exceeded(simple_workflow):
    """测试超过最大步数 (使用 Next 构造动态环)"""
    simple_workflow.max_steps = 2
    
    # 使用 Next 构造动态循环，避开 validate 阶段的静态环检测
    def node_a():
        return Next(node="B")
    
    def node_b():
        return Next(node="A")
        
    simple_workflow.add_node("A", node_a)
    simple_workflow.add_node("B", node_b)
    simple_workflow.set_entry_point("A")
    
    # 预期运行：A(1) -> B(2) -> A(3 > max 2) -> Error
    with pytest.raises(WorkflowError, match="Workflow exceeded max steps"):
        await simple_workflow.execute(None)

@pytest.mark.asyncio
async def test_retry_mechanism(simple_workflow):
    """测试节点重试机制"""
    simple_workflow.enable_retry = True
    simple_workflow.max_retries = 3
    
    mock_func = MagicMock(side_effect=[ValueError("Fail 1"), ValueError("Fail 2"), "Success"])
    
    # 包装为异步
    async def flaky_node():
        return mock_func()
        
    simple_workflow.add_node("Flaky", flaky_node)
    simple_workflow.set_entry_point("Flaky")
    
    res = await simple_workflow.execute(None)
    
    assert res == "Success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_execute_not_validated(simple_workflow):
    """测试执行前未通过验证"""
    # 不设置 entry point
    simple_workflow.add_node("A", lambda: 1)
    with pytest.raises(WorkflowError, match="Workflow validation failed"):
        await simple_workflow.execute(None)

# ========================= 5. 状态持久化测试 =========================

@pytest.mark.asyncio
async def test_persistence(simple_workflow, mock_storage):
    """测试状态持久化调用"""
    simple_workflow.storage = mock_storage
    
    simple_workflow.add_node("A", lambda: "output_a")
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.execute("input", session_id="sess_123")
    
    # 验证 storage.set 被调用
    assert mock_storage.set.called
    call_args = mock_storage.set.call_args[0]
    key, data = call_args
    
    assert key == "workflow:sess_123"
    assert data["step"] == 1
    
    # 只有 A 一个节点，执行完后 find_next_node 返回 None
    # 因此持久化时的 last_node 状态为 None（表示指针已移出）
    assert data["last_node"] is None 
    
    # 但历史记录里必须有 A
    assert data["context"]["input"] == "input"
    assert data["context"]["history"]["A"] == "output_a"

@pytest.mark.asyncio
async def test_persistence_failure_safe(simple_workflow, mock_storage):
    """测试持久化失败不影响主流程"""
    with patch("gecko.compose.workflow.logger") as mock_logger:
        simple_workflow.storage = mock_storage
        mock_storage.set.side_effect = Exception("DB Error")
    
        simple_workflow.add_node("A", lambda: "ok")
        simple_workflow.set_entry_point("A")
    
        res = await simple_workflow.execute("in", session_id="sess_fail")
    
        assert res == "ok" # 主流程成功
        
        # 验证警告日志
        mock_logger.warning.assert_called()
        args, kwargs = mock_logger.warning.call_args
        assert "Failed to persist workflow state" in args[0]

# ========================= 6. 其他功能测试 =========================

def test_result_normalization(simple_workflow):
    """测试结果标准化逻辑"""
    # Pydantic Model
    class MyModel(BaseModel):
        val: int
    assert simple_workflow._normalize_result(MyModel(val=1)) == {"val": 1}
    
    # Message
    msg = Message.user("hello")
    assert simple_workflow._normalize_result(msg)["content"] == "hello"
    
    # Primitive
    assert simple_workflow._normalize_result(123) == 123

def test_visualization(simple_workflow):
    """测试 Mermaid 和 Print Structure 生成"""
    simple_workflow.add_node("Start", lambda: None)
    simple_workflow.add_node("End", lambda: None)
    simple_workflow.add_edge("Start", "End", lambda ctx: True)
    simple_workflow.set_entry_point("Start")
    
    mermaid = simple_workflow.to_mermaid()
    assert "Start((Start))" in mermaid
    assert "Start --|condition|--> End" in mermaid
    
    # 简单调用 print_structure 确保不报错
    simple_workflow.print_structure()

@pytest.mark.asyncio
async def test_context_summary_and_output():
    """测试 WorkflowContext 的辅助方法"""
    ctx = WorkflowContext(input="init")
    assert ctx.get_last_output() == "init"
    
    ctx.history["last_output"] = "step1"
    assert ctx.get_last_output() == "step1"
    
    summary = ctx.get_summary()
    assert summary["status"] == "completed" # default empty executions
    assert summary["total_nodes"] == 0

# ========================= 7. 不可调用对象测试 =========================

@pytest.mark.asyncio
async def test_not_callable_node(simple_workflow):
    """测试添加了不可调用的对象作为节点"""
    simple_workflow.add_node("NotFunc", "I am a string")
    simple_workflow.set_entry_point("NotFunc")
    
    with pytest.raises(WorkflowError, match="is not callable"):
        await simple_workflow.execute(None)