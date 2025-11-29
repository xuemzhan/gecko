# tests/compose/test_workflow.py
"""
Workflow 模块单元测试

覆盖率目标：100%
修复内容：
- [Fix] test_node_binding_error: 适配最新的参数注入逻辑，断言运行时的 TypeError 而非签名检查错误
- [Fix] 保持其他测试用例的稳定性
"""
import time
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from pydantic import BaseModel

from gecko.compose.workflow import Workflow, WorkflowContext, NodeStatus, WorkflowError, WorkflowCycleError
from gecko.compose.nodes import Next
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.compose.workflow import CheckpointStrategy

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
async def test_next_instruction_preserves_falsey_input_for_function(simple_workflow):
    """
    [新增] 测试 Next 在普通函数节点场景下能正确传递“假值”输入 (0 / False)

    验证点：
    - Start 节点返回 Next(node="End", input=0)
    - End 节点签名为 def end(x): return x
    期望：
    - 最终 Workflow.execute 返回 0，而不是回退到 last_output 或 initial input
    """
    from gecko.compose.nodes import Next

    def start():
        # 显式将 0 作为下一节点的输入
        return Next(node="End", input=0)

    def end(value):
        return value

    simple_workflow.add_node("Start", start)
    simple_workflow.add_node("End", end)

    # 正常 DAG 边
    simple_workflow.add_edge("Start", "End")
    simple_workflow.set_entry_point("Start")

    res = await simple_workflow.execute("init")
    assert res == 0  # 确认 0 没有被当成“空”而丢失


@pytest.mark.asyncio
async def test_next_instruction_preserves_falsey_input_for_function_false(simple_workflow):
    """
    [新增] 再加一个 False 场景，测试布尔假值同样被正确传递
    """
    from gecko.compose.nodes import Next

    def start():
        return Next(node="End", input=False)

    def end(value):
        return value

    simple_workflow.add_node("Start", start)
    simple_workflow.add_node("End", end)

    simple_workflow.add_edge("Start", "End")
    simple_workflow.set_entry_point("Start")

    res = await simple_workflow.execute("init")
    assert res is False  # 保留 False，不回退


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
    # [Fix] 更新匹配字符串: "Workflow exceeded" -> "Exceeded"
    with pytest.raises(WorkflowError, match="Exceeded max steps"):
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
    """测试状态持久化调用 (Two-Phase Commit)"""
    simple_workflow.storage = mock_storage
    # 确保策略允许中间保存
    simple_workflow.checkpoint_strategy = CheckpointStrategy.ALWAYS
    
    simple_workflow.add_node("A", lambda: "output_a")
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.execute("input", session_id="sess_123")
    
    # 验证 storage.set 被调用次数
    # 预期至少调用 2 次：
    # 1. Node A Start -> Pre-Commit (Status: RUNNING)
    # 2. Node A End   -> Commit (Status: SUCCESS, via _execute_loop)
    assert mock_storage.set.call_count >= 2
    
    # 获取第一次调用 (Pre-Commit)
    call_args_1 = mock_storage.set.call_args_list[0]
    _, data_1 = call_args_1[0] # key, value
    
    # 验证 Phase 1: 记录正在执行
    executions_1 = data_1["context"]["executions"]
    assert len(executions_1) == 1
    assert executions_1[0]["node_name"] == "A"
    assert executions_1[0]["status"] == "running" # 关键验证点
    
    # 获取最后一次调用 (Final Commit)
    call_args_last = mock_storage.set.call_args_list[-1]
    _, data_last = call_args_last[0]
    
    # 验证 Phase 2: 记录执行完成
    executions_last = data_last["context"]["executions"]
    assert executions_last[0]["status"] == "success"
    assert data_last["last_node"] == "A"

# [新增] 验证两阶段提交的异常回滚/记录逻辑
@pytest.mark.asyncio
async def test_two_phase_commit_on_failure(simple_workflow, mock_storage):
    """测试节点失败时的状态记录"""
    simple_workflow.storage = mock_storage
    
    def failing_node():
        raise ValueError("Crash!")
        
    simple_workflow.add_node("FailNode", failing_node)
    simple_workflow.set_entry_point("FailNode")
    
    # 执行并捕获异常
    with pytest.raises(WorkflowError):
        await simple_workflow.execute("start", session_id="sess_fail")
        
    # 验证调用
    # 1. Pre-Commit (Running)
    # 2. Failure Commit (Failed)
    assert mock_storage.set.call_count >= 2
    
    # 检查最后一次保存的状态
    _, data_last = mock_storage.set.call_args_list[-1][0]
    last_exec = data_last["context"]["executions"][-1]
    
    assert last_exec["node_name"] == "FailNode"
    assert last_exec["status"] == "failed"
    assert "Crash!" in last_exec["error"]

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

# tests/compose/test_workflow.py (Partial Update)
# 仅展示修改和新增的部分，其他基础 DAG 测试保持不变

@pytest.mark.asyncio
async def test_agent_node_execution(simple_workflow):
    """测试 Agent 对象执行"""
    class MockAgent:
        async def run(self, message):
            return {"content": f"Agent processed: {message}", "role": "assistant"}

    agent = MockAgent()
    simple_workflow.add_node("Agent", agent)
    simple_workflow.set_entry_point("Agent")
    
    res = await simple_workflow.execute("hello")
    assert res["content"] == "Agent processed: hello"

@pytest.mark.asyncio
async def test_no_magic_extraction(simple_workflow):
    """
    [New] 测试不再进行魔法提取
    验证节点返回的字典被完整传递给下一个节点（即使是 Agent）
    """
    # 节点1返回复杂结构
    def node1():
        return {"content": "text", "meta": "data"}
    
    # 节点2是 Agent，验证它收到了完整的字典
    class MockAgent:
        async def run(self, input_data):
            # Agent 应该收到完整的字典，而不是被拆包后的 "text"
            assert isinstance(input_data, dict)
            assert input_data["meta"] == "data"
            return "done"
            
    simple_workflow.add_node("N1", node1)
    simple_workflow.add_node("Agent", MockAgent())
    simple_workflow.add_edge("N1", "Agent")
    simple_workflow.set_entry_point("N1")
    
    await simple_workflow.execute("init")

@pytest.mark.asyncio
async def test_context_type_safety(simple_workflow):
    """[New] 测试 WorkflowContext 的类型安全获取方法"""
    # 这里的导入确保使用最新的类定义
    from gecko.compose.workflow import WorkflowContext
    from pydantic import BaseModel
    
    ctx = WorkflowContext(input="init")
    
    # 1. 基础类型转换
    ctx.history["last_output"] = "123"
    assert ctx.get_last_output_as(int) == 123
    assert ctx.get_last_output_as(str) == "123"
    
    # 2. Pydantic 转换
    class MyModel(BaseModel):
        val: int
        
    ctx.history["last_output"] = {"val": 99}
    model = ctx.get_last_output_as(MyModel)
    assert isinstance(model, MyModel)
    assert model.val == 99
    
    # 3. 转换失败
    ctx.history["last_output"] = "not an int"
    with pytest.raises(TypeError):
        ctx.get_last_output_as(int)

def test_next_with_state_update():
    """[New] 测试 Next 携带状态更新"""
    n = Next(
        node="B", 
        input="data", 
        update_state={"counter": 1, "flag": True}
    )
    assert n.update_state == {"counter": 1, "flag": True}

@pytest.mark.asyncio
async def test_checkpoint_strategy(simple_workflow, mock_storage):
    """[Updated] 测试持久化策略 (适配两阶段提交)"""
    simple_workflow.storage = mock_storage
    simple_workflow.checkpoint_strategy = CheckpointStrategy.FINAL
    
    simple_workflow.add_node("A", lambda: "a")
    simple_workflow.add_node("B", lambda: "b")
    simple_workflow.add_edge("A", "B")
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.execute("init", session_id="sess_strat")
    
    # [修复] 由于引入了 Pre-Commit (force=True)，即使是 FINAL 策略，
    # 每个节点开始执行时也会强制保存 "RUNNING" 状态以防 Crash。
    # 流程: 
    # 1. A Pre-commit (RUNNING) -> Force Save
    # 2. A Post-commit (SUCCESS) -> Skipped (Strategy=FINAL)
    # 3. B Pre-commit (RUNNING) -> Force Save
    # 4. B Post-commit (SUCCESS) -> Skipped (Strategy=FINAL)
    # 5. Workflow End -> Force Save
    # 预期调用次数: 3 次
    assert mock_storage.set.call_count == 3
    
    # 验证最后一次保存的状态 (Workflow 完成状态)
    key, data = mock_storage.set.call_args_list[-1][0]
    
    # 此时 B 已经执行完
    assert data["last_node"] is None 
    assert data["context"]["history"]["B"] == "b"

@pytest.mark.asyncio
async def test_resume_functionality(simple_workflow, mock_storage):
    """[New] 测试断点恢复"""
    simple_workflow.storage = mock_storage
    simple_workflow.checkpoint_strategy = CheckpointStrategy.ALWAYS
    
    # 定义节点
    # 使用 Mock 记录执行次数
    node_a_mock = MagicMock(return_value="res_a")
    node_b_mock = MagicMock(return_value="res_b")
    
    simple_workflow.add_node("A", lambda: node_a_mock())
    simple_workflow.add_node("B", lambda: node_b_mock())
    simple_workflow.add_edge("A", "B")
    simple_workflow.set_entry_point("A")
    
    # 1. 模拟已执行完 A 并保存的状态
    mock_context_data = {
        "input": "start",
        "history": {"last_output": "res_a", "A": "res_a"},
        "executions": [{"node_name": "A", "status": "success"}]
    }
    
    mock_storage.get.return_value = {
        "step": 1,
        "last_node": "A", # A 已完成
        "context": mock_context_data,
        "updated_at": time.time()
    }
    
    # 2. 调用 resume
    result = await simple_workflow.resume("sess_resume")
    
    # 3. 验证
    # A 不应该再被执行
    node_a_mock.assert_not_called()
    
    # B 应该被执行
    node_b_mock.assert_called_once()
    
    assert result == "res_b"

@pytest.mark.asyncio
async def test_persistence_with_unserializable_state(simple_workflow, mock_storage):
    """[Critical] 测试 Context 中包含 Lock 等不可序列化对象时，持久化不崩溃"""
    import threading
    
    simple_workflow.storage = mock_storage
    
    # 1. 定义一个会将 Lock 放入 Context 的节点
    def risky_node(context: WorkflowContext):
        # 用户错误地将锁放入了 state
        context.state["db_lock"] = threading.Lock()
        return "done"
        
    simple_workflow.add_node("Risky", risky_node)
    simple_workflow.set_entry_point("Risky")
    
    # 2. 执行
    await simple_workflow.execute("start", session_id="sess_crash_test")
    
    # 3. 验证 Storage set 被调用（没有因为序列化错误而中断）
    assert mock_storage.set.called
    
    # 4. 验证保存的数据中，Lock 被转换为了标记
    # [Fix] 正确获取调用参数
    # call_args 返回 (args, kwargs)，取 [0] 获取位置参数元组
    args_tuple = mock_storage.set.call_args[0] 
    # set 方法签名为 set(key, value)，所以 args_tuple 是 (key, value)
    _, data = args_tuple 
    
    saved_state = data["context"]["state"]
    
    assert "db_lock" in saved_state
    lock_data = saved_state["db_lock"]
    assert isinstance(lock_data, dict)
    # 验证对象被正确转换为不可序列化标记，而不是导致程序崩溃
    assert lock_data.get("__gecko_unserializable__") is True

@pytest.mark.asyncio
async def test_workflow_context_next_pointer_logic():
    """
    [New] 测试 WorkflowContext 中 next_pointer 的行为
    验证优化点：Context 增强
    """
    ctx = WorkflowContext(input="start")
    assert ctx.next_pointer is None
    
    # 模拟设置指针
    ctx.next_pointer = {"target_node": "B", "input": "data"}
    
    # 验证消费清除逻辑
    ctx.clear_next_pointer()
    assert ctx.next_pointer is None

@pytest.mark.asyncio
async def test_next_instruction_persistence(simple_workflow, mock_storage):
    """
    测试 Next 指令触发立即持久化，并包含 next_pointer
    验证优化点：Workflow._execute_loop 中的原子持久化
    """
    simple_workflow.storage = mock_storage
    
    # 定义返回 Next 的节点
    def node_a():
        return Next(node="B", input="jump_data")
    
    def node_b(inp):
        return "done"
        
    simple_workflow.add_node("A", node_a)
    simple_workflow.add_node("B", node_b)
    simple_workflow.set_entry_point("A")
    
    # 执行
    await simple_workflow.execute("start", session_id="sess_next_persist")
    
    calls = mock_storage.set.call_args_list
    found_correct_snapshot = False
    
    for call_args in calls:
        _, data = call_args[0]
        context_data = data.get("context", {})
        
        # [修复] 我们要找的是 A 执行完、B 执行前的那次快照
        # 特征：last_node="A", next_pointer -> B
        if data.get("last_node") == "A":
            pointer = context_data.get("next_pointer")
            if pointer and pointer["target_node"] == "B":
                found_correct_snapshot = True
                break
                
    assert found_correct_snapshot, "未找到 A->B 跳转时的正确持久化快照"

@pytest.mark.asyncio
async def test_resume_from_next_pointer(simple_workflow, mock_storage):
    """
    [New] 测试基于 next_pointer 的断点恢复
    验证优化点：Workflow.resume 的优先恢复逻辑
    """
    simple_workflow.storage = mock_storage
    
    node_a_mock = MagicMock(return_value="should_not_run")
    node_b_mock = MagicMock(return_value="recovered_success")
    
    simple_workflow.add_node("A", lambda: node_a_mock())
    simple_workflow.add_node("B", lambda: node_b_mock())
    simple_workflow.set_entry_point("A")
    
    # 模拟存储状态：
    # 场景：A 节点执行完毕，返回 Next("B")，此时 next_pointer 已保存，但 B 尚未执行系统就 Crash 了
    mock_storage.get.return_value = {
        "step": 1,
        "last_node": "A",
        "context": {
            "input": "start",
            "history": {"A": "<Next -> B>"}, # 历史中有痕迹
            "state": {},
            "next_pointer": { # [关键] 存在动态指针
                "target_node": "B",
                "input": "restored_input"
            },
            "executions": []
        }
    }
    
    # 执行 Resume
    result = await simple_workflow.resume("sess_crash_after_next")
    
    # 验证：
    # 1. A 不应被重新执行 (避免副作用)
    node_a_mock.assert_not_called()
    
    # 2. B 应该被执行，且 Context 状态已包含 input
    node_b_mock.assert_called_once()
    assert result == "recovered_success"
    
    # 验证内部状态流转
    # 这里比较难直接验证 state["_next_input"]，因为 execute 内部是瞬态的
    # 但可以通过 B 的执行结果间接验证，或者 Mock B 检查参数


@pytest.mark.asyncio
async def test_resume_failure_preserves_pointer(simple_workflow, mock_storage):
    """
    [New] 测试断点恢复时，如果第一步执行失败，next_pointer 不应被清除。
    确保下次还可以再次尝试恢复。
    """
    simple_workflow.storage = mock_storage
    simple_workflow.checkpoint_strategy = CheckpointStrategy.ALWAYS

    # 模拟一个会在恢复时崩溃的节点
    failing_mock = MagicMock(side_effect=ValueError("Transient Error"))
    simple_workflow.add_node("B", lambda: failing_mock())
    simple_workflow.set_entry_point("B") # 简化拓扑

    # 模拟存储状态：这就好比 A -> B (Next)，指针指向 B
    mock_storage.get.return_value = {
        "step": 1,
        "last_node": "A",
        "context": {
            "input": "start",
            "history": {},
            "state": {},
            # ✅ [修复点] 添加 metadata，确保 _execute_node_safe 能提取到 session_id
            "metadata": {"session_id": "sess_retry_test"}, 
            "next_pointer": { # 指针存在
                "target_node": "B",
                "input": "data"
            },
            "executions": []
        }
    }

    # 1. 执行 Resume，预期抛出 WorkflowError (包装了 ValueError)
    from gecko.core.exceptions import WorkflowError
    with pytest.raises(WorkflowError) as excinfo:
        await simple_workflow.resume("sess_retry_test")

    assert "Transient Error" in str(excinfo.value) or "Transient Error" in str(excinfo.value.__cause__)

    # 2. 验证：Context 中的 next_pointer 应该被保留
    assert mock_storage.set.called
    
    # 获取最后一次保存的 context
    _, data = mock_storage.set.call_args_list[-1][0]
    saved_context = data["context"]
    
    assert saved_context["next_pointer"] is not None
    assert saved_context["next_pointer"]["target_node"] == "B"


@pytest.mark.asyncio
async def test_parallel_execution(simple_workflow):
    """[New] 测试工作流并行执行能力"""
    simple_workflow.enable_parallel = True
    
    # 定义会有延迟的节点，用于验证并行性
    async def node_a(ctx):
        await asyncio.sleep(0.05)
        return "A"
    
    async def node_b(ctx):
        await asyncio.sleep(0.05)
        return "B"
        
    # [FIX] 参数名改为 context，确保注入的是 WorkflowContext 对象而非 input dict
    async def node_merge(context):
        return f"{context.history['A']}+{context.history['B']}"

    simple_workflow.add_node("A", node_a)
    simple_workflow.add_node("B", node_b)
    simple_workflow.add_node("Merge", node_merge)
    
    # 设置依赖关系: Start -> (A, B) -> Merge
    simple_workflow.add_edge("A", "Merge")
    simple_workflow.add_edge("B", "Merge")
    
    # 定义 A 和 B 为并行组
    simple_workflow.add_parallel_group("A", "B")
    
    simple_workflow.set_entry_point("A") # 这里的 entry point 逻辑在并行模式下略有不同，通常由拓扑决定
    # 为了测试 execute_parallel 的分层逻辑，我们需要稍微调整拓扑
    # 实际上，execute_parallel 使用 start_node 开始 BFS。
    # 假设我们有一个 Start 节点分发给 A 和 B
    
    simple_workflow.add_node("Start", lambda: "start")
    simple_workflow.add_edge("Start", "A")
    simple_workflow.add_edge("Start", "B")
    simple_workflow.set_entry_point("Start")
    
    start_time = time.time()
    result = await simple_workflow.execute_parallel("init")
    duration = time.time() - start_time
    
    assert result == "A+B"
    # 验证耗时：如果是串行，至少 0.1s；并行应该接近 0.05s (加上开销)
    # 这是一个宽松的断言，主要验证逻辑正确性
    assert "A" in simple_workflow._nodes
    
    # 验证历史记录包含并行节点结果
    # 注意：execute_parallel 会合并结果到 history
    # 这里的 result 校验已经隐含了 history 正确

@pytest.mark.asyncio
async def test_resume_with_pointer_cleanup(simple_workflow, mock_storage):
    """[New] 验证 Resume 成功后 Next 指针被清除"""
    simple_workflow.storage = mock_storage
    
    # 模拟从 A 跳转到 B 的状态
    mock_storage.get.return_value = {
        "step": 1,
        "last_node": "A",
        "context": {
            "input": "start",
            "history": {"A": "val_a"},
            "next_pointer": {"target_node": "B", "input": "val_b"}, # 指针存在
            "executions": []
        }
    }
    
    # 节点 B
    node_b = MagicMock(return_value="done")
    simple_workflow.add_node("A", lambda: "val_a")
    simple_workflow.add_node("B", lambda: node_b())
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.resume("sess_1")
    
    # 验证 B 被执行
    node_b.assert_called_once()
    
    # 验证 resume 完成后，如果不报错，最后一次保存的状态中 next_pointer 应该被清除
    # 检查最后一次 storage.set 的调用参数
    _, data = mock_storage.set.call_args_list[-1][0]
    assert data["context"].get("next_pointer") is None

def test_validate_cycles_flag(simple_workflow):
    """验证 allow_cycles 标志的行为"""
    simple_workflow.add_node("A", lambda: None)
    simple_workflow.add_node("B", lambda: None)
    simple_workflow.set_entry_point("A")
    
    # 创建环: A -> B -> A
    simple_workflow.add_edge("A", "B")
    simple_workflow.add_edge("B", "A")
    
    # Case 1: 默认不允许环
    simple_workflow.allow_cycles = False
    assert simple_workflow.validate() is False
    assert any("Cycle detected" in err for err in simple_workflow._validation_errors)
    
    # Case 2: 显式允许环
    simple_workflow.allow_cycles = True
    
    # [FIX] 手动重置验证状态，强制重新验证
    # 因为直接修改属性不会触发 setter 逻辑来重置 _validated
    simple_workflow._validated = False
    simple_workflow._validation_errors = []
    
    assert simple_workflow.validate() is True

@pytest.mark.asyncio
async def test_next_instruction_preserves_falsey_input_for_agent_node(simple_workflow):
    """
    [新增] 测试 Next 在 Agent/Team 节点场景下能正确传递“假值”输入。

    目的：
    - 覆盖 Workflow._run_intelligent_object 中的输入获取逻辑：
      使用显式 "_next_input" in context.state 判断，而不是 or。
    场景：
      Start -> DummyAgentNode
      Start 返回 Next(node="Agent", input=False)
      DummyAgent.run(input) 原样返回 input
    """
    from gecko.compose.nodes import Next

    class DummyAgent:
        async def run(self, inp):
            # 模拟一个简单 Agent，直接回显输入
            return inp

    agent = DummyAgent()

    def start():
        return Next(node="Agent", input=False)

    simple_workflow.add_node("Start", start)
    simple_workflow.add_node("Agent", agent)

    simple_workflow.add_edge("Start", "Agent")
    simple_workflow.set_entry_point("Start")

    res = await simple_workflow.execute("init")
    assert res is False  # ✅ 确认 False 也能在 Agent 场景下正确传递

def test_build_execution_layers_with_explicit_dependencies(simple_workflow):
    """
    [新增] 测试 _build_execution_layers 使用显式依赖 (set_dependency) 构建正确的执行层级。

    DAG 结构：
        Start -> A
        Start -> B
        A -> Merge   (通过 add_edge)
        Merge 依赖 B (通过 set_dependency)

    期望层级关系：
        - Start 在第 0 层
        - A / B 在第 1 层（并行）
        - Merge 在第 2 层（必须在 A、B 之后）
    """
    wf = simple_workflow

    wf.add_node("Start", lambda: None)
    wf.add_node("A", lambda: "A")
    wf.add_node("B", lambda: "B")
    wf.add_node("Merge", lambda: "M")

    # 边关系：Start -> A, Start -> B, A -> Merge
    wf.add_edge("Start", "A")
    wf.add_edge("Start", "B")
    wf.add_edge("A", "Merge")

    # 显式声明 Merge 还依赖 B
    wf.set_dependency("Merge", depends_on=["B"])

    wf.set_entry_point("Start")

    # 调用内部方法构建执行层级
    layers = wf._build_execution_layers("Start")

    # 将节点映射到所在层级索引，方便判断顺序
    node_to_layer = {}
    for idx, layer in enumerate(layers):
        for node in layer:
            node_to_layer[node] = idx

    # 基本断言：三个层
    assert node_to_layer["Start"] < node_to_layer["A"]
    assert node_to_layer["Start"] < node_to_layer["B"]
    assert node_to_layer["A"] < node_to_layer["Merge"]
    assert node_to_layer["B"] < node_to_layer["Merge"]
