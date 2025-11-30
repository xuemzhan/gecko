"""
Workflow 模块单元测试 (Refactored for v0.3 Modular Architecture)

覆盖范围：
1. Graph: DAG 构建、环检测、拓扑验证
2. Executor: 参数注入、重试、结果标准化
3. Persistence: 上下文瘦身、两阶段提交、异步清洗
4. Engine: 主循环、断点恢复 (Resume)、外观模式 (Facade)
"""
import time
import pytest
import asyncio
import threading
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import BaseModel

from gecko.compose.workflow import (
    Workflow, 
    WorkflowContext, 
    NodeStatus, 
    CheckpointStrategy
)
from gecko.compose.nodes import Next
from gecko.core.exceptions import WorkflowError, WorkflowCycleError
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface

# ========================= Fixtures =========================

@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=SessionInterface)
    storage.set = AsyncMock()
    storage.get = AsyncMock()
    return storage

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus

@pytest.fixture
def simple_workflow(mock_event_bus):
    """创建一个基础 Workflow 实例"""
    return Workflow(name="TestWorkflow", event_bus=mock_event_bus)

# ========================= 1. Graph: 结构与验证 =========================

def test_add_duplicate_node(simple_workflow):
    """测试添加重复节点"""
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
    """测试设置非法入口"""
    with pytest.raises(ValueError, match="not found"):
        simple_workflow.set_entry_point("X")

def test_validate_basic(simple_workflow):
    """测试基础验证逻辑"""
    simple_workflow.add_node("A", lambda: None)
    # 无入口点
    assert simple_workflow.validate() is False
    
    simple_workflow.set_entry_point("A")
    assert simple_workflow.validate() is True

def test_validate_cycle_detection(simple_workflow):
    """测试环检测 (Facade -> Graph)"""
    simple_workflow.add_node("A", lambda: None)
    simple_workflow.add_node("B", lambda: None)
    simple_workflow.set_entry_point("A")
    
    simple_workflow.add_edge("A", "B")
    simple_workflow.add_edge("B", "A")
    
    # 默认不允许环
    assert simple_workflow.validate() is False
    
    # 允许环
    simple_workflow.allow_cycles = True
    assert simple_workflow.validate() is True

def test_check_connectivity_warning(simple_workflow):
    """
    测试不可达节点警告
    [Fix] Patch 路径修正为 gecko.compose.workflow.graph
    """
    with patch("gecko.compose.workflow.graph.logger") as mock_logger:
        simple_workflow.add_node("A", lambda: None)
        simple_workflow.add_node("B", lambda: None) # 孤立点
        simple_workflow.set_entry_point("A")
        
        simple_workflow.validate()
        
        mock_logger.warning.assert_called()
        args, _ = mock_logger.warning.call_args
        assert "Unreachable nodes detected" in args[0]

def test_visualization_facade(simple_workflow):
    """测试可视化代理方法"""
    simple_workflow.add_node("Start", lambda: None)
    simple_workflow.set_entry_point("Start")
    assert "Start((Start))" in simple_workflow.to_mermaid()

# ========================= 2. Executor: 执行与绑定 =========================

@pytest.mark.asyncio
async def test_node_binding_strategies(simple_workflow):
    """测试智能参数绑定 (Smart Binding)"""
    
    # 1. 接收 context
    async def node_ctx(context: WorkflowContext):
        return f"ctx:{context.input}"
    
    # 2. [Fix] 接收 workflow_context (旧 API 兼容)
    def node_wf_ctx(workflow_context: WorkflowContext):
        return f"wf:{workflow_context.input}"
    
    # 3. 接收 input (第一个未绑定参数)
    def node_input(val):
        return f"in:{val}"

    simple_workflow.add_node("A", node_ctx)
    simple_workflow.add_node("B", node_wf_ctx)
    simple_workflow.add_node("C", node_input)
    
    simple_workflow.add_edge("A", "B")
    simple_workflow.add_edge("B", "C")
    simple_workflow.set_entry_point("A")
    
    # 执行流: A -> B -> C
    # Input 传递: "start" -> "ctx:start" -> "wf:start" -> "in:wf:start"
    # 注意: 中间结果会被当作下一个节点的 input
    
    # 我们需要这里稍微 Mock 一下逻辑，或者直接断言最后一步
    # 由于 A, B 返回值会被放入 history['last_output']，C 会读取它
    
    # 为了简化测试，我们分别测试单独执行，或者直接跑通
    # 这里跑通整个流程：
    res = await simple_workflow.execute("start")
    
    # A 返回 "ctx:start"
    # B 接收 "ctx:start", 返回 "wf:start" (注意：B 这里实际上没有用到 last_output 作为参数，只用了 context.input)
    # 等等，node_wf_ctx 读取的是 workflow_context.input (原始输入 "start")
    # 所以 B 返回 "wf:start"
    # C 接收 B 的输出 "wf:start", 返回 "in:wf:start"
    
    assert res == "in:wf:start"

@pytest.mark.asyncio
async def test_node_execution_failure(simple_workflow):
    """测试节点执行失败被正确包装"""
    def crash():
        raise ValueError("Boom")
        
    simple_workflow.add_node("Crash", crash)
    simple_workflow.set_entry_point("Crash")
    
    with pytest.raises(WorkflowError, match="Workflow execution failed"):
        await simple_workflow.execute(None)

@pytest.mark.asyncio
async def test_retry_mechanism(simple_workflow):
    """测试重试机制"""
    # 通过 Engine 初始化 Executor 参数
    wf = Workflow("RetryWF", enable_retry=True, max_retries=2)
    
    mock_func = MagicMock(side_effect=[ValueError("Fail 1"), "Success"])
    wf.add_node("Flaky", lambda: mock_func())
    wf.set_entry_point("Flaky")
    
    res = await wf.execute(None)
    assert res == "Success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_next_instruction_with_falsey_input(simple_workflow):
    """测试 Next 指令传递 0 或 False"""
    def start():
        return Next(node="End", input=0)
    
    def end(val):
        return val
        
    simple_workflow.add_node("Start", start)
    simple_workflow.add_node("End", end)
    simple_workflow.set_entry_point("Start")
    simple_workflow.add_edge("Start", "End")
    
    res = await simple_workflow.execute("init")
    assert res == 0 # 确保 0 没被当成 None

# ========================= 3. Persistence: 瘦身与存储 =========================

def test_context_slimming():
    """
    [New] 测试上下文瘦身逻辑 (WorkflowContext.to_storage_payload)
    """
    ctx = WorkflowContext(input="init")
    
    # 1. 模拟大量历史
    for i in range(50):
        ctx.history[f"node_{i}"] = "data" * 100
        
    # 2. 模拟执行轨迹 (Trace)
    from gecko.compose.workflow.models import NodeExecution
    ctx.add_execution(NodeExecution(node_name="A"))
    
    # 执行瘦身 (保留最近 5 条)
    payload = ctx.to_storage_payload(max_history_steps=5)
    
    # 断言
    assert "executions" not in payload  # 轨迹应被移除
    assert len(payload["history"]) == 5 # 历史应被裁剪
    assert "node_49" in payload["history"] # 最新的保留
    assert "node_0" not in payload["history"] # 旧的丢弃

@pytest.mark.asyncio
async def test_persistence_two_phase_commit(simple_workflow, mock_storage):
    """测试两阶段提交 (RUNNING -> SUCCESS)"""
    simple_workflow.storage = mock_storage
    simple_workflow.checkpoint_strategy = CheckpointStrategy.ALWAYS
    
    simple_workflow.add_node("A", lambda: "ok")
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.execute("in", session_id="sess_2pc")
    
    # 至少 2 次保存: Pre-Commit (Running) + Post-Commit (Success)
    assert mock_storage.set.call_count >= 2
    
    # 检查第一次保存 (Pre-Commit)
    # set(key, value) -> call_args[0][1] 是 value
    first_call_data = mock_storage.set.call_args_list[0][0][1]
    # 此时 executions 应该被移除了 (Context Slimming 生效)
    # 如果要验证 status=RUNNING，需要检查 context 对象本身，但它被序列化了。
    # 我们主要验证 to_storage_payload 被调用，即 executions 字段不在 payload 中
    assert "executions" not in first_call_data["context"]

@pytest.mark.asyncio
async def test_persistence_with_unserializable(simple_workflow, mock_storage):
    """测试不可序列化对象 (Lock) 不会导致崩溃"""
    simple_workflow.storage = mock_storage
    
    # 参数名从 ctx 改为 context，以触发 Executor 的 Context 注入
    def risky_node(context):
        context.state["lock"] = threading.Lock()
        return "done"
        
    simple_workflow.add_node("A", risky_node)
    simple_workflow.set_entry_point("A")
    
    await simple_workflow.execute("in", session_id="sess_lock")
    
    # 验证保存成功
    assert mock_storage.set.called
    # 验证最后一次保存的数据
    last_data = mock_storage.set.call_args[0][1]
    saved_lock = last_data["context"]["state"]["lock"]
    
    # 应该被转换为标记字典
    assert saved_lock.get("__gecko_unserializable__") is True

@pytest.mark.asyncio
async def test_persistence_failure_safe(simple_workflow, mock_storage):
    """
    测试存储失败不中断流程
    [Fix] Patch 路径修正为 gecko.compose.workflow.persistence
    """
    with patch("gecko.compose.workflow.persistence.logger") as mock_logger:
        simple_workflow.storage = mock_storage
        mock_storage.set.side_effect = Exception("DB Down")
        
        simple_workflow.add_node("A", lambda: "ok")
        simple_workflow.set_entry_point("A")
        
        res = await simple_workflow.execute("in", session_id="sess_fail")
        assert res == "ok"
        
        mock_logger.warning.assert_called()

# ========================= 4. Engine: Resume & Branching =========================

@pytest.mark.asyncio
async def test_resume_functionality(simple_workflow, mock_storage):
    """测试断点恢复"""
    simple_workflow.storage = mock_storage
    
    node_a = MagicMock(return_value="A")
    node_b = MagicMock(return_value="B")
    
    simple_workflow.add_node("A", lambda: node_a())
    simple_workflow.add_node("B", lambda: node_b())
    simple_workflow.add_edge("A", "B")
    simple_workflow.set_entry_point("A")
    
    # 模拟存储: A 已完成
    mock_storage.get.return_value = {
        "step": 1,
        "last_node": "A",
        "context": {
            "input": "start",
            "history": {"A": "A", "last_output": "A"},
            "state": {},
            "executions": [] # 模拟被瘦身后的数据
        }
    }
    
    res = await simple_workflow.resume("sess_res")
    
    assert res == "B"
    node_a.assert_not_called()
    node_b.assert_called_once()

@pytest.mark.asyncio
async def test_condition_evaluation_error(simple_workflow):
    """
    测试条件评估错误
    [Fix] Patch 路径修正为 gecko.compose.workflow.engine
    """
    with patch("gecko.compose.workflow.engine.logger") as mock_logger:
        def bad_cond(ctx):
            raise ValueError("Eval Error")
            
        simple_workflow.add_node("A", lambda: 1)
        simple_workflow.add_node("B", lambda: 2)
        simple_workflow.set_entry_point("A")
        simple_workflow.add_edge("A", "B", bad_cond)
        
        # 此时条件报错 -> False -> 没路走了 -> 结束
        res = await simple_workflow.execute(None)
        assert res == 1
        
        mock_logger.error.assert_called()
        assert "Condition evaluation failed" in mock_logger.error.call_args[0][0]

@pytest.mark.asyncio
async def test_max_steps_exceeded(simple_workflow):
    """测试最大步数限制"""
    simple_workflow.max_steps = 2
    
    # [Fix] 使用关键字参数 node="..."
    simple_workflow.add_node("A", lambda: Next(node="B"))
    simple_workflow.add_node("B", lambda: Next(node="A"))
    simple_workflow.set_entry_point("A")
    
    with pytest.raises(WorkflowError, match="Exceeded max steps"):
        await simple_workflow.execute(None)

# ========================= 5. Context Methods =========================

def test_context_helper_methods():
    """测试 Context 的辅助方法 (Summary, Type Check)"""
    ctx = WorkflowContext(input="init")
    ctx.history["last_output"] = 123
    
    # get_last_output_as
    assert ctx.get_last_output_as(int) == 123
    assert ctx.get_last_output_as(str) == "123"
    
    with pytest.raises(TypeError):
        ctx.history["last_output"] = "abc"
        ctx.get_last_output_as(int)
        
    # get_summary
    s = ctx.get_summary()
    assert s["status"] == "completed"

# ========================= 6. 边缘情况与类型覆盖 (补全 100%) =========================

def test_result_normalization_types(simple_workflow):
    """
    [Coverage] 覆盖 _normalize_result 的所有类型分支
    """
    from pydantic import BaseModel
    from gecko.core.message import Message
    
    # 1. Pydantic Model
    class OutputModel(BaseModel):
        val: int
    res_model = simple_workflow._normalize_result(OutputModel(val=10))
    assert res_model == {"val": 10}
    
    # 2. Gecko Message
    res_msg = simple_workflow._normalize_result(Message.user("hello"))
    assert res_msg["content"] == "hello"
    assert res_msg["role"] == "user"
    
    # 3. 具备 model_dump 的任意对象
    class CustomObj:
        def model_dump(self):
            return {"custom": True}
    res_custom = simple_workflow._normalize_result(CustomObj())
    assert res_custom == {"custom": True}

def test_safe_preview_exception(simple_workflow):
    """
    [Coverage] 覆盖 _safe_preview 的异常捕获逻辑
    """
    class BadRepr:
        def __repr__(self):
            raise ValueError("I hate being printed")
        def __str__(self):
            raise ValueError("I hate being printed")
            
    # 访问私有方法进行测试
    preview = simple_workflow.executor._safe_preview(BadRepr())
    assert preview == "<Unprintable>"

@pytest.mark.asyncio
async def test_agent_node_dispatch(simple_workflow):
    """
    [Coverage] 覆盖 _run_intelligent_object (Agent/Team 调用)
    """
    class MockAgent:
        async def run(self, input_data):
            return f"Agent processed: {input_data}"
            
    simple_workflow.add_node("Agent", MockAgent())
    simple_workflow.set_entry_point("Agent")
    
    res = await simple_workflow.execute("raw_data")
    assert res == "Agent processed: raw_data"

@pytest.mark.asyncio
async def test_parallel_interface_stubs(simple_workflow):
    """
    [Coverage] 覆盖 Engine 中的并行接口存根代码
    """
    # 1. add_parallel_group
    simple_workflow.add_node("A", lambda: 1)
    simple_workflow.add_parallel_group("A") # 正常调用
    
    with pytest.raises(ValueError, match="not found"):
        simple_workflow.add_parallel_group("NonExistent")
        
    # 2. execute_parallel (目前透传给 execute)
    simple_workflow.set_entry_point("A")
    res = await simple_workflow.execute_parallel(None)
    assert res == 1

def test_explicit_dependency_setting(simple_workflow):
    """
    [Coverage] 覆盖 set_dependency 逻辑
    """
    simple_workflow.add_node("A", lambda: None)
    simple_workflow.add_node("B", lambda: None)
    
    # 单个依赖
    simple_workflow.set_dependency("B", "A")
    assert "A" in simple_workflow.graph.node_dependencies["B"]
    
    # 列表依赖
    simple_workflow.set_dependency("A", ["B"]) # 虽造成环，但 set_dependency 只管存
    assert "B" in simple_workflow.graph.node_dependencies["A"]