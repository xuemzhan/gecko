# tests/unit/test_workflow.py
"""Workflow 模块测试"""
import pytest
from gecko.compose.workflow import Workflow, WorkflowContext, Next
from gecko.core.exceptions import WorkflowError, WorkflowCycleError

# ========== 基础功能测试 ==========

def test_workflow_creation():
    """测试 Workflow 创建"""
    wf = Workflow(name="test_workflow")
    
    assert wf.name == "test_workflow"
    assert len(wf.nodes) == 0
    assert wf.entry_point is None

def test_add_node():
    """测试添加节点"""
    wf = Workflow()
    
    @wf.add_node("node_a", lambda ctx: "result_a")
    def node_a(ctx):
        return "result_a"
    
    assert "node_a" in wf.nodes
    assert wf.nodes["node_a"] is node_a

def test_add_edge():
    """测试添加边"""
    wf = Workflow()
    
    wf.add_node("a", lambda ctx: "a")
    wf.add_node("b", lambda ctx: "b")
    wf.add_edge("a", "b")
    
    assert "a" in wf.edges
    assert len(wf.edges["a"]) == 1
    assert wf.edges["a"][0][0] == "b"

def test_set_entry_point():
    """测试设置入口点"""
    wf = Workflow()
    wf.add_node("start", lambda ctx: "start")
    wf.set_entry_point("start")
    
    assert wf.entry_point == "start"

# ========== DAG 验证测试 ==========

def test_validate_no_entry_point():
    """测试验证：无入口点"""
    wf = Workflow()
    wf.add_node("a", lambda ctx: "a")
    
    is_valid = wf.validate()
    
    assert is_valid is False
    errors = wf.get_validation_errors()
    assert any("entry point" in e.lower() for e in errors)

def test_validate_cycle_detection():
    """测试环检测"""
    wf = Workflow()
    
    wf.add_node("a", lambda ctx: "a")
    wf.add_node("b", lambda ctx: "b")
    wf.add_node("c", lambda ctx: "c")
    
    wf.add_edge("a", "b")
    wf.add_edge("b", "c")
    wf.add_edge("c", "a")  # 形成环
    
    wf.set_entry_point("a")
    
    is_valid = wf.validate()
    
    assert is_valid is False
    errors = wf.get_validation_errors()
    assert any("cycle" in e.lower() for e in errors)

def test_validate_success():
    """测试验证成功"""
    wf = Workflow()
    
    wf.add_node("start", lambda ctx: "start")
    wf.add_node("process", lambda ctx: "process")
    wf.add_node("end", lambda ctx: "end")
    
    wf.add_edge("start", "process")
    wf.add_edge("process", "end")
    
    wf.set_entry_point("start")
    
    is_valid = wf.validate()
    
    assert is_valid is True
    assert len(wf.get_validation_errors()) == 0

def test_unreachable_nodes_warning(caplog):
    """测试孤立节点警告"""
    import logging
    caplog.set_level(logging.WARNING)
    
    wf = Workflow()
    
    wf.add_node("start", lambda ctx: "start")
    wf.add_node("orphan", lambda ctx: "orphan")  # 孤立节点
    
    wf.set_entry_point("start")
    
    wf.validate()
    
    # 应该有警告日志
    assert any("unreachable" in record.message.lower() for record in caplog.records)

# ========== 执行测试 ==========

@pytest.mark.asyncio
async def test_simple_execution():
    """测试简单执行"""
    wf = Workflow()
    
    wf.add_node("step1", lambda ctx: "result1")
    wf.add_node("step2", lambda ctx: f"result2_{ctx.history['step1']}")
    
    wf.add_edge("step1", "step2")
    wf.set_entry_point("step1")
    
    result = await wf.execute("input")
    
    assert result == "result2_result1"

@pytest.mark.asyncio
async def test_execution_with_context():
    """测试上下文传递"""
    wf = Workflow()
    
    def node_a(ctx: WorkflowContext):
        ctx.state["counter"] = 1
        return "a"
    
    def node_b(ctx: WorkflowContext):
        ctx.state["counter"] += 1
        return f"b_{ctx.state['counter']}"
    
    wf.add_node("a", node_a)
    wf.add_node("b", node_b)
    wf.add_edge("a", "b")
    wf.set_entry_point("a")
    
    result = await wf.execute("test")
    
    assert result == "b_2"

@pytest.mark.asyncio
async def test_conditional_routing():
    """测试条件路由"""
    wf = Workflow()
    
    wf.add_node("start", lambda ctx: 10)
    wf.add_node("large", lambda ctx: "large")
    wf.add_node("small", lambda ctx: "small")
    
    # 条件：如果结果 > 5 则去 large
    wf.add_edge("start", "large", condition=lambda ctx: ctx.history["start"] > 5)
    wf.add_edge("start", "small", condition=lambda ctx: ctx.history["start"] <= 5)
    
    wf.set_entry_point("start")
    
    result = await wf.execute("test")
    
    assert result == "large"

@pytest.mark.asyncio
async def test_next_instruction():
    """测试 Next 指令"""
    wf = Workflow()
    
    def decision_node(ctx: WorkflowContext):
        # 动态决定跳转
        if ctx.input == "skip":
            return Next(node="end", input="skipped")
        return "normal"
    
    wf.add_node("start", decision_node)
    wf.add_node("middle", lambda ctx: "middle")
    wf.add_node("end", lambda ctx: f"end_{ctx.history['last_output']}")
    
    wf.add_edge("start", "middle")
    wf.add_edge("middle", "end")
    
    wf.set_entry_point("start")
    
    # 测试跳过
    result = await wf.execute("skip")
    assert result == "end_skipped"

@pytest.mark.asyncio
async def test_max_steps_protection():
    """测试最大步数保护"""
    wf = Workflow(max_steps=3)
    
    # 创建一个会无限循环的工作流（但没有环）
    wf.add_node("a", lambda ctx: Next(node="a"))  # 自己跳自己
    wf.set_entry_point("a")
    
    # 添加一条边让验证通过
    wf.add_node("b", lambda ctx: "b")
    wf.add_edge("a", "b")
    
    with pytest.raises(WorkflowError, match="exceeded max steps"):
        await wf.execute("test")

# ========== 重试机制测试 ==========

@pytest.mark.asyncio
async def test_retry_on_failure():
    """测试失败重试"""
    wf = Workflow(enable_retry=True, max_retries=3)
    
    call_count = 0
    
    def flaky_node(ctx):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    wf.add_node("flaky", flaky_node)
    wf.set_entry_point("flaky")
    
    result = await wf.execute("test")
    
    assert result == "success"
    assert call_count == 3

# ========== 执行追踪测试 ==========

@pytest.mark.asyncio
async def test_execution_tracking():
    """测试执行追踪"""
    wf = Workflow()
    
    wf.add_node("a", lambda ctx: "a")
    wf.add_node("b", lambda ctx: "b")
    wf.add_edge("a", "b")
    wf.set_entry_point("a")
    
    # 通过捕获上下文来验证
    captured_context = None
    
    async def capture_wrapper(input_data):
        nonlocal captured_context
        ctx = WorkflowContext(input=input_data)
        result = await wf._execute_loop(ctx)
        captured_context = ctx
        return result
    
    # 手动调用（绕过 execute 的验证）
    wf._validated = True
    await capture_wrapper("test")
    
    # 验证执行记录
    assert captured_context is not None
    assert len(captured_context.executions) == 2
    assert captured_context.executions[0].node_name == "a"
    assert captured_context.executions[1].node_name == "b"
    
    # 验证摘要
    summary = captured_context.get_execution_summary()
    assert summary["total_nodes"] == 2
    assert summary["status_counts"]["success"] == 2

# ========== 可视化测试 ==========

def test_to_mermaid():
    """测试 Mermaid 图生成"""
    wf = Workflow()
    
    wf.add_node("start", lambda ctx: "start")
    wf.add_node("process", lambda ctx: "process")
    wf.add_node("end", lambda ctx: "end")
    
    wf.add_edge("start", "process")
    wf.add_edge("process", "end")
    wf.set_entry_point("start")
    
    mermaid = wf.to_mermaid()
    
    assert "graph TD" in mermaid
    assert "start" in mermaid
    assert "process" in mermaid
    assert "end" in mermaid
    assert "-->" in mermaid

def test_print_structure(capsys):
    """测试结构打印"""
    wf = Workflow(name="TestFlow")
    
    wf.add_node("a", lambda ctx: "a")
    wf.add_node("b", lambda ctx: "b")
    wf.add_edge("a", "b")
    wf.set_entry_point("a")
    
    wf.print_structure()
    
    captured = capsys.readouterr()
    assert "TestFlow" in captured.out
    assert "Entry Point: a" in captured.out