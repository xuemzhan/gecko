# tests/compose/test_workflow.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.core.exceptions import WorkflowError, WorkflowCycleError

@pytest.mark.asyncio
async def test_workflow_async_condition():
    """测试 Workflow 对 async 条件函数的支持 (修复验证)"""
    wf = Workflow()

    wf.add_node("start", lambda ctx: "start_done")
    wf.add_node("async_path", lambda ctx: "async_ok")
    wf.add_node("sync_path", lambda ctx: "sync_ok")

    # 定义异步条件
    async def async_cond(ctx):
        await asyncio.sleep(0.01)
        return True

    wf.add_edge("start", "async_path", condition=async_cond)
    wf.add_edge("start", "sync_path", condition=lambda ctx: False) # 不走这条

    wf.set_entry_point("start")

    result = await wf.execute("input")
    assert result == "async_ok"
    
    # [修复] 这里的断言修改为直接检查节点名称字符串
    # 错误代码: assert wf.nodes["async_path"] in str(wf.to_mermaid())
    # 修正后:
    assert "async_path" in wf.to_mermaid()

@pytest.mark.asyncio
async def test_workflow_persistence():
    """测试 Workflow 状态持久化"""
    mock_storage = AsyncMock()
    wf = Workflow(storage=mock_storage)
    
    wf.add_node("A", lambda ctx: "A_out")
    wf.add_node("B", lambda ctx: "B_out")
    wf.add_edge("A", "B")
    wf.set_entry_point("A")
    
    await wf.execute("input", session_id="sess_123")
    
    # 验证 storage.set 被调用
    assert mock_storage.set.called
    # 应该调用了2次 (节点A后, 节点B后)
    assert mock_storage.set.call_count >= 2 
    
    call_args = mock_storage.set.call_args_list[0]
    key = call_args[0][0]
    val = call_args[0][1]
    
    assert key == "workflow:sess_123"
    assert val["last_node"] == "A"
    assert "context" in val

@pytest.mark.asyncio
async def test_dag_validation():
    wf = Workflow()
    wf.add_node("A", lambda x: x)
    wf.add_node("B", lambda x: x)
    
    # 制造环
    wf.add_edge("A", "B")
    wf.add_edge("B", "A")
    wf.set_entry_point("A")
    
    assert wf.validate() is False
    errors = wf.get_validation_errors()
    assert any("Cycle detected" in e for e in errors)
    
    with pytest.raises(WorkflowError):
        await wf.execute("input")