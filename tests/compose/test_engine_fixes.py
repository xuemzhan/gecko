# tests/compose/test_engine_fixes.py
import pytest
import anyio
from unittest.mock import MagicMock, patch

from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext
from gecko.compose.team import Team, MemberResult
from gecko.compose.nodes import Next

@pytest.mark.asyncio
async def test_workflow_deletes_state_integration():
    """
    [Integration] 验证 Workflow 能正确处理节点发出的删除指令
    """
    wf = Workflow("DelTest")
    
    def deleter_node(ctx: WorkflowContext):
        # User explicitly deletes a key
        if "token" in ctx.state:
            del ctx.state["token"]
        return "deleted"
        
    def checker_node(ctx: WorkflowContext):
        return "token" in ctx.state
        
    wf.add_node("del", deleter_node)
    wf.add_node("check", checker_node)
    wf.set_entry_point("del")
    wf.add_edge("del", "check")
    
    ctx = WorkflowContext(input={}, state={"token": "secret"})
    
    # 注入 context 运行
    res = await wf.execute(None, _resume_context=ctx)
    
    # checker node returns False (token not in state)
    assert res is False
    # Verify main context
    assert "token" not in ctx.state

@pytest.mark.asyncio
async def test_team_next_unwrapping_integration():
    """
    [Integration] [Fix P0-6] 验证 Team 成员返回 Next 时，Workflow 能正确跳转
    """
    wf = Workflow("TeamJump")
    
    # 1. Create a Team that returns a Next instruction
    async def router_agent(inp):
        return Next(node="Target", input="jumped")
        
    team = Team([router_agent], name="RouterTeam")
    
    # 2. Define nodes
    wf.add_node("Start", team)
    wf.add_node("OriginalNext", lambda: "should_not_run")
    wf.add_node("Target", lambda x: f"arrived: {x}")
    
    # 3. Graph: Start -> OriginalNext (Default path)
    wf.set_entry_point("Start")
    wf.add_edge("Start", "OriginalNext")
    # Disconnected node "Target", reachable only via jump
    
    res = await wf.execute("init")
    
    assert res == "arrived: jumped"
    # Verify trace to ensure OriginalNext was skipped
    node_names = [e.node_name for e in wf.persistence.storage or []]  # type: ignore
    # (Checking execution plan metadata if available, or just relying on output)

@pytest.mark.asyncio
async def test_executor_offloads_sync_function():
    """
    [Unit] [Fix P1-3] 验证同步函数被卸载到线程池
    """
    wf = Workflow("SyncTest")
    
    # Define a sync function
    def sync_heavy_task(ctx):
        return "done"
        
    wf.add_node("A", sync_heavy_task)
    wf.set_entry_point("A")
    
    # Mock anyio.to_thread.run_sync
    with patch("gecko.compose.workflow.executor.anyio.to_thread.run_sync", new_callable=MagicMock) as mock_run_sync:
        # Since run_sync is awaited, we need it to return an awaitable or result depending on implementation.
        # But wait, executor calls await run_sync(...). 
        # In AsyncMock, await returns the value.
        async def side_effect(func, *args):
            return func()
            
        # We need to mock it as an async function (or AsyncMock) 
        # because the executor does `await run_sync(...)`
        mock_run_sync.side_effect = side_effect
        
        await wf.execute(None)
        
        # Assert run_sync was called
        assert mock_run_sync.called

@pytest.mark.asyncio
async def test_persistence_offloads_serialization():
    """
    [Unit] [Fix P2-6] 验证持久化序列化被卸载到线程池
    """
    from gecko.plugins.storage.interfaces import SessionInterface
    mock_storage = MagicMock(spec=SessionInterface)
    mock_storage.set = MagicMock(side_effect=lambda k, v: None) # Mock async set as well if needed
    
    # Mock storage set to be awaitable
    async def async_set(*args): pass
    mock_storage.set = async_set

    wf = Workflow("PersistTest", storage=mock_storage)
    wf.add_node("A", lambda: "ok")
    wf.set_entry_point("A")
    
    # Patch persistence.run_sync
    with patch("gecko.compose.workflow.persistence.run_sync", new_callable=MagicMock) as mock_run_sync:
        async def side_effect(func, *args):
            return func()
        mock_run_sync.side_effect = side_effect
        
        await wf.execute(None, session_id="test_sess")
        
        # Persistence calls run_sync to clean/serialize context
        assert mock_run_sync.called