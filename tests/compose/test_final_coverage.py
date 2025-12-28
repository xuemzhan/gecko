"""
Final coverage gap filling for gecko.compose
Targeting specific missing lines identified in the coverage report.
"""
import pytest
import anyio
from unittest.mock import MagicMock, AsyncMock, patch

from gecko.compose.team import Team, ExecutionStrategy
from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext, NodeExecution, CheckpointStrategy
from gecko.compose.workflow.executor import NodeExecutor
from gecko.compose.workflow.state import COWDict
from gecko.compose.workflow.graph import WorkflowGraph
from gecko.compose.workflow.persistence import PersistenceManager
from gecko.compose.nodes import Next
from gecko.core.exceptions import WorkflowCycleError, WorkflowError
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface


# ========================= Team Coverage =========================

@pytest.mark.asyncio
async def test_team_resolve_input_raw():
    """Coverage: team.py line 257 (Fallback to raw input)"""
    team = Team([])
    # Pass a string directly, not a Context object
    # Should return the string itself
    assert team._resolve_input("raw_string") == "raw_string"

@pytest.mark.asyncio
async def test_team_execute_member_type_error():
    """Coverage: team.py line 276 (TypeError for non-executable)"""
    team = Team(["not_callable"]) # type: ignore
    with patch("gecko.compose.team.logger"):
        res = await team.run("in")
    assert res[0].is_success is False
    assert "not executable" in res[0].error # type: ignore

@pytest.mark.asyncio
async def test_team_race_generic_crash():
    """Coverage: team.py lines 202-205 (Generic exception in race)"""
    team = Team([], strategy=ExecutionStrategy.RACE)
    
    # Mock create_task_group to raise generic Exception
    with patch("anyio.create_task_group", side_effect=Exception("System Crash")):
        with patch("gecko.compose.team.logger") as mock_logger:
            # Should not raise, but log exception and return failure list
            res = await team.run("in")
            
            mock_logger.exception.assert_called_with(
                "Race execution crashed", 
                team=team.name, 
                error="System Crash"
            )
            # Should return empty list or failed list (members is empty so empty list)
            assert isinstance(res, list)

# ========================= Engine Coverage =========================

def test_engine_misc_methods():
    """Coverage: engine.py lines 107, 153, 183"""
    wf = Workflow()
    
    # 107: allow_cycles setter
    wf.allow_cycles = True
    assert wf._allow_cycles is True
    
    # 153: print_structure
    # Just ensure it doesn't crash
    wf.print_structure() 
    
    # 183: add_parallel_group (Deprecated)
    wf.add_node("A", lambda: None)
    wf.add_parallel_group("A")

@pytest.mark.asyncio
async def test_engine_resume_edges():
    """Coverage: engine.py lines 551, 569-570, 607-608"""
    mock_storage = AsyncMock()
    wf = Workflow(storage=mock_storage)
    
    # 551: Session not found
    mock_storage.get.return_value = None
    with pytest.raises(ValueError, match="not found"):
        await wf.resume("missing")

    # Setup for Next Pointer Resume
    mock_storage.get.return_value = {
        "context": {
            "input": "in", 
            "history": {}, 
            "state": {}, 
            "executions": [],
            "next_pointer": {"target_node": "Target", "input": "jumped_data"}
        },
        "last_node": None
    }
    wf.add_node("Target", lambda x: x)
    wf.set_entry_point("Target")
    
    # 569-570: Resume with Next Pointer Input
    res = await wf.resume("sess")
    assert res == "jumped_data"
    
    # 607-608: Resume via Static Flow (Next Node)
    # [Fix] Populate history to simulate upstream 'A' has finished
    # Otherwise, 'B' will be skipped because its dependency 'A' is missing from history
    mock_storage.get.return_value = {
        "context": {
            "input": "in", 
            "history": {"A": "A_result"}, # <--- 关键修复：添加上游历史
            "state": {}, 
            "executions": []
        },
        "last_node": "A"
    }
    
    # Reset graph for static flow test
    wf = Workflow(storage=mock_storage) # Re-init to clear previous graph
    wf.add_node("A", lambda: "A")
    wf.add_node("B", lambda: "B")
    wf.add_edge("A", "B")
    wf.set_entry_point("A")
    
    res = await wf.resume("sess_static")
    assert res == "B"

@pytest.mark.asyncio
async def test_engine_execute_session_hooks():
    """Coverage: engine.py lines 242, 316"""
    mock_storage = AsyncMock()
    wf = Workflow(storage=mock_storage, checkpoint_strategy="always")
    wf.add_node("A", lambda: "ok")
    wf.set_entry_point("A")
    
    # Execute with session_id to trigger persistence hooks
    await wf.execute("in", session_id="sess_id")
    
    assert mock_storage.set.called

@pytest.mark.asyncio
async def test_engine_skipped_node_logic():
    """Coverage: engine.py lines 413, 415-416"""
    wf = Workflow()
    wf.add_node("A", lambda: 1)
    wf.add_node("B", lambda: 2)
    wf.set_entry_point("A")
    # Condition returns False -> Skip
    wf.add_edge("A", "B", condition=lambda c: False)
    
    ctx = WorkflowContext(input={})
    await wf.execute(None, _resume_context=ctx)
    
    # B should be skipped and not in history
    assert "B" not in ctx.history

# ========================= Executor Coverage =========================

def test_executor_normalize_message():
    """Coverage: executor.py line 133"""
    exe = NodeExecutor()
    msg = Message.user("hi")
    res = exe._normalize_result(msg)
    assert res["role"] == "user"

@pytest.mark.asyncio
async def test_executor_intelligent_obj_fallback():
    """Coverage: executor.py line 191"""
    exe = NodeExecutor()
    
    class Agent:
        async def run(self, x): return x
        
    ctx = WorkflowContext(input="init")
    # No _next_input, should fallback to get_last_output (which is input)
    res = await exe._run_intelligent_object(Agent(), ctx)
    assert res == "init"

@pytest.mark.asyncio
async def test_executor_prepare_args_branches():
    """Coverage: executor.py lines 240, 250-251"""
    exe = NodeExecutor()
    
    # 240: Argument named 'context' without type hint
    def func_named_ctx(context):
        return context.input
        
    ctx = WorkflowContext(input="val")
    res = await exe.execute_node("A", func_named_ctx, ctx)
    assert res == "val"
    
    # 250-251: Extra arguments logic
    def func_args(a, b):
        return a + b
        
    # 'a' gets context input, 'b' missing? No, logic is:
    # remaining params get input. 
    # If we have 1 input and 2 args, 2nd arg missing -> TypeError.
    # We want to test the append line.
    def func_one_arg(x):
        return x
    
    res2 = await exe.execute_node("B", func_one_arg, ctx)
    assert res2 == "val"

# ========================= Graph Coverage =========================

def test_graph_cycle_error():
    """Coverage: graph.py lines 235-237"""
    g = WorkflowGraph()
    g.add_node("A", lambda: None)
    g.add_node("B", lambda: None)
    g.add_edge("A", "B")
    g.add_edge("B", "A")
    g.set_entry_point("A")
    
    # [Fix] validate() catches the error, so we must call _detect_cycles() directly
    # to verify it raises WorkflowCycleError and hit the coverage line.
    with pytest.raises(WorkflowCycleError):
        g._detect_cycles()

# ========================= State Coverage =========================

def test_state_branches():
    """Coverage: state.py lines 71-72, 105, 116, 123-124"""
    cow = COWDict({"a": 1})
    del cow["a"]
    
    # 71-72: __contains__ deleted
    assert "a" not in cow
    
    # 105: pop deleted
    assert cow.pop("a", "default") == "default"
    
    # 116: update with invalid other (non-iterable) -> Should catch and pass
    # NOTE: update implementation:
    # if other: try: for k,v in dict(other).items()... except: pass
    # dict(1) raises TypeError.
    cow.update(1) 
    
    # 123-124: update with kwargs
    cow.update(b=2)
    assert cow["b"] == 2

# ========================= Models Coverage =========================

def test_models_branches():
    """Coverage: models.py lines 53, 93, 121, 131, 156-159"""
    # 53: Duration 0
    ne = NodeExecution(node_name="A")
    assert ne.duration == 0.0
    
    # 93: get_last_output default
    ctx = WorkflowContext(input="init")
    assert ctx.get_last_output() == "init"
    
    # 121, 131: to_storage_payload logic
    ctx.history = {"A": 1, "last_output": 2}
    payload = ctx.to_storage_payload(max_history_steps=1)
    # last_output should be preserved explicitly if it wasn't in the slice
    assert payload["history"]["last_output"] == 2
    
    # 156-159: Type coercion fail
    ctx.history["last_output"] = "abc"
    with pytest.raises(TypeError):
        ctx.get_last_output_as(int)

# ========================= Persistence Coverage =========================

@pytest.mark.asyncio
async def test_persistence_manual():
    """Coverage: persistence.py line 57"""
    pm = PersistenceManager(None, strategy=CheckpointStrategy.MANUAL)
    # Should return immediately (coverage line hit)
    await pm.save_checkpoint("sess", 1, "node", WorkflowContext(input=""))