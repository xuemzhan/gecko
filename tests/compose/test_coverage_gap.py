"""
Coverage Gap Filling Tests for gecko.compose

Targeting missing lines in:
- gecko/compose/team.py
- gecko/compose/workflow/engine.py
- gecko/compose/workflow/executor.py
- gecko/compose/workflow/state.py
- gecko/compose/workflow/models.py
- gecko/compose/workflow/persistence.py
- gecko/compose/workflow/graph.py
"""
import pytest
import anyio
import time
from unittest.mock import MagicMock, AsyncMock, patch

from gecko.compose.team import Team, ExecutionStrategy, MemberResult
from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext, NodeExecution, CheckpointStrategy
from gecko.compose.workflow.executor import NodeExecutor, is_async_callable
from gecko.compose.workflow.state import COWDict
from gecko.compose.workflow.graph import WorkflowGraph
from gecko.compose.workflow.persistence import PersistenceManager
from gecko.compose.nodes import Next
from gecko.core.exceptions import WorkflowError
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface


# ========================= 1. Team Coverage =========================

@pytest.mark.asyncio
async def test_team_semaphore_logic():
    """
    Coverage: team.py lines 140, 146 (Semaphore acquire/release)
    """
    # max_concurrent=1 to force semaphore usage
    team = Team([lambda x: x], max_concurrent=1)
    res = await team.run("test")
    assert res[0].result == "test"

@pytest.mark.asyncio
async def test_team_execute_all_timeout():
    """
    Coverage: team.py lines 155-158 (Timeout handling in _execute_all)
    """
    async def slow_task(x):
        await anyio.sleep(0.5)
        return x
        
    team = Team([slow_task], strategy=ExecutionStrategy.ALL)
    # Timeout 0.1s -> Task won't finish -> results[i] remains None initially
    # Code should fill it with error
    with patch("gecko.compose.team.logger"):
        results = await team.run("test", timeout=0.1)
    
    assert results[0].is_success is False
    assert "Timed out" in results[0].error # type: ignore

@pytest.mark.asyncio
async def test_team_race_timeout_runtime_error():
    """
    Coverage: team.py line 214 (Timeout exception in _execute_race)
    """
    async def slow_task(x):
        await anyio.sleep(0.5)
        return x
        
    team = Team([slow_task], strategy=ExecutionStrategy.RACE)
    
    # In Race mode, timeout raises RuntimeError
    with pytest.raises(RuntimeError, match="timed out"):
        await team.run("test", timeout=0.1)

@pytest.mark.asyncio
async def test_team_race_cancellation_ignore():
    """
    Coverage: team.py line 196 (except CancelledError: pass)
    """
    # Hard to hit explicitly since anyio swallows cancellation in TaskGroup, 
    # but we can try to force a cancellation scenario.
    # Actually this block catches cancellation during the race.
    async def fast(x): return "fast"
    async def slow(x): await anyio.sleep(1); return "slow"
    
    team = Team([fast, slow], strategy=ExecutionStrategy.RACE)
    res = await team.run("test")
    assert len(res) == 1
    assert res[0].result == "fast"

@pytest.mark.asyncio
async def test_team_execute_member_type_error():
    """
    Coverage: team.py line 276 (TypeError for invalid member)
    """
    team = Team(["not_callable"]) # type: ignore
    with patch("gecko.compose.team.logger"):
        res = await team.run("test")
    assert res[0].is_success is False
    assert "not executable" in res[0].error # type: ignore

# ========================= 2. Workflow Engine Coverage =========================

def test_engine_properties():
    """
    Coverage: engine.py lines 107, 125 (Property setters)
    """
    wf = Workflow()
    # Storage setter
    mock_storage = MagicMock()
    wf.storage = mock_storage
    assert wf.persistence.storage == mock_storage
    
    # Allow cycles setter
    wf.allow_cycles = True
    assert wf._allow_cycles is True
    assert wf.graph._validated is False

@pytest.mark.asyncio
async def test_engine_resume_missing_data():
    """
    Coverage: engine.py line 551 (Session not found)
    """
    mock_storage = AsyncMock()
    mock_storage.get.return_value = None
    wf = Workflow(storage=mock_storage)
    
    with pytest.raises(ValueError, match="not found"):
        await wf.resume("missing_sess")

@pytest.mark.asyncio
async def test_engine_resume_dynamic_jump():
    """
    Coverage: engine.py lines 565, 569-570, 576-582 (Resume via Next Pointer)
    """
    mock_storage = AsyncMock()
    ctx_data = {
        "input": "in",
        "history": {},
        "state": {},
        "executions": [],
        "next_pointer": {"target_node": "Target", "input": "jumped_in"}
    }
    mock_storage.get.return_value = {
        "context": ctx_data,
        "last_node": "Prev"
    }
    
    wf = Workflow(storage=mock_storage)
    wf.add_node("Target", lambda x: f"Resumed: {x}")
    wf.set_entry_point("Target") # Just to make graph valid, resume ignores this
    
    res = await wf.resume("sess_jump")
    assert res == "Resumed: jumped_in"

@pytest.mark.asyncio
async def test_engine_execute_legacy_next_dict():
    """
    Coverage: engine.py line 486 (Legacy <Next string check)
    """
    wf = Workflow()
    # Node returning a dict that looks like a legacy Next
    def legacy_node(ctx):
        return {"node": "B", "input": "legacy_val", "marker": "<Next"}
        
    wf.add_node("A", legacy_node)
    wf.set_entry_point("A")
    
    # Capture history to check extraction
    ctx = WorkflowContext(input={})
    await wf.execute(None, _resume_context=ctx)
    
    # Should extract "input" field
    assert ctx.history["A"] == "legacy_val"

@pytest.mark.asyncio
async def test_engine_execute_skipped_status():
    """
    Coverage: engine.py line 413, 415-416 (SKIPPED status)
    """
    wf = Workflow()
    wf.add_node("A", lambda: 1)
    wf.add_node("B", lambda: 2)
    wf.set_entry_point("A")
    
    # Conditional edge that returns False
    wf.add_edge("A", "B", condition=lambda ctx: False)
    
    ctx = WorkflowContext(input={})
    await wf.execute(None, _resume_context=ctx)
    
    # B should be skipped
    # Currently engine._merge_layer_results skips updating history for SKIPPED nodes
    assert "B" not in ctx.history

@pytest.mark.asyncio
async def test_engine_session_persistence_hooks():
    """
    Coverage: engine.py lines 316, 242 (Session ID handling)
    """
    mock_storage = AsyncMock()
    wf = Workflow(storage=mock_storage, checkpoint_strategy="always")
    wf.add_node("A", lambda: "ok")
    wf.set_entry_point("A")
    
    await wf.execute("in", session_id="sess_hooks")
    
    # Verify save_checkpoint called
    assert mock_storage.set.call_count >= 2

# ========================= 3. Executor Coverage =========================

def test_is_async_callable_coverage():
    """
    Coverage: executor.py line 42 (Async object with __call__)
    """
    class AsyncObj:
        async def __call__(self):
            return 1
            
    assert is_async_callable(AsyncObj()) is True
    assert is_async_callable(lambda: 1) is False

@pytest.mark.asyncio
async def test_executor_normalize_pydantic_v1():
    """
    Coverage: executor.py line 131 (Pydantic v1 .dict() support)
    """
    class V1Model:
        def dict(self):
            return {"v1": True}
            
    exe = NodeExecutor()
    res = exe._normalize_result(V1Model())
    assert res == {"v1": True}

@pytest.mark.asyncio
async def test_executor_normalize_message():
    """
    Coverage: executor.py line 133 (Message support)
    """
    msg = Message.user("hello")
    exe = NodeExecutor()
    res = exe._normalize_result(msg)
    assert res["role"] == "user"

@pytest.mark.asyncio
async def test_executor_retry_exhaustion():
    """
    Coverage: executor.py line 163 (Raise last error)
    """
    exe = NodeExecutor(enable_retry=True, max_retries=1)
    
    count = 0
    def failing():
        nonlocal count
        count += 1
        raise ValueError("Fail")
    
    # [Fix] Expect WorkflowError instead of bare ValueError
    with pytest.raises(WorkflowError) as exc_info:
        await exe.execute_node("A", failing, WorkflowContext(input={}))
    
    # [Fix] Verify the cause is the original ValueError
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "Fail"
    
    # Initial + 1 retry = 2 calls
    assert count == 2

@pytest.mark.asyncio
async def test_executor_smart_binding_coverage():
    """
    Coverage: executor.py lines 240, 250-251 (Context type hint & input fallback)
    """
    exe = NodeExecutor()
    
    # 1. Type Hint Injection
    def type_hint_func(c: WorkflowContext):
        return c.input
        
    ctx = WorkflowContext(input="hinted")
    res = await exe.execute_node("A", type_hint_func, ctx)
    assert res == "hinted"
    
    # 2. Input fallback (explicit test for append)
    def input_func(x):
        return x
        
    ctx2 = WorkflowContext(input="val")
    # Manually ensure no _next_input to force last_output usage
    res2 = await exe.execute_node("B", input_func, ctx2)
    assert res2 == "val"

# ========================= 4. State & Models Coverage =========================

def test_state_del_local_and_missing():
    """
    Coverage: state.py lines 82, 86-88 (delitem branches)
    """
    # 1. Del key only in local
    base = {"a": 1}
    cow = COWDict(base)
    cow["c"] = 3
    del cow["c"] # Hits line 82
    assert "c" not in cow
    
    # 2. Del missing key
    with pytest.raises(KeyError):
        del cow["missing"] # Hits line 88

def test_state_pop_deleted():
    """
    Coverage: state.py line 95 (pop deleted)
    """
    cow = COWDict({"a": 1})
    del cow["a"]
    assert cow.pop("a", "default") == "default"

def test_state_update_branch():
    """
    Coverage: state.py line 109, 112 (update with other)
    """
    cow = COWDict()
    cow.update({"a": 1})
    assert cow["a"] == 1
    
    cow.update(b=2)
    assert cow["b"] == 2

def test_model_coverage():
    """
    Coverage: models.py lines 51-53, 93, 121, 131, 156-159
    """
    # 51-53: Duration zero
    ne = NodeExecution(node_name="A", start_time=100, end_time=0)
    assert ne.duration == 0.0
    
    # 93: get_last_output default
    ctx = WorkflowContext(input="init")
    assert ctx.get_last_output() == "init"
    
    # 121, 131: to_storage_payload
    ctx.history["a"] = 1
    ctx.history["last_output"] = 99
    # max_history=1, should keep 'a' (if latest) or 'last_output'
    # 'a' is key, 'last_output' is key.
    # Logic: keep last N keys.
    # Let's add many keys
    ctx.history = {f"k{i}": i for i in range(10)}
    ctx.history["last_output"] = 999
    
    payload = ctx.to_storage_payload(max_history_steps=2)
    # last_output is usually preserved if logic says so
    assert "last_output" in payload["history"]
    assert len(payload["history"]) <= 3 # 2 + last_output potentially
    
    # 156-159: get_last_output_as type fail
    ctx.history["last_output"] = "abc"
    with pytest.raises(TypeError):
        ctx.get_last_output_as(int)

# ========================= 5. Persistence & Graph Coverage =========================

@pytest.mark.asyncio
async def test_persistence_manual_strategy():
    """
    Coverage: persistence.py line 57 (Manual strategy return)
    """
    pm = PersistenceManager(None, strategy=CheckpointStrategy.MANUAL)
    # Should return None/Void immediately, no error even if storage is None
    await pm.save_checkpoint("id", 1, "node", WorkflowContext(input=""))

@pytest.mark.asyncio
async def test_persistence_load_no_storage():
    """
    Coverage: persistence.py line 93
    """
    pm = PersistenceManager(None)
    assert await pm.load_checkpoint("id") is None

def test_graph_validate_cache():
    """
    Coverage: graph.py line 74
    """
    g = WorkflowGraph()
    g.add_node("A", lambda: None)
    g.set_entry_point("A")
    g.validate()
    assert g._validated is True
    # Call again to hit cache return
    g.validate()

def test_graph_ambiguity_check():
    """
    Coverage: graph.py line 96
    """
    g = WorkflowGraph()
    g.add_node("A", lambda: None)
    g.add_node("B", lambda: None)
    g.add_node("C", lambda: None)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.set_entry_point("A")
    
    # enable_parallel=False should detect ambiguity
    valid, errors = g.validate(enable_parallel=False)
    assert valid is False
    assert "ambiguous branching" in errors[0]

def test_graph_build_layers_invalid_start():
    """
    Coverage: graph.py line 129
    """
    g = WorkflowGraph()
    assert g.build_execution_layers("Missing") == []