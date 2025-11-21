# tests/compose/test_team.py
"""
Team 模块单元测试

覆盖率目标：100%
测试范围：
1. Team 初始化与基本执行
2. 成员类型支持 (Agent vs Callable)
3. 并发控制 (Semaphore)
4. 容错机制 (Partial Failure)
5. 智能输入解析 (WorkflowContext 集成)
6. 结果标准化策略
"""
import pytest
import asyncio
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

from pydantic import BaseModel

from gecko.compose.team import Team
from gecko.core.agent import Agent
from gecko.core.message import Message
from gecko.core.output import AgentOutput

# ========================= Mock 对象 =========================

class MockContext:
    """模拟 WorkflowContext (Duck Typing)"""
    def __init__(self, input_data: Any, history: dict = None, state: dict = None):
        self.input = input_data
        self.history = history or {}
        self.state = state or {}

class MockAgent:
    """模拟 Agent"""
    def __init__(self, return_value="agent_result"):
        self.return_value = return_value
        # 模拟 run 方法
        self.run = AsyncMock(return_value=AgentOutput(content=return_value))

# ========================= 1. 初始化与基础执行测试 =========================

@pytest.mark.asyncio
async def test_team_init_and_call():
    """测试 Team 初始化及 __call__ 协议"""
    async def simple_task(x):
        return f"processed {x}"
    
    team = Team(members=[simple_task], name="TestTeam")
    
    assert team.name == "TestTeam"
    assert len(team.members) == 1
    assert team.max_concurrent == 0
    
    # 测试 __call__
    results = await team("data")
    assert results == ["processed data"]

@pytest.mark.asyncio
async def test_mixed_members_execution():
    """测试混合成员类型 (Agent + Callable)"""
    agent = MockAgent(return_value="agent_done")
    
    async def func_task(x):
        return f"func_{x}"
    
    # 同步函数也应该被支持（通过 ensure_awaitable）
    def sync_task(x):
        return f"sync_{x}"
    
    team = Team(members=[agent, func_task, sync_task])
    results = await team.run("input")
    
    # 验证 Agent.run 被调用
    agent.run.assert_awaited_once_with("input")
    
    # 验证结果顺序和内容
    assert results == ["agent_done", "func_input", "sync_input"]

@pytest.mark.asyncio
async def test_invalid_member_type():
    """测试无效的成员类型"""
    team = Team(members=["not_callable_string"]) # type: ignore
    
    # _execute_member 会抛出 TypeError，并在 worker 中被捕获
    results = await team.run("input")
    
    assert len(results) == 1
    assert "Error: Member not_callable_string is not executable" in results[0]

# ========================= 2. 并发控制与容错测试 =========================

@pytest.mark.asyncio
async def test_concurrency_semaphore():
    """测试并发限制 (Semaphore)"""
    # 使用 patch 验证 anyio.Semaphore 被创建
    with patch("anyio.Semaphore") as mock_sem_cls:
        mock_sem = MagicMock()
        
        # [Fix] acquire 必须是 AsyncMock，因为它被 await 调用
        mock_sem.acquire = AsyncMock()
        # [Fix] release 是同步调用，保持 MagicMock 即可
        mock_sem.release = MagicMock()
        
        mock_sem_cls.return_value = mock_sem
        
        async def task(x): return x
        
        team = Team(members=[task, task], max_concurrent=1)
        await team.run("test")
        
        # 验证 Semaphore 被初始化且值为 1
        mock_sem_cls.assert_called_once_with(1)
        # 验证 acquire 被 await 调用过 2 次
        assert mock_sem.acquire.await_count == 2
        # 验证 release 被调用过 2 次
        assert mock_sem.release.call_count == 2

@pytest.mark.asyncio
async def test_partial_failure():
    """测试部分失败容错机制"""
    async def success_task(x):
        return "success"
    
    async def failing_task(x):
        raise ValueError("Boom!")
    
    team = Team(members=[success_task, failing_task])
    
    # 使用 patch logger 验证错误日志
    with patch("gecko.compose.team.logger") as mock_logger:
        results = await team.run("input")
        
        # 验证结果：应该包含一个成功，一个失败信息
        assert results[0] == "success"
        assert "Error: Boom!" in results[1]
        
        # 验证日志记录了错误
        mock_logger.error.assert_called()
        args, kwargs = mock_logger.error.call_args
        assert "Team member execution failed" in args[0]

# ========================= 3. 输入解析逻辑测试 =========================

@pytest.mark.asyncio
async def test_resolve_input_raw():
    """测试普通输入解析"""
    team = Team([])
    assert team._resolve_input("raw_string") == "raw_string"
    assert team._resolve_input(123) == 123

@pytest.mark.asyncio
async def test_resolve_input_context_priority():
    """测试 WorkflowContext 优先级解析"""
    team = Team([])
    
    # 1. 只有 Input
    ctx1 = MockContext(input_data="global_input")
    assert team._resolve_input(ctx1) == "global_input"
    
    # 2. 有 Last Output (覆盖 Input)
    ctx2 = MockContext(input_data="global", history={"last_output": "prev_step"})
    assert team._resolve_input(ctx2) == "prev_step"
    
    # 3. 有 Next Input (最高优先级)
    ctx3 = MockContext(
        input_data="global", 
        history={"last_output": "prev"}, 
        state={"_next_input": "jump_data"}
    )
    val = team._resolve_input(ctx3)
    assert val == "jump_data"
    # 注意：_resolve_input 会 pop 掉 _next_input
    assert "_next_input" not in ctx3.state

@pytest.mark.asyncio
async def test_data_handover_cleaning():
    """测试 Data Handover 清洗 (dict -> content)"""
    team = Team([])
    
    # 模拟上一个节点返回的是序列化后的 AgentOutput
    complex_output = {
        "content": "clean_text",
        "tool_calls": [],
        "usage": {"total_tokens": 100}
        # 注意没有 'role' 字段，否则会被识别为 Message
    }
    
    ctx = MockContext(input_data="x", history={"last_output": complex_output})
    
    # 应该只提取 content
    assert team._resolve_input(ctx) == "clean_text"
    
    # 如果是 Message 对象 (含 role)，则不应清洗，应该保持原样让 Agent 处理
    message_dict = {
        "role": "user",
        "content": "hello"
    }
    ctx_msg = MockContext(input_data="x", history={"last_output": message_dict})
    assert team._resolve_input(ctx_msg) == message_dict

# ========================= 4. 结果标准化测试 =========================

def test_process_result_default():
    """测试默认结果提取 (return_full_output=False)"""
    team = Team([], return_full_output=False)
    
    # AgentOutput -> content
    ao = AgentOutput(content="ao_content")
    assert team._process_result(ao) == "ao_content"
    
    # Message -> content
    msg = Message.user("msg_content")
    assert team._process_result(msg) == "msg_content"
    
    # Dict with content -> content
    d = {"content": "dict_content", "other": 1}
    assert team._process_result(d) == "dict_content"
    
    # Other -> as is
    assert team._process_result(123) == 123

def test_process_result_full():
    """测试完整结果返回 (return_full_output=True)"""
    team = Team([], return_full_output=True)
    
    # AgentOutput -> dict
    ao = AgentOutput(content="ao")
    res = team._process_result(ao)
    assert isinstance(res, dict)
    assert res["content"] == "ao"
    
    # BaseModel -> dict
    class MyModel(BaseModel):
        val: int
    m = MyModel(val=99)
    res_m = team._process_result(m)
    assert isinstance(res_m, dict)
    assert res_m["val"] == 99
    
    # Other -> as is
    assert team._process_result("str") == "str"

def test_repr():
    """测试字符串表示"""
    team = Team([], name="MyTeam", max_concurrent=5)
    assert "MyTeam" in repr(team)
    assert "concurrency=5" in repr(team)