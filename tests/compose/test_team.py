# tests/compose/test_team.py (Updated for Phase 1)
import pytest
import asyncio
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

from pydantic import BaseModel

from gecko.compose.team import Team, MemberResult
from gecko.core.agent import Agent
from gecko.core.message import Message
from gecko.core.output import AgentOutput

# ========================= Mock 对象 =========================

class MockContext:
    def __init__(self, input_data: Any, history: dict = None, state: dict = None):
        self.input = input_data
        self.history = history or {}
        self.state = state or {}

class MockAgent:
    def __init__(self, return_value="agent_result"):
        self.return_value = return_value
        self.run = AsyncMock(return_value=AgentOutput(content=return_value))

# ========================= 测试用例 =========================

@pytest.mark.asyncio
async def test_team_init_and_call():
    """测试 Team 初始化及 __call__ 协议"""
    async def simple_task(x):
        return f"processed {x}"
    
    team = Team(members=[simple_task], name="TestTeam")
    
    # 测试 __call__ 返回 MemberResult 列表
    results = await team("data")
    assert len(results) == 1
    assert isinstance(results[0], MemberResult)
    assert results[0].result == "processed data"
    assert results[0].is_success is True

@pytest.mark.asyncio
async def test_mixed_members_execution():
    """测试混合成员类型 (Agent + Callable)"""
    agent = MockAgent(return_value="agent_done")
    
    async def func_task(x):
        return f"func_{x}"
    
    def sync_task(x):
        return f"sync_{x}"
    
    team = Team(members=[agent, func_task, sync_task])
    results = await team.run("input")
    
    # 验证结果对象
    assert results[0].result == "agent_done"
    assert results[1].result == "func_input"
    assert results[2].result == "sync_input"
    assert all(r.is_success for r in results)

@pytest.mark.asyncio
async def test_invalid_member_type():
    """测试无效的成员类型"""
    team = Team(members=["not_callable_string"]) # type: ignore
    
    results = await team.run("input")
    
    assert len(results) == 1
    assert results[0].is_success is False
    assert "not executable" in results[0].error

@pytest.mark.asyncio
async def test_partial_failure():
    """测试部分失败容错机制"""
    async def success_task(x):
        return "success"
    
    async def failing_task(x):
        raise ValueError("Boom!")
    
    team = Team(members=[success_task, failing_task])
    
    with patch("gecko.compose.team.logger") as mock_logger:
        results = await team.run("input")
        
        # 验证结果结构
        assert results[0].result == "success"
        assert results[0].is_success is True
        
        assert results[1].is_success is False
        assert results[1].result is None
        assert "Boom!" in results[1].error
        
        mock_logger.error.assert_called()

@pytest.mark.asyncio
async def test_data_handover_no_cleaning():
    """
    [Updated] 测试 Data Handover 不再自动清洗
    验证字典能够原样传递给下游，而不是只提取 content
    """
    team = Team([])
    
    # 模拟上一个节点返回的是复杂字典
    complex_output = {
        "content": "clean_text",
        "tool_calls": [],
        "usage": {"total_tokens": 100}
    }
    
    ctx = MockContext(input_data="x", history={"last_output": complex_output})
    
    # 断言：现在应该返回完整的字典，而不是仅仅是 "clean_text"
    resolved = team._resolve_input(ctx)
    assert resolved == complex_output
    assert resolved["usage"]["total_tokens"] == 100

@pytest.mark.asyncio
async def test_member_result_value_property():
    """测试 MemberResult.value 便捷属性"""
    success_res = MemberResult(member_index=0, result="ok", is_success=True)
    assert success_res.value == "ok"
    
    fail_res = MemberResult(member_index=1, error="failed", is_success=False)
    with pytest.raises(RuntimeError, match="failed"):
        _ = fail_res.value