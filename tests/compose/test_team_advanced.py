# tests/compose/test_team_advanced.py
import pytest
import asyncio
import time
from gecko.compose.team import Team, ExecutionStrategy

@pytest.mark.asyncio
async def test_team_input_sharding():
    """测试输入分片功能"""
    
    # 模拟一个处理函数，简单返回输入
    async def worker(inp):
        return f"Processed: {inp}"
    
    members = [worker, worker, worker]
    
    # 定义分片逻辑：把 list 的第 i 个元素发给第 i 个 worker
    def shard_mapper(raw_input, idx):
        return raw_input[idx]
    
    team = Team(members, input_mapper=shard_mapper) # type: ignore
    
    # 输入是一个列表
    raw_data = ["Page 1", "Page 2", "Page 3"]
    
    results = await team.run(raw_data)
    
    # 验证结果顺序和内容
    assert len(results) == 3
    assert results[0].value == "Processed: Page 1"
    assert results[1].value == "Processed: Page 2"
    assert results[2].value == "Processed: Page 3"

@pytest.mark.asyncio
async def test_team_race_strategy():
    """测试赛马模式 (Race)"""
    
    # 慢马：需要 0.2s
    async def slow_horse(inp):
        try:
            await asyncio.sleep(0.2)
            return "Slow"
        except asyncio.CancelledError:
            # 验证会被取消
            return "Cancelled"

    # 快马：需要 0.05s
    async def fast_horse(inp):
        await asyncio.sleep(0.05)
        return "Fast"
        
    members = [slow_horse, fast_horse]
    
    team = Team(members, strategy=ExecutionStrategy.RACE)
    
    start_time = time.time()
    results = await team.run("start")
    duration = time.time() - start_time
    
    # 验证：
    # 1. 只返回了一个结果 (Winner)
    assert len(results) == 1
    # 2. 赢家是快马
    assert results[0].value == "Fast"
    assert results[0].member_index == 1
    # 3. 总耗时接近快马的时间 (0.05s)，而不是慢马 (0.2s)
    #    给一点 buffer，断言 < 0.15s 即可证明并没有等慢马
    assert duration < 0.15

@pytest.mark.asyncio
async def test_team_race_all_fail():
    """测试赛马模式下全员失败"""
    async def failing_horse(inp):
        raise ValueError("Trip")
        
    members = [failing_horse, failing_horse]
    team = Team(members, strategy=ExecutionStrategy.RACE) # type: ignore
    
    results = await team.run("start")

    # 预期：没有赢家，返回包含每个成员失败信息的 MemberResult 列表
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        # 每个元素都是 MemberResult，标记为失败并包含错误信息
        assert hasattr(r, "is_success") and r.is_success is False
        assert getattr(r, "error", None) is not None