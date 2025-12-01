# tests/core/memory/test_summary_optimization.py
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from gecko.core.memory.summary import SummaryTokenMemory
from gecko.core.message import Message

@pytest.mark.asyncio
async def test_summary_debounce_and_background():
    """验证摘要更新的防抖和后台执行逻辑"""
    
    # 1. Setup Mock Model
    mock_model = MagicMock()
    # 模拟一个慢速的 LLM 响应 (0.1s)
    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(0.1) 
        mock_resp = MagicMock()
        mock_resp.choices[0].message.get.return_value = "New Summary"
        return mock_resp
    
    mock_model.acompletion = AsyncMock(side_effect=slow_completion)
    # Token 计数 Mock
    mock_model.count_tokens.return_value = 10 

    # 2. 初始化 Memory
    # max_tokens 很小，迫使每次都触发摘要
    # min_update_interval=0.2s, background=True
    memory = SummaryTokenMemory(
        session_id="test",
        model=mock_model,
        max_tokens=50, 
        summary_reserve_tokens=10,
        min_update_interval=0.2,
        background_update=True
    )

    # 构造一批消息，总长度超过 max_tokens
    msgs = [{"role": "user", "content": "msg"}] * 10 

    # 3. 第一次调用 (Trigger)
    start_time = time.time()
    await memory.get_history(msgs)
    duration = time.time() - start_time
    
    # 验证：非阻塞返回
    # 因为 background=True，get_history 应该瞬间返回，不等待 LLM (0.1s)
    assert duration < 0.05, "Background update failed, blocked main thread"
    
    # 等待后台任务完成
    await asyncio.sleep(0.15)
    assert memory.current_summary == "New Summary"
    assert mock_model.acompletion.call_count == 1

    # 4. 第二次调用 (Debounce)
    # 此时距离上次更新 < 0.2s，应该触发防抖，不调用 LLM
    await memory.get_history(msgs)
    
    # 验证：LLM 调用次数未增加
    assert mock_model.acompletion.call_count == 1
    
    # 5. 第三次调用 (Expired)
    # 等待超过间隔
    await asyncio.sleep(0.2)
    await memory.get_history(msgs)
    
    # 验证：再次触发后台任务
    # 给一点时间让后台任务启动
    await asyncio.sleep(0.01) 
    # 注意：acompletion 可能还没完成，但应该已经被调用了
    # 这里的断言取决于 create_task 的调度速度，通常 call_count 会变成 2
    # 为了稳健，我们等待一下
    await asyncio.sleep(0.15)
    assert mock_model.acompletion.call_count == 2