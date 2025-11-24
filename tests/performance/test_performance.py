# tests/performance/test_performance.py
"""
性能基准测试

确保重构没有引入性能回退
"""
import pytest
import time
import asyncio
from gecko.core.message import Message
from gecko.core.memory import TokenMemory

# ========== Message 序列化性能 ==========

def test_message_serialization_performance():
    """测试 Message 序列化性能"""
    msg = Message.user("Test message " * 50)  # 中等长度消息
    
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        msg.to_openai_format()
    duration = time.perf_counter() - start
    
    ops_per_sec = iterations / duration
    
    print(f"\nMessage serialization: {ops_per_sec:.0f} ops/sec")
    
    # 应该至少 10000 ops/sec
    assert ops_per_sec > 10000, f"Too slow: {ops_per_sec:.0f} ops/sec"

def test_message_deserialization_performance():
    """测试 Message 反序列化性能"""
    data = {
        "role": "assistant",
        "content": "Response " * 50,
        "tool_calls": None
    }
    
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        Message(**data)
    duration = time.perf_counter() - start
    
    ops_per_sec = iterations / duration
    
    print(f"\nMessage deserialization: {ops_per_sec:.0f} ops/sec")
    assert ops_per_sec > 10000

# ========== Memory Token 计数性能 ==========

def test_token_counting_performance():
    """测试 Token 计数性能"""
    memory = TokenMemory(session_id="bench", max_tokens=4000)
    
    messages = [
        Message.user(f"Message {i}: " + "test " * 20)
        for i in range(100)
    ]
    
    start = time.perf_counter()
    for msg in messages:
        memory.count_message_tokens(msg)
    duration = time.perf_counter() - start
    
    msgs_per_sec = len(messages) / duration
    
    print(f"\nToken counting: {msgs_per_sec:.0f} messages/sec")
    assert msgs_per_sec > 100  # 应该至少 100 msgs/sec

# ========== 日志性能 ==========

def test_logging_overhead():
    """测试日志开销"""
    from gecko.core.logging import get_logger
    
    logger = get_logger("benchmark")
    
    iterations = 1000
    start = time.perf_counter()
    for i in range(iterations):
        logger.debug("benchmark event", iteration=i, value=i*2)
    duration = time.perf_counter() - start
    
    logs_per_sec = iterations / duration
    
    print(f"\nLogging throughput: {logs_per_sec:.0f} logs/sec")
    
    # 日志不应该拖慢系统，应该至少 1000 logs/sec
    assert logs_per_sec > 1000, f"Logging too slow: {logs_per_sec:.0f} logs/sec"

# ========== 端到端性能 ==========

@pytest.mark.asyncio
async def test_agent_execution_baseline():
    """测试 Agent 执行基准性能"""
    from unittest.mock import AsyncMock, MagicMock
    from gecko.core.agent import Agent
    from gecko.core.toolbox import ToolBox

    # [Fix] 使用 MagicMock 作为基类，因为 ModelProtocol 包含同步和异步方法
    model = MagicMock()
    
    # 1. 模拟异步推理方法
    model.acompletion = AsyncMock()
    
    # 2. [Fix] 模拟新增的同步计数方法，以通过 isinstance(model, ModelProtocol) 检查
    model.count_tokens = MagicMock(return_value=10)

    # 配置返回值
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Quick response",
        "tool_calls": None
    }
    model.acompletion.return_value = mock_response

    # 创建 Agent
    toolbox = ToolBox([])
    memory = TokenMemory(session_id="bench")
    
    # 现在 model 满足 ModelProtocol 协议，不会抛出 TypeError
    agent = Agent(model=model, toolbox=toolbox, memory=memory)

    # 测试 10 次执行
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        await agent.run([Message.user("test")])
    duration = time.perf_counter() - start

    avg_time = duration / iterations

    print(f"\nAgent execution: {avg_time*1000:.2f} ms/run")

    # 单次执行应该在 50ms 内（Mock 模式）
    assert avg_time < 0.05, f"Agent too slow: {avg_time*1000:.2f} ms"

# ========== 内存使用测试 ==========

def test_memory_usage():
    """测试内存占用"""
    import sys
    
    # 创建大量消息
    messages = [
        Message.user(f"Message {i}")
        for i in range(1000)
    ]
    
    # 粗略估算内存占用
    size = sys.getsizeof(messages)
    per_message = size / len(messages)
    
    print(f"\nMemory per message: ~{per_message:.0f} bytes")
    
    # 单个消息应该小于 1KB
    assert per_message < 1024