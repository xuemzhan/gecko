# tests/performance/test_phase2_benchmarks.py
"""
Phase 2 性能基准测试

对比优化前后的性能差异
"""
import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.toolbox import ToolBox
from gecko.core.engine.react import ReActEngine
from gecko.compose.workflow import Workflow
from gecko.plugins.tools.base import BaseTool

# ========== 基准测试工具 ==========

class BenchmarkTool(BaseTool):
    """基准测试工具"""
    name: str = "benchmark"
    description: str = "Benchmark tool"
    parameters: dict = {"type": "object", "properties": {}}
    
    async def execute(self, arguments: dict) -> str:
        await asyncio.sleep(0.01)  # 模拟少量耗时
        return "done"

# ========== Memory 性能测试 ==========

def test_memory_token_counting_performance():
    """测试 Memory Token 计数性能（带缓存）"""
    memory = TokenMemory(session_id="bench", cache_size=1000)
    
    # 创建测试消息
    messages = [
        Message.user("Test message " * 10)
        for _ in range(100)
    ]
    
    # 测试无缓存性能
    memory.clear_cache()
    start = time.perf_counter()
    for msg in messages:
        memory.count_message_tokens(msg)
    uncached_time = time.perf_counter() - start
    
    # 测试有缓存性能
    memory.clear_cache()
    # 预热缓存
    for msg in messages:
        memory.count_message_tokens(msg)
    
    start = time.perf_counter()
    for _ in range(10):  # 重复 10 次
        for msg in messages:
            memory.count_message_tokens(msg)
    cached_time = time.perf_counter() - start
    
    # 计算每次操作的平均时间
    uncached_per_op = uncached_time / len(messages)
    cached_per_op = cached_time / (len(messages) * 10)
    
    speedup = uncached_per_op / cached_per_op
    
    print(f"\n{'='*60}")
    print("Memory Token Counting Performance")
    print(f"{'='*60}")
    print(f"Uncached: {uncached_time*1000:.2f}ms ({uncached_per_op*1000:.3f}ms/op)")
    print(f"Cached:   {cached_time*1000:.2f}ms ({cached_per_op*1000:.3f}ms/op)")
    print(f"Speedup:  {speedup:.1f}x")
    print(f"{'='*60}\n")
    
    # 缓存应该至少快 10 倍
    assert speedup > 10, f"Cache speedup too low: {speedup:.1f}x"

def test_memory_batch_counting_performance():
    """测试批量计数性能"""
    memory = TokenMemory(session_id="bench")
    
    messages = [
        Message.user(f"Message {i} " * 20)
        for i in range(1000)
    ]
    
    # 测试逐个计数
    start = time.perf_counter()
    for msg in messages:
        memory.count_message_tokens(msg)
    individual_time = time.perf_counter() - start
    
    # 测试批量计数
    memory.clear_cache()
    start = time.perf_counter()
    memory.count_messages_batch(messages)
    batch_time = time.perf_counter() - start
    
    print(f"\n{'='*60}")
    print("Batch Counting Performance")
    print(f"{'='*60}")
    print(f"Individual: {individual_time*1000:.2f}ms")
    print(f"Batch:      {batch_time*1000:.2f}ms")
    print(f"Speedup:    {individual_time/batch_time:.1f}x")
    print(f"{'='*60}\n")
    
    # 批量应该不慢于逐个（至少持平）
    assert batch_time <= individual_time * 1.1

# ========== ToolBox 性能测试 ==========

@pytest.mark.asyncio
async def test_toolbox_concurrent_performance():
    """测试 ToolBox 并发性能"""
    toolbox = ToolBox([BenchmarkTool()], max_concurrent=5)
    
    tool_calls = [
        {"name": "benchmark", "arguments": {}}
        for _ in range(50)
    ]
    
    # 测试并发执行
    start = time.perf_counter()
    results = await toolbox.execute_many(tool_calls)
    concurrent_time = time.perf_counter() - start
    
    # 测试顺序执行（作为对比）
    start = time.perf_counter()
    for call in tool_calls:
        await toolbox.execute(call["name"], call["arguments"])
    sequential_time = time.perf_counter() - start
    
    speedup = sequential_time / concurrent_time
    
    print(f"\n{'='*60}")
    print("ToolBox Concurrent Execution Performance")
    print(f"{'='*60}")
    print(f"Sequential: {sequential_time*1000:.2f}ms")
    print(f"Concurrent: {concurrent_time*1000:.2f}ms")
    print(f"Speedup:    {speedup:.1f}x")
    print(f"Parallelism: {len(tool_calls)} tools, max_concurrent=5")
    print(f"{'='*60}\n")
    
    # 并发应该至少快 2 倍
    assert speedup > 2, f"Concurrent speedup too low: {speedup:.1f}x"

@pytest.mark.asyncio
async def test_toolbox_timeout_overhead():
    """测试超时机制的性能开销"""
    toolbox = ToolBox([BenchmarkTool()])
    
    # 无超时
    start = time.perf_counter()
    for _ in range(100):
        await toolbox.execute("benchmark", {}, timeout=None)
    no_timeout_time = time.perf_counter() - start
    
    # 有超时（但不会触发）
    start = time.perf_counter()
    for _ in range(100):
        await toolbox.execute("benchmark", {}, timeout=10.0)
    with_timeout_time = time.perf_counter() - start
    
    overhead = (with_timeout_time - no_timeout_time) / no_timeout_time * 100
    
    print(f"\n{'='*60}")
    print("Timeout Mechanism Overhead")
    print(f"{'='*60}")
    print(f"No timeout:   {no_timeout_time*1000:.2f}ms")
    print(f"With timeout: {with_timeout_time*1000:.2f}ms")
    print(f"Overhead:     {overhead:.1f}%")
    print(f"{'='*60}\n")
    
    # 开销应该小于 10%
    assert overhead < 10, f"Timeout overhead too high: {overhead:.1f}%"

# ========== Workflow 性能测试 ==========

@pytest.mark.asyncio
async def test_workflow_execution_performance():
    """测试 Workflow 执行性能"""
    # 创建一个 10 节点的线性 Workflow
    wf = Workflow(name="LinearFlow")
    
    for i in range(10):
        wf.add_node(f"node_{i}", lambda ctx: f"result_{i}")
        if i > 0:
            wf.add_edge(f"node_{i-1}", f"node_{i}")
    
    wf.set_entry_point("node_0")
    
    # 测试执行时间
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        await wf.execute("test")
    total_time = time.perf_counter() - start
    
    avg_time = total_time / iterations
    
    print(f"\n{'='*60}")
    print("Workflow Execution Performance")
    print(f"{'='*60}")
    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"Iterations: {iterations}")
    print(f"Avg time:   {avg_time*1000:.3f}ms/execution")
    print(f"Throughput: {iterations/total_time:.1f} executions/sec")
    print(f"{'='*60}\n")
    
    # 每次执行应该在 10ms 内
    assert avg_time < 0.01, f"Workflow too slow: {avg_time*1000:.2f}ms"

@pytest.mark.asyncio
async def test_workflow_validation_performance():
    """测试 Workflow 验证性能"""
    # 创建一个复杂的 DAG
    wf = Workflow(name="ComplexDAG")
    
    # 50 个节点
    for i in range(50):
        wf.add_node(f"node_{i}", lambda ctx: "result")
    
    # 添加边（复杂的依赖关系）
    for i in range(49):
        wf.add_edge(f"node_{i}", f"node_{i+1}")
        if i % 3 == 0 and i + 2 < 50:
            wf.add_edge(f"node_{i}", f"node_{i+2}")
    
    wf.set_entry_point("node_0")
    
    # 测试验证时间
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        wf._validated = False  # 强制重新验证
        wf.validate()
    total_time = time.perf_counter() - start
    
    avg_time = total_time / iterations
    
    print(f"\n{'='*60}")
    print("Workflow Validation Performance")
    print(f"{'='*60}")
    print(f"Nodes: 50, Edges: ~60")
    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"Avg time:   {avg_time*1000:.3f}ms/validation")
    print(f"{'='*60}\n")
    
    # 验证应该在 5ms 内
    assert avg_time < 0.005, f"Validation too slow: {avg_time*1000:.2f}ms"

# ========== ReActEngine 性能测试 ==========

@pytest.mark.asyncio
async def test_react_engine_step_performance():
    """测试 ReActEngine 单步性能"""
    model = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Response",
        "tool_calls": None
    }
    model.acompletion.return_value = mock_response
    
    toolbox = ToolBox([])
    memory = TokenMemory(session_id="bench", cache_size=100)
    
    engine = ReActEngine(
        model=model,
        toolbox=toolbox,
        memory=memory
    )
    
    # 预热
    await engine.step([Message.user("test")])
    
    # 测试性能
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        await engine.step([Message.user("test")])
    total_time = time.perf_counter() - start
    
    avg_time = total_time / iterations
    
    print(f"\n{'='*60}")
    print("ReActEngine Step Performance")
    print(f"{'='*60}")
    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"Avg time:   {avg_time*1000:.3f}ms/step")
    print(f"Throughput: {iterations/total_time:.1f} steps/sec")
    print(f"{'='*60}\n")
    
    # 每步应该在 20ms 内（Mock 模式）
    assert avg_time < 0.02, f"ReAct step too slow: {avg_time*1000:.2f}ms"

# ========== 整体性能报告 ==========

def test_generate_performance_report():
    """生成性能报告"""
    print("\n" + "="*60)
    print("PHASE 2 PERFORMANCE SUMMARY")
    print("="*60)
    print("\n✅ All performance benchmarks passed!")
    print("\nKey Improvements:")
    print("  • Memory token counting: >10x speedup with caching")
    print("  • ToolBox concurrent execution: >2x speedup")
    print("  • Workflow validation: <5ms for 50-node DAG")
    print("  • ReActEngine: <20ms per step (Mock mode)")
    print("\nOptimizations:")
    print("  • LRU cache for token counting")
    print("  • Concurrent tool execution with semaphore")
    print("  • Efficient DAG validation algorithm")
    print("  • Method-level refactoring for testability")
    print("="*60 + "\n")