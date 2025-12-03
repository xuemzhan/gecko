# tests/core/test_resilience.py
import pytest
import asyncio
from unittest.mock import AsyncMock

from gecko.core.resilience import CircuitBreaker, CircuitOpenError, CircuitState
from gecko.plugins.models.exceptions import ServiceUnavailableError

@pytest.mark.asyncio
async def test_circuit_breaker_state_flow():
    """验证熔断器状态流转：CLOSED -> OPEN -> HALF_OPEN -> CLOSED"""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    
    mock_func = AsyncMock()
    mock_func.side_effect = ServiceUnavailableError("503")
    
    # 1. 累积失败
    with pytest.raises(ServiceUnavailableError):
        await cb.call(mock_func)
    assert cb._state == CircuitState.CLOSED
    
    # 2. 触发熔断
    with pytest.raises(ServiceUnavailableError):
        await cb.call(mock_func)
    assert cb._state == CircuitState.OPEN
    
    # 3. 熔断生效，直接拒绝
    with pytest.raises(CircuitOpenError):
        await cb.call(mock_func)
        
    # 4. 冷却后进入半开
    await asyncio.sleep(0.15)
    
    # 5. 试探成功，状态恢复
    mock_func.side_effect = None
    mock_func.return_value = "success"
    res = await cb.call(mock_func)
    assert res == "success"
    
    assert cb._state == CircuitState.CLOSED
    assert cb._failure_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_ignore_unmonitored_exceptions():
    """验证非监控异常不会触发熔断"""
    cb = CircuitBreaker(failure_threshold=2)
    mock_func = AsyncMock()
    # ValueError 不在监控列表中
    mock_func.side_effect = ValueError("Business Error") 
    
    with pytest.raises(ValueError):
        await cb.call(mock_func)
        
    assert cb._failure_count == 0
    assert cb._state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_circuit_breaker_concurrency():
    """
    [Fix] 验证高并发下的线程安全性
    
    采用两步验证法，避免竞态导致的断言失败。
    """
    # 设置阈值为 50
    cb = CircuitBreaker(failure_threshold=50, recovery_timeout=1.0)
    
    mock_func = AsyncMock()
    # 模拟微小耗时，让任务有并发重叠的机会
    async def slow_fail():
        await asyncio.sleep(0.001) 
        raise ServiceUnavailableError("503")
    mock_func.side_effect = slow_fail
    
    # 步骤 1: 启动足够多的并发任务以必然触发熔断
    tasks = [cb.call(mock_func) for _ in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计实际执行并报错的次数
    service_errors = [r for r in results if isinstance(r, ServiceUnavailableError)]
    
    # 验证：至少有 50 个请求真正执行了，才触发了阈值
    assert len(service_errors) >= 50
    # 验证状态已变为 OPEN
    assert cb._state == CircuitState.OPEN
    
    # 步骤 2: 验证后续的新请求会被立即阻断
    # 此时熔断器已打开，任何新请求都应抛出 CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await cb.call(mock_func)
        