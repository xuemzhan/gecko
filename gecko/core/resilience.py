# gecko/core/resilience.py
"""
系统韧性模块 (Resilience)

核心功能：
- CircuitBreaker: 熔断器，防止故障级联，保护系统资源。
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Tuple, Type, TypeVar

from gecko.core.exceptions import GeckoError, ModelError
from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"       # 正常状态：允许请求通过
    OPEN = "open"           # 熔断状态：拒绝请求，快速失败
    HALF_OPEN = "half_open" # 半开状态：允许少量请求尝试恢复


class CircuitOpenError(GeckoError):
    """熔断器开启异常，表示服务暂时不可用"""
    pass


class CircuitBreaker:
    """
    熔断器实现 (线程安全/协程安全)
    
    配置参数:
        failure_threshold: 触发熔断的连续失败次数
        recovery_timeout: 熔断后进入半开状态的冷却时间(秒)
        monitor_exceptions: 触发计数增加的异常类型元组
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        monitor_exceptions: Tuple[Type[Exception], ...] = (ModelError,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.monitor_exceptions = monitor_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()  # 保护状态变更的互斥锁

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        通过熔断器执行函数
        
        Args:
            func: 目标异步函数
            *args, **kwargs: 传递给函数的参数
        """
        # 1. 状态检查 (加锁读取以保证一致性)
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否过了冷却期
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("CircuitBreaker entering HALF_OPEN state")
                else:
                    raise CircuitOpenError("Service suspended due to high failure rate")

        try:
            # 2. 执行实际逻辑
            result = await func(*args, **kwargs)

            # 3. 成功后逻辑
            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下成功，立即关闭熔断器（恢复正常）
                async with self._lock:
                    self._reset()
            elif self._failure_count > 0:
                # [性能优化] 只有在 CLOSED 状态且有失败计数时才获取锁进行重置
                # 避免每次成功调用都请求锁，提升高并发下的吞吐量
                async with self._lock:
                    if self._failure_count > 0: # Double check
                        self._failure_count = 0
            
            return result

        except self.monitor_exceptions as e:
            # 4. 捕获监控范围内的异常，记录失败
            await self._record_failure()
            raise e

    async def _record_failure(self):
        """记录失败并判断是否熔断"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            # 如果在半开状态失败，或者达到阈值，立即熔断
            if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    logger.error(
                        f"CircuitBreaker OPENED. Failures: {self._failure_count}/{self.failure_threshold}"
                    )

    def _reset(self):
        """重置熔断器状态"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        logger.info("CircuitBreaker CLOSED (Service Recovered)")