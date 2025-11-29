# gecko/core/retry.py
"""
统一重试策略模块

提供可配置的重试机制，支持：
- 多种退避策略
- 熔断器模式
- 异常过滤
- 执行回调
"""
from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar, Union

from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BackoffStrategy(str, Enum):
    """退避策略"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """
    重试配置
    
    示例:
        ```python
        config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=30.0,
            backoff=BackoffStrategy.EXPONENTIAL,
            retryable=(ConnectionError, TimeoutError),
        )
        ```
    """
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    multiplier: float = 2.0
    jitter: float = 0.1
    retryable: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable: Tuple[Type[Exception], ...] = ()

    def calculate_delay(self, attempt: int) -> float:
        """计算第 N 次重试的延迟"""
        if attempt <= 0:
            return 0.0

        if self.backoff == BackoffStrategy.CONSTANT:
            base = self.initial_delay
        elif self.backoff == BackoffStrategy.LINEAR:
            base = self.initial_delay * attempt
        else:  # EXPONENTIAL
            base = self.initial_delay * (self.multiplier ** (attempt - 1))

        delay = min(base, self.max_delay)

        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """判断是否应重试"""
        if isinstance(exception, self.non_retryable):
            return False
        return isinstance(exception, self.retryable)


@dataclass
class RetryState:
    """重试状态追踪"""
    attempt: int = 0
    total_delay: float = 0.0
    last_exception: Optional[Exception] = None
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


class Retrier:
    """
    重试执行器
    
    示例:
        ```python
        retrier = Retrier(RetryConfig(max_attempts=5))
        
        # 异步使用
        result = await retrier.run_async(unstable_function, arg1, arg2)
        
        # 同步使用
        result = retrier.run_sync(sync_function, arg1)
        ```
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        on_success: Optional[Callable[[int], None]] = None,
        on_failure: Optional[Callable[[int, Exception], None]] = None,
    ):
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure

    async def run_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """执行异步函数（带重试）"""
        state = RetryState()

        for attempt in range(1, self.config.max_attempts + 1):
            state.attempt = attempt

            try:
                result = await func(*args, **kwargs)
                self._call_success(attempt)
                return result

            except Exception as e:
                state.last_exception = e

                if not self.config.should_retry(e):
                    logger.debug(f"Exception not retryable: {type(e).__name__}")
                    raise

                if attempt >= self.config.max_attempts:
                    self._call_failure(attempt, e)
                    raise

                delay = self.config.calculate_delay(attempt)
                state.total_delay += delay

                self._call_retry(attempt, e, delay)
                logger.warning(
                    f"Retry {attempt}/{self.config.max_attempts} "
                    f"after {delay:.2f}s: {e}"
                )

                await asyncio.sleep(delay)

        # 不应到达此处
        raise state.last_exception  # type: ignore

    def run_sync(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """执行同步函数（带重试）"""
        state = RetryState()

        for attempt in range(1, self.config.max_attempts + 1):
            state.attempt = attempt

            try:
                result = func(*args, **kwargs)
                self._call_success(attempt)
                return result

            except Exception as e:
                state.last_exception = e

                if not self.config.should_retry(e):
                    raise

                if attempt >= self.config.max_attempts:
                    self._call_failure(attempt, e)
                    raise

                delay = self.config.calculate_delay(attempt)
                state.total_delay += delay

                self._call_retry(attempt, e, delay)
                logger.warning(
                    f"Retry {attempt}/{self.config.max_attempts} "
                    f"after {delay:.2f}s: {e}"
                )

                time.sleep(delay)

        raise state.last_exception  # type: ignore

    def _call_retry(self, attempt: int, exc: Exception, delay: float) -> None:
        if self.on_retry:
            try:
                self.on_retry(attempt, exc, delay)
            except Exception as e:
                logger.warning(f"on_retry callback failed: {e}")

    def _call_success(self, attempt: int) -> None:
        if self.on_success:
            try:
                self.on_success(attempt)
            except Exception as e:
                logger.warning(f"on_success callback failed: {e}")

    def _call_failure(self, attempt: int, exc: Exception) -> None:
        if self.on_failure:
            try:
                self.on_failure(attempt, exc)
            except Exception as e:
                logger.warning(f"on_failure callback failed: {e}")


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    retryable: Tuple[Type[Exception], ...] = (Exception,),
    **config_kwargs
) -> Callable:
    """
    重试装饰器
    
    示例:
        ```python
        @retry(max_attempts=5, initial_delay=2.0)
        async def unstable_api_call():
            ...
        
        @retry(retryable=(ConnectionError,))
        def sync_operation():
            ...
        ```
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff=backoff,
        retryable=retryable,
        **config_kwargs
    )
    retrier = Retrier(config)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retrier.run_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return retrier.run_sync(func, *args, **kwargs)
            return sync_wrapper

    return decorator


# ==================== 预设配置 ====================

class RetryPresets:
    """预设重试配置"""

    @staticmethod
    def api_calls() -> RetryConfig:
        """API 调用"""
        return RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=30.0,
            backoff=BackoffStrategy.EXPONENTIAL,
            retryable=(ConnectionError, TimeoutError, OSError),
        )

    @staticmethod
    def llm_calls() -> RetryConfig:
        """LLM API 调用"""
        return RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=60.0,
            backoff=BackoffStrategy.EXPONENTIAL,
            jitter=0.2,
        )

    @staticmethod
    def storage() -> RetryConfig:
        """存储操作"""
        return RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=10.0,
            backoff=BackoffStrategy.EXPONENTIAL,
        )

    @staticmethod
    def quick() -> RetryConfig:
        """快速重试（2次）"""
        return RetryConfig(
            max_attempts=2,
            initial_delay=0.1,
            max_delay=1.0,
            backoff=BackoffStrategy.CONSTANT,
        )


__all__ = [
    "RetryConfig",
    "RetryState",
    "Retrier",
    "BackoffStrategy",
    "RetryPresets",
    "retry",
]