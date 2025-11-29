# gecko/core/telemetry.py
"""
OpenTelemetry 集成模块

提供分布式追踪能力，支持：
- 自动 Span 创建
- 上下文传播
- 与主流 APM 系统集成
"""

from __future__ import annotations

import functools
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from gecko.core.logging import get_logger

logger = get_logger(__name__)

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, StatusCode, Tracer

# 使用新的常量方式而非 ResourceAttributes
OTEL_AVAILABLE = True

T = TypeVar("T")

# 请求 ID 上下文
request_id_var: ContextVar[str] = ContextVar("gecko_request_id", default="")


class TelemetryConfig:
    """遥测配置"""

    def __init__(
        self,
        service_name: str = "gecko",
        service_version: str = "0.2.0",
        environment: str = "development",
        enabled: bool = True,
        exporter: Optional[SpanExporter] = None,
        # 新增配置项
        batch_export_interval_ms: int = 5000,
        max_export_batch_size: int = 512,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.enabled = enabled and OTEL_AVAILABLE
        self.exporter = exporter
        self.batch_export_interval_ms = batch_export_interval_ms
        self.max_export_batch_size = max_export_batch_size


class GeckoTelemetry:
    """
    Gecko 遥测管理器

    示例:
```python
    # 初始化
    telemetry = GeckoTelemetry(TelemetryConfig(
        service_name="my-agent-service",
        environment="production"
    ))
    telemetry.setup()

    # 使用装饰器
    @telemetry.trace_async("process_request")
    async def process_request(data):
        ...

    # 使用上下文管理器
    async with telemetry.async_span("custom_operation") as span:
        if span:
            span.set_attribute("custom.key", "value")
        await do_something()
```
    """

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self._tracer: Optional[Tracer] = None
        self._provider: Optional[SDKTracerProvider] = None
        self._initialized = False

    def setup(self) -> None:
        """初始化 OpenTelemetry"""
        if not self.config.enabled:
            logger.info("Telemetry disabled")
            return

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not installed. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )
            return

        try:
            # 创建 Resource - 使用字符串常量替代废弃的 ResourceAttributes
            resource = Resource.create(
                {
                    "service.name": self.config.service_name,
                    "service.version": self.config.service_version,
                    "deployment.environment": self.config.environment,
                }
            )

            # 创建 TracerProvider
            self._provider = SDKTracerProvider(resource=resource)

            # 添加 Exporter
            if self.config.exporter:
                processor = BatchSpanProcessor(
                    self.config.exporter,
                    schedule_delay_millis=self.config.batch_export_interval_ms,
                    max_export_batch_size=self.config.max_export_batch_size,
                )
                self._provider.add_span_processor(processor)

            # 设置全局 Provider
            trace.set_tracer_provider(self._provider)

            # 获取 Tracer
            self._tracer = trace.get_tracer(
                self.config.service_name, self.config.service_version
            )

            self._initialized = True
            logger.info(
                "Telemetry initialized",
                service=self.config.service_name,
                environment=self.config.environment,
            )

        except Exception as e:
            logger.error("Failed to initialize telemetry", error=str(e), exc_info=True)

    def shutdown(self) -> None:
        """关闭遥测并刷新所有 span"""
        if self._provider:
            try:
                self._provider.shutdown()
                logger.info("Telemetry shutdown successfully")
            except Exception as e:
                logger.error("Failed to shutdown telemetry", error=str(e))

    @property
    def tracer(self) -> Optional[Tracer]:
        """获取 Tracer"""
        return self._tracer

    @property
    def is_enabled(self) -> bool:
        """是否已启用"""
        return self._initialized and self._tracer is not None

    # ==================== Span 创建 ==================== 

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[trace.SpanKind] = None,
    ):
        """
        创建 Span(同步上下文管理器)
        
        Args:
            name: Span 名称
            attributes: 初始属性
            kind: Span 类型
        """
        if not self.is_enabled:
            yield None
            return

        span_kind = kind or trace.SpanKind.INTERNAL

        with self._tracer.start_as_current_span( # type: ignore
            name, kind=span_kind, attributes=attributes or {}
        ) as span:
            # 注入请求 ID
            request_id = request_id_var.get()
            if request_id:
                span.set_attribute("gecko.request_id", request_id)

            try:
                yield span
            except Exception as e:
                # 修复: 使用正确的 set_status 方式
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[trace.SpanKind] = None,
    ):
        """
        创建 Span(异步上下文管理器)
        
        Args:
            name: Span 名称
            attributes: 初始属性
            kind: Span 类型
        """
        if not self.is_enabled:
            yield None
            return

        span_kind = kind or trace.SpanKind.INTERNAL

        with self._tracer.start_as_current_span( # type: ignore
            name, kind=span_kind, attributes=attributes or {}
        ) as span:
            request_id = request_id_var.get()
            if request_id:
                span.set_attribute("gecko.request_id", request_id)

            try:
                yield span
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    # ==================== 装饰器 ====================

    def trace(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        同步函数追踪装饰器
        
        Args:
            name: 自定义 Span 名称,默认使用函数名
            attributes: 初始属性
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.span(span_name, attributes):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def trace_async(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        异步函数追踪装饰器
        
        Args:
            name: 自定义 Span 名称,默认使用函数名
            attributes: 初始属性
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                async with self.async_span(span_name, attributes):
                    return await func(*args, **kwargs) # type: ignore

            return wrapper # type: ignore

        return decorator

    # ==================== 工具方法 ====================

    def set_request_id(self, request_id: str) -> None:
        """设置当前请求 ID"""
        request_id_var.set(request_id)

    def get_request_id(self) -> str:
        """获取当前请求 ID"""
        return request_id_var.get()

    def add_event(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """向当前 Span 添加事件"""
        if not self.is_enabled:
            return

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any) -> None:
        """向当前 Span 设置属性"""
        if not self.is_enabled:
            return

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, value)

    def get_current_span(self) -> Optional[Span]:
        """获取当前活动的 Span"""
        if not self.is_enabled:
            return None
        return trace.get_current_span()


# 全局遥测实例
_telemetry: Optional[GeckoTelemetry] = None


def get_telemetry() -> GeckoTelemetry:
    """获取全局遥测实例"""
    global _telemetry
    if _telemetry is None:
        _telemetry = GeckoTelemetry()
    return _telemetry


def configure_telemetry(config: TelemetryConfig) -> GeckoTelemetry:
    """配置并初始化遥测"""
    global _telemetry
    _telemetry = GeckoTelemetry(config)
    _telemetry.setup()
    return _telemetry


# ==================== 预置 Span 名称 ====================


class SpanNames:
    """预定义的 Span 名称常量"""

    # Agent
    AGENT_RUN = "gecko.agent.run"
    AGENT_STREAM = "gecko.agent.stream"

    # Engine
    ENGINE_STEP = "gecko.engine.step"
    ENGINE_TOOL_CALL = "gecko.engine.tool_call"

    # Workflow
    WORKFLOW_EXECUTE = "gecko.workflow.execute"
    WORKFLOW_NODE = "gecko.workflow.node"

    # Storage
    STORAGE_GET = "gecko.storage.get"
    STORAGE_SET = "gecko.storage.set"
    STORAGE_DELETE = "gecko.storage.delete"

    # Model
    MODEL_COMPLETION = "gecko.model.completion"
    MODEL_STREAM = "gecko.model.stream"
    MODEL_EMBEDDING = "gecko.model.embedding"


# ==================== 导出 ====================

__all__ = [
    "GeckoTelemetry",
    "TelemetryConfig",
    "SpanNames",
    "get_telemetry",
    "configure_telemetry",
    "request_id_var",
    "OTEL_AVAILABLE",
]