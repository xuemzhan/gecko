# gecko/core/telemetry.py
"""
OpenTelemetry 集成模块（v0.4.0 修复增强版）

提供分布式追踪能力，支持：
- 自动 Span 创建
- 上下文传播
- 与主流 APM 系统集成（可选依赖：未安装 opentelemetry 时自动降级）
- 与 Gecko logging 的 trace_context 联动（把 logging trace_id/span_id 注入到 span 属性）

核心设计原则（工业级）：
1) OpenTelemetry 作为“可选依赖”：
   - 未安装 opentelemetry 时，不应影响 gecko import / 单元测试 / demo 运行
2) TelemetryConfig 作为纯配置对象：
   - enabled 表示用户意图，不在 config 初始化阶段绑定 OTEL_AVAILABLE
   - OpenTelemetry 是否可用由 GeckoTelemetry.setup() 决定
3) 单一版本源：
   - service_version 默认从 gecko.version.__version__ 填充
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, TypeVar

from gecko.core.logging import get_logger

logger = get_logger(__name__)

# ===== 与 logging 追踪上下文集成（从 gecko.core.logging 导入上下文变量） =====
# 说明：logging 模块通常会提供 trace_id_var / span_id_var 等 ContextVar
# span 创建时，我们会把这两个值写进 span attributes 里，方便日志与链路追踪关联。
from gecko.core.logging import (
    trace_id_var as logging_trace_id_var,
    span_id_var as logging_span_id_var,
)

# ===== OpenTelemetry 可选依赖导入 =====
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.trace import Span, StatusCode, Tracer

    OTEL_AVAILABLE = True
except Exception:  # noqa: BLE001
    # 依赖不可用：保持模块可导入，运行时自动降级
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    SDKTracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    SpanExporter = Any  # type: ignore[misc,assignment]
    Span = Any  # type: ignore[misc,assignment]
    StatusCode = Any  # type: ignore[misc,assignment]
    Tracer = Any  # type: ignore[misc,assignment]
    OTEL_AVAILABLE = False

T = TypeVar("T")

# ===== 请求级上下文（可选） =====
# 用于把“同一请求”的标识注入到 span 中（例如 Web 请求/一次 agent run）
request_id_var: ContextVar[str] = ContextVar("gecko_request_id", default="")


@dataclass
class TelemetryConfig:
    """
    遥测配置（纯配置对象）

    ✅ 修复要点：
    - service_version 必须显式存在，避免 AttributeError
    - 默认从 gecko.version.__version__ 填充（单一来源）
    - enabled 仅代表“用户是否想启用 telemetry”，不要在这里绑定 OTEL_AVAILABLE
      （否则测试 patch OTEL_AVAILABLE=True 会失效）
    """
    service_name: str = "gecko"
    service_version: Optional[str] = None
    environment: str = "development"
    enabled: bool = True

    # exporter 可选：用户可注入 OTLP / Jaeger / Zipkin 等 exporter
    exporter: Optional[SpanExporter] = None # type: ignore

    # BatchSpanProcessor 参数
    batch_export_interval_ms: int = 5000
    max_export_batch_size: int = 512

    def __post_init__(self) -> None:
        # 延迟读取版本号，避免循环依赖与硬编码
        if self.service_version is None:
            from gecko.version import __version__ as gecko_version
            self.service_version = gecko_version


class GeckoTelemetry:
    """
    Gecko Telemetry 管理器

    - setup(): 初始化 OTel（若可用）
    - span()/async_span(): 创建 span（上下文管理器）
    - trace_sync()/trace_async(): 装饰器追踪
    - set_request_id(): 设置请求级标识
    """

    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        self.config = config or TelemetryConfig()
        self._tracer: Optional[Tracer] = None # type: ignore
        self._provider: Optional[SDKTracerProvider] = None # type: ignore
        self._initialized: bool = False

    def setup(self) -> None:
        """
        初始化 OpenTelemetry。

        逻辑：
        1) 若用户配置禁用，则直接返回
        2) 若 OpenTelemetry 未安装（OTEL_AVAILABLE=False），则降级并返回
        3) 若可用则初始化 Resource/Provider/Tracer
        """
        # 1) 用户配置禁用
        if not self.config.enabled:
            logger.info("Telemetry disabled by config")
            return

        # 2) 依赖不可用：降级，不影响主流程
        if (not OTEL_AVAILABLE) or (trace is None) or (Resource is None) or (SDKTracerProvider is None):
            logger.warning(
                "OpenTelemetry not installed; telemetry will be disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
            return

        try:
            # 3) 构建 Resource（避免使用 deprecated 的 ResourceAttributes）
            resource = Resource.create(
                {
                    "service.name": self.config.service_name,
                    "service.version": self.config.service_version,
                    "deployment.environment": self.config.environment,
                } # type: ignore
            )

            # 4) 创建 Provider
            self._provider = SDKTracerProvider(resource=resource)

            # 5) 配置 Exporter（如果用户提供 exporter）
            if self.config.exporter is not None and BatchSpanProcessor is not None:
                processor = BatchSpanProcessor(
                    self.config.exporter,
                    schedule_delay_millis=self.config.batch_export_interval_ms,
                    max_export_batch_size=self.config.max_export_batch_size,
                )
                # SDKTracerProvider 正常具备 add_span_processor
                self._provider.add_span_processor(processor) # type: ignore

            # 6) 设置全局 Provider 并获取 tracer
            trace.set_tracer_provider(self._provider) # type: ignore
            self._tracer = trace.get_tracer(self.config.service_name, self.config.service_version)

            self._initialized = True
            logger.info(
                "Telemetry initialized",
                service=self.config.service_name,
                version=self.config.service_version,
                environment=self.config.environment,
            )
        except Exception as e:  # noqa: BLE001
            # 初始化失败：不抛出异常阻塞框架，记录日志即可
            self._initialized = False
            logger.error("Failed to initialize telemetry", error=str(e), exc_info=True)

    def shutdown(self) -> None:
        """
        关闭遥测并 flush/span 处理器。

        注意：
        - shutdown 不应抛出异常影响主流程（生产中常在退出阶段调用）
        """
        if self._provider is None:
            return

        try:
            self._provider.shutdown()
            logger.info("Telemetry shutdown successfully")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to shutdown telemetry", error=str(e))

    @property
    def tracer(self) -> Optional[Tracer]: # type: ignore
        """获取 tracer（可能为 None：未 setup 或依赖不可用）"""
        return self._tracer

    @property
    def is_enabled(self) -> bool:
        """Telemetry 是否已实际启用（已 setup 成功且 tracer 可用）"""
        return bool(self._initialized and self._tracer is not None)

    # ====================== Span 创建（同步/异步） ======================

    def _inject_context_attributes(self, span: Any) -> None:
        """
        给 span 注入上下文属性：
        - gecko.logging.trace_id / gecko.logging.span_id
        - gecko.request_id

        说明：
        - 这些属性用于把日志与链路追踪关联起来
        - 若 span 不支持 set_attribute（mock 或 no-op），则忽略
        """
        try:
            trace_id = logging_trace_id_var.get()
            span_id = logging_span_id_var.get()
            if trace_id:
                span.set_attribute("gecko.logging.trace_id", trace_id)
            if span_id:
                span.set_attribute("gecko.logging.span_id", span_id)

            req_id = request_id_var.get()
            if req_id:
                span.set_attribute("gecko.request_id", req_id)
        except Exception:
            # 注入失败不应影响主流程
            return

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[Any] = None,
    ):
        """
        创建同步 span（上下文管理器）

        用法：
            with telemetry.span("op") as span:
                if span:
                    span.set_attribute("k", "v")
                ...
        """
        if not self.is_enabled or trace is None or self._tracer is None:
            yield None
            return

        span_kind = kind or trace.SpanKind.INTERNAL
        with self._tracer.start_as_current_span(  # type: ignore[attr-defined]
            name,
            kind=span_kind,
            attributes=attributes or {},
        ) as span:
            self._inject_context_attributes(span)
            yield span

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[Any] = None,
    ):
        """
        创建异步 span（异步上下文管理器）

        用法：
            async with telemetry.async_span("op") as span:
                ...
        """
        if not self.is_enabled or trace is None or self._tracer is None:
            yield None
            return

        span_kind = kind or trace.SpanKind.INTERNAL
        with self._tracer.start_as_current_span(  # type: ignore[attr-defined]
            name,
            kind=span_kind,
            attributes=attributes or {},
        ) as span:
            self._inject_context_attributes(span)
            yield span

    # ====================== 装饰器（同步/异步） ======================

    def trace_sync(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        同步函数追踪装饰器

        - name 为空时，默认使用 "module.qualname"
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

        - name 为空时，默认使用 "module.qualname"
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                async with self.async_span(span_name, attributes):
                    return await func(*args, **kwargs)  # type: ignore[misc]

            return wrapper  # type: ignore[misc]

        return decorator

    # ====================== 工具方法 ======================

    def set_request_id(self, request_id: str) -> None:
        """设置当前上下文的 request_id"""
        request_id_var.set(request_id)

    def get_request_id(self) -> str:
        """获取当前上下文的 request_id"""
        return request_id_var.get()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """向当前 span 添加事件（如果存在且 recording）"""
        if not self.is_enabled or trace is None:
            return

        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes=attributes or {})
        except Exception:
            return

    def set_attribute(self, key: str, value: Any) -> None:
        """向当前 span 设置属性（如果存在且 recording）"""
        if not self.is_enabled or trace is None:
            return

        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute(key, value)
        except Exception:
            return

    def get_current_span(self) -> Optional[Span]: # type: ignore
        """获取当前活动 span（如果可用）"""
        if not self.is_enabled or trace is None:
            return None
        try:
            return trace.get_current_span()
        except Exception:
            return None


# ====================== 全局实例与配置入口 ======================

_telemetry: Optional[GeckoTelemetry] = None


def get_telemetry() -> GeckoTelemetry:
    """
    获取全局 telemetry 实例（首次调用会自动 setup）。

    ✅ 版本一致性：
    - service_version 统一来自 gecko.version.__version__（单一来源）
    """
    from gecko.config import settings as _settings  # 延迟导入避免循环依赖
    from gecko.version import __version__ as gecko_version

    global _telemetry
    if _telemetry is None:
        cfg = TelemetryConfig(
            service_name=_settings.telemetry_service_name,
            service_version=gecko_version,
            environment=_settings.telemetry_environment,
            enabled=_settings.telemetry_enabled,
        )
        _telemetry = GeckoTelemetry(cfg)
        _telemetry.setup()
    return _telemetry


def configure_telemetry(config: TelemetryConfig) -> GeckoTelemetry:
    """显式配置全局 telemetry（会立即 setup）"""
    global _telemetry
    _telemetry = GeckoTelemetry(config)
    _telemetry.setup()
    return _telemetry


# ====================== 预置 Span 名称（可选） ======================

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


__all__ = [
    "GeckoTelemetry",
    "TelemetryConfig",
    "SpanNames",
    "get_telemetry",
    "configure_telemetry",
    "request_id_var",
    "OTEL_AVAILABLE",
]

# ======================================================================
# 中文用例（可运行示例）
# 说明：
# 1) 若环境未安装 OpenTelemetry，本示例会自动降级（span 返回 None），不会报错
# 2) 若安装了 OpenTelemetry，可配置 exporter 输出到控制台/OTLP/Jaeger 等
# ======================================================================

if __name__ == "__main__":
    import asyncio
    from gecko.core.logging import trace_context, get_context_logger

    log = get_context_logger(__name__)

    # 1) 初始化 telemetry（若未安装 OTel，会降级，但仍可运行）
    telemetry = GeckoTelemetry(
        TelemetryConfig(
            service_name="gecko-demo",
            environment="dev",
            enabled=True,  # 用户意图：想启用
        )
    )
    telemetry.setup()

    # 2) 设置 request_id（可选）
    telemetry.set_request_id("REQ-001")

    # 3) 同步 span 用法（中文演示）
    with trace_context():  # 让 logging 拥有 trace_id/span_id，上报到 span attributes
        with telemetry.span("中文示例-同步操作") as sp:
            if sp:
                sp.set_attribute("demo.key", "demo.value")
            log.info("这是同步 span 内的一条日志（会带 trace_id/span_id）")

    # 4) 异步 span 用法（中文演示）
    async def async_work():
        with trace_context():
            async with telemetry.async_span("中文示例-异步操作") as sp:
                if sp:
                    sp.set_attribute("demo.async", True)
                log.info("这是异步 span 内的一条日志（会带 trace_id/span_id）")
                await asyncio.sleep(0.01)
                telemetry.add_event("异步事件", {"step": 1})

    asyncio.run(async_work())

    # 5) 装饰器用法（中文演示）
    @telemetry.trace_sync("中文示例-装饰器-同步")
    def sync_fn(x: int) -> int:
        log.info(f"执行 sync_fn，入参={x}")
        return x + 1

    @telemetry.trace_async("中文示例-装饰器-异步")
    async def async_fn(x: int) -> int:
        log.info(f"执行 async_fn，入参={x}")
        await asyncio.sleep(0.01)
        return x * 2

    sync_fn(10)
    asyncio.run(async_fn(21))

    # 6) shutdown（中文演示）
    telemetry.shutdown()
    log.info("Telemetry 示例结束")
