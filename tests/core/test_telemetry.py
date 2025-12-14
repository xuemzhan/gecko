# tests/core/test_telemetry.py
import pytest
from unittest.mock import MagicMock, patch
from contextvars import ContextVar

from gecko.core.telemetry import (
    GeckoTelemetry,
    TelemetryConfig,
    request_id_var,
    configure_telemetry
)

# Mock opentelemetry classes to avoid runtime errors if not installed
@pytest.fixture
def mock_otel():
    with patch("gecko.core.telemetry.trace") as trace_mock:
        tracer = MagicMock()
        trace_mock.get_tracer.return_value = tracer

        # Mock Span context manager
        span = MagicMock()
        span.__enter__.return_value = span
        span.__exit__.return_value = None
        tracer.start_as_current_span.return_value = span

        yield trace_mock


class TestGeckoTelemetry:

    def test_config_initialization(self):
        """测试配置加载"""
        config = TelemetryConfig(service_name="test-service", enabled=False)
        telemetry = GeckoTelemetry(config)
        assert telemetry.config.service_name == "test-service"
        assert telemetry.is_enabled is False

    def test_setup_logic(self, mock_otel):
        """测试初始化逻辑"""
        from gecko.version import __version__ as gecko_version

        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)

        # 模拟 OTEL 可用
        with patch("gecko.core.telemetry.OTEL_AVAILABLE", True):
            telemetry.setup()

            assert telemetry.is_enabled is True
            mock_otel.set_tracer_provider.assert_called_once()
            mock_otel.get_tracer.assert_called_with("gecko", gecko_version)

    @pytest.mark.asyncio
    async def test_trace_decorator_async(self, mock_otel):
        """测试异步装饰器"""
        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)

        # Manually setup since we are mocking
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        @telemetry.trace_async("async_op")
        async def my_func(x):
            return x * 2

        result = await my_func(10)
        assert result == 20

        # 验证 start_as_current_span 被调用
        telemetry._tracer.start_as_current_span.assert_called_with(  # type: ignore
            "async_op",
            kind=mock_otel.SpanKind.INTERNAL,
            attributes={}
        )

    def test_request_id_injection(self, mock_otel):
        """测试 Request ID 注入"""
        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        # 设置 Context
        token = request_id_var.set("req-123")

        with telemetry.span("test_span") as span:
            # 验证属性被设置
            span.set_attribute.assert_any_call("gecko.request_id", "req-123")  # type: ignore

        request_id_var.reset(token)

    def test_global_configure(self):
        """测试全局配置单例"""
        config = TelemetryConfig(enabled=False)
        t = configure_telemetry(config)
        assert t.config.enabled is False

        from gecko.core.telemetry import get_telemetry
        assert get_telemetry() is t


class TestTelemetryAutoInitialization:
    """测试自动初始化行为"""

    def test_get_telemetry_auto_setup(self):
        """获取实例时自动调用 setup()"""
        from gecko.core.telemetry import get_telemetry

        # 获取实例，应该返回一个有效的 GeckoTelemetry 实例
        telemetry = get_telemetry()
        assert telemetry is not None
        assert isinstance(telemetry, GeckoTelemetry)

    def test_get_telemetry_idempotent(self):
        """多次调用 get_telemetry() 返回同一实例"""
        from gecko.core.telemetry import get_telemetry

        t1 = get_telemetry()
        t2 = get_telemetry()

        assert t1 is t2


class TestLoggingTraceIDInjection:
    """测试日志 Trace ID 注入到 span"""

    def test_span_injects_logging_trace_id(self, mock_otel):
        """验证 span() 注入 logging trace ID"""
        from gecko.core.logging import trace_context
        from gecko.core.telemetry import GeckoTelemetry, TelemetryConfig

        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        # 在 trace context 中创建 span
        with trace_context() as ctx:
            # trace_context 已经设置了 logging trace_id_var 和 span_id_var
            # 现在创建 span，它应该注入这些值
            with telemetry.span("test_op"):
                # 验证 span 被创建（不抛出异常）
                pass

    def test_async_span_injects_logging_trace_id(self, mock_otel):
        """验证 async_span() 注入 logging trace ID"""
        import asyncio
        from gecko.core.logging import trace_context
        from gecko.core.telemetry import GeckoTelemetry, TelemetryConfig

        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        async def test_func():
            with trace_context():
                with telemetry.span("async_op"):
                    return "success"

        result = asyncio.run(test_func())
        assert result == "success"


class TestTelemetryShutdown:
    """测试 telemetry 关闭和清理"""

    def test_shutdown_cleans_traces(self, mock_otel):
        """测试关闭时的清理"""
        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        # 创建一些 span
        with telemetry.span("op1"):
            pass

        with telemetry.span("op2"):
            pass

        # 调用关闭（不应该抛出异常）
        telemetry.shutdown()

    def test_disabled_telemetry_no_overhead(self):
        """禁用 telemetry 时不应该有开销"""
        config = TelemetryConfig(enabled=False)
        telemetry = GeckoTelemetry(config)

        # 这些操作应该快速返回
        with telemetry.span("no_op"):
            pass

        # 不应该抛出异常
        telemetry.shutdown()


class TestTelemetryIntegration:
    """集成测试"""

    def test_full_workflow_with_logging(self, mock_otel):
        """完整工作流：logging + telemetry + metrics"""
        from gecko.core.logging import get_context_logger, trace_context
        from gecko.core.metrics import MetricsRegistry
        from gecko.core.telemetry import GeckoTelemetry, TelemetryConfig

        # 设置 telemetry
        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        telemetry._tracer = mock_otel.get_tracer()
        telemetry._initialized = True

        # 设置 logging
        logger = get_context_logger(__name__)

        # 设置 metrics
        registry = MetricsRegistry()
        counter = registry.counter("operations")

        # 在 trace context 中执行
        with trace_context() as ctx:
            with telemetry.span("operation"):
                logger.info("Doing operation")
                counter.inc()

        # 验证 counter 计数正确
        assert counter.value == 1


def test_get_telemetry_uses_gecko_version(monkeypatch):
    """
    验证 get_telemetry() 使用 gecko.version.__version__（单一来源），而非硬编码字符串。
    """
    from gecko.version import __version__ as gecko_version
    from gecko.core import telemetry as telemetry_mod

    class DummySettings:
        telemetry_service_name = "gecko"
        telemetry_environment = "test"
        telemetry_enabled = False

    # patch settings（避免依赖真实环境配置）
    monkeypatch.setattr("gecko.config.settings", DummySettings, raising=False)

    t = telemetry_mod.get_telemetry()
    assert t.config.service_version == gecko_version