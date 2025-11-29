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
        config = TelemetryConfig(enabled=True)
        telemetry = GeckoTelemetry(config)
        
        # 模拟 OTEL 可用
        with patch("gecko.core.telemetry.OTEL_AVAILABLE", True):
            telemetry.setup()
            
            assert telemetry.is_enabled is True
            mock_otel.set_tracer_provider.assert_called_once()
            mock_otel.get_tracer.assert_called_with("gecko", "0.2.0")

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
        telemetry._tracer.start_as_current_span.assert_called_with( # type: ignore
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
            span.set_attribute.assert_any_call("gecko.request_id", "req-123") # type: ignore
            
        request_id_var.reset(token)

    def test_global_configure(self):
        """测试全局配置单例"""
        config = TelemetryConfig(enabled=False)
        t = configure_telemetry(config)
        assert t.config.enabled is False
        
        from gecko.core.telemetry import get_telemetry
        assert get_telemetry() is t