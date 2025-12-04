# tests/core/test_logging.py
"""
日志系统单元测试

测试结构化日志、追踪上下文注入、ContextLogger 等功能。
"""
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from gecko.core.logging import (
    get_logger,
    get_context_logger,
    trace_context,
    generate_trace_id,
    generate_span_id,
    setup_logging,
    trace_id_var,
    span_id_var,
    extra_context_var,
    ContextLogger,
)


class TestTraceIDGeneration:
    """测试 ID 生成函数"""

    def test_trace_id_format(self):
        """Trace ID 应为 16 位十六进制字符串"""
        tid = generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 16
        assert all(c in "0123456789abcdef" for c in tid)

    def test_span_id_format(self):
        """Span ID 应为 8 位十六进制字符串"""
        sid = generate_span_id()
        assert isinstance(sid, str)
        assert len(sid) == 8
        assert all(c in "0123456789abcdef" for c in sid)

    def test_ids_are_unique(self):
        """生成的 ID 应该不重复"""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

        sids = {generate_span_id() for _ in range(100)}
        assert len(sids) == 100


class TestTraceContext:
    """测试追踪上下文管理器"""

    def test_trace_context_basic(self):
        """基础上下文设置"""
        # 清除之前的上下文
        trace_id_var.set("")
        span_id_var.set("")

        with trace_context() as ctx:
            assert "trace_id" in ctx
            assert "span_id" in ctx
            assert trace_id_var.get() == ctx["trace_id"]
            assert span_id_var.get() == ctx["span_id"]

        # 上下文退出后应该恢复
        assert trace_id_var.get() == ""
        assert span_id_var.get() == ""

    def test_trace_context_with_custom_ids(self):
        """使用自定义 ID"""
        custom_tid = "custom-trace-123"
        custom_sid = "custom-sp1"

        with trace_context(trace_id=custom_tid, span_id=custom_sid) as ctx:
            assert ctx["trace_id"] == custom_tid
            assert ctx["span_id"] == custom_sid
            assert trace_id_var.get() == custom_tid
            assert span_id_var.get() == custom_sid

    def test_trace_context_with_extra_fields(self):
        """带额外字段的上下文"""
        with trace_context(user_id="user-123", action="login") as ctx:
            # ctx 仅包含 trace_id 和 span_id，extra 字段存储在 extra_context_var
            assert ctx["trace_id"] is not None
            assert ctx["span_id"] is not None
            
            # 验证 extra_context_var 包含传入的字段
            extra = extra_context_var.get()
            assert extra["user_id"] == "user-123"
            assert extra["action"] == "login"

    def test_trace_context_nesting(self):
        """嵌套上下文"""
        with trace_context(user_id="outer") as ctx1:
            outer_tid = ctx1["trace_id"]
            
            with trace_context(user_id="inner") as ctx2:
                # 内层应该继承外层 trace_id（如果未指定）
                # 但 span_id 应该是新的
                assert ctx2["trace_id"] == outer_tid
                inner_sid = ctx2["span_id"]
                assert inner_sid != ctx1["span_id"]
                
                # 验证当前上下文
                assert trace_id_var.get() == outer_tid
                assert span_id_var.get() == inner_sid
            
            # 退出内层后恢复外层 span_id
            assert span_id_var.get() == ctx1["span_id"]

    def test_trace_context_reuse_existing(self):
        """复用现有 trace_id"""
        with trace_context() as ctx1:
            tid1 = ctx1["trace_id"]
            
            # 新上下文如果不指定 trace_id，应复用当前的
            with trace_context() as ctx2:
                assert ctx2["trace_id"] == tid1


class TestContextLogger:
    """测试带上下文的日志记录器"""

    def test_context_logger_creation(self):
        """创建 ContextLogger"""
        base_logger = get_logger(__name__)
        ctx_logger = ContextLogger(base_logger)
        assert ctx_logger._logger is base_logger

    def test_context_logger_enrich(self):
        """测试 _enrich 方法"""
        base_logger = MagicMock()
        ctx_logger = ContextLogger(base_logger)
        
        with trace_context(user_id="test-user"):
            enriched = ctx_logger._enrich(action="login", extra="value")
            
            assert "trace_id" in enriched
            assert "span_id" in enriched
            assert enriched["user_id"] == "test-user"
            assert enriched["action"] == "login"
            assert enriched["extra"] == "value"

    def test_context_logger_info(self):
        """测试 info 级别日志"""
        base_logger = MagicMock()
        base_logger.info = MagicMock()
        ctx_logger = ContextLogger(base_logger)
        ctx_logger._is_structlog = False
        
        with trace_context(user_id="user-1"):
            ctx_logger.info("Test message", request_id="req-123")
            
            # 验证 logging 层收到了正确的参数
            base_logger.info.assert_called_once()
            call_args = base_logger.info.call_args
            assert call_args[0][0] == "Test message"
            
            # 验证 extra 字段包含上下文信息
            extra = call_args[1]["extra"]
            assert extra["user_id"] == "user-1"
            assert extra["request_id"] == "req-123"

    def test_context_logger_exception(self):
        """测试异常日志记录"""
        base_logger = MagicMock()
        base_logger.exception = MagicMock()
        ctx_logger = ContextLogger(base_logger)
        ctx_logger._is_structlog = False
        
        ctx_logger.exception("Error occurred", error_code="500")
        
        base_logger.exception.assert_called_once()
        call_args = base_logger.exception.call_args
        assert call_args[0][0] == "Error occurred"
        assert call_args[1]["exc_info"] is True
        
        extra = call_args[1]["extra"]
        assert extra["error_code"] == "500"

    def test_context_logger_all_levels(self):
        """测试所有日志级别"""
        base_logger = MagicMock()
        ctx_logger = ContextLogger(base_logger)
        ctx_logger._is_structlog = False
        
        base_logger.debug = MagicMock()
        base_logger.info = MagicMock()
        base_logger.warning = MagicMock()
        base_logger.error = MagicMock()
        
        ctx_logger.debug("Debug msg")
        ctx_logger.info("Info msg")
        ctx_logger.warning("Warning msg")
        ctx_logger.error("Error msg")
        
        assert base_logger.debug.called
        assert base_logger.info.called
        assert base_logger.warning.called
        assert base_logger.error.called


class TestLoggingSetup:
    """测试日志系统初始化"""

    def test_setup_logging_default(self):
        """测试默认初始化"""
        setup_logging(force=True)
        # 应该不抛异常
        assert True

    def test_setup_logging_with_level(self):
        """测试指定日志级别"""
        # 即使 structlog 可用，setup_logging 也应该被调用而不抛出异常
        setup_logging(level="DEBUG", force=True)
        # 验证初始化标志被设置
        from gecko.core.logging import _initialized
        assert _initialized is True

    def test_get_logger(self):
        """获取日志记录器"""
        logger = get_logger(__name__)
        assert logger is not None

    def test_get_context_logger(self):
        """获取带上下文的日志记录器"""
        ctx_logger = get_context_logger(__name__)
        assert isinstance(ctx_logger, ContextLogger)


class TestLoggingIntegration:
    """集成测试"""

    def test_full_trace_workflow(self):
        """完整的追踪工作流"""
        ctx_logger = get_context_logger(__name__)
        
        with trace_context(request_id="req-001", user_id="user-123") as ctx:
            # 这里应该能获得所有上下文信息
            extra = extra_context_var.get()
            assert extra["request_id"] == "req-001"
            assert extra["user_id"] == "user-123"

    def test_context_isolation_across_traces(self):
        """不同追踪上下文之间的隔离"""
        with trace_context(user_id="user-1") as ctx1:
            tid1 = trace_id_var.get()
            assert extra_context_var.get()["user_id"] == "user-1"
            
        # 上下文应该完全清除
        assert trace_id_var.get() == ""
        assert extra_context_var.get() == {}
