# tests/core/test_tracing.py
"""
请求追踪模块单元测试

测试 trace_id、span_id 生成、上下文管理等功能。
"""
import pytest

from gecko.core.tracing import (
    generate_trace_id,
    generate_span_id,
    get_trace_id,
    get_span_id,
    get_context,
    trace_context,
    set_trace_context,
    clear_trace_context,
)
from gecko.core.logging import trace_id_var, span_id_var, extra_context_var


class TestTracingIDGeneration:
    """测试 ID 生成"""

    def test_generate_trace_id(self):
        """生成 trace_id"""
        tid = generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 16
        assert all(c in "0123456789abcdef" for c in tid)

    def test_generate_span_id(self):
        """生成 span_id"""
        sid = generate_span_id()
        assert isinstance(sid, str)
        assert len(sid) == 8
        assert all(c in "0123456789abcdef" for c in sid)


class TestTracingGetters:
    """测试获取追踪信息"""

    def setup_method(self):
        """每个测试前清空上下文"""
        trace_id_var.set("")
        span_id_var.set("")
        extra_context_var.set({})

    def test_get_trace_id_empty(self):
        """获取未设置的 trace_id"""
        assert get_trace_id() == ""

    def test_get_span_id_empty(self):
        """获取未设置的 span_id"""
        assert get_span_id() == ""

    def test_get_context_empty(self):
        """获取空上下文"""
        ctx = get_context()
        assert ctx == {}

    def test_get_trace_id_with_value(self):
        """获取已设置的 trace_id"""
        trace_id_var.set("abc123")
        assert get_trace_id() == "abc123"

    def test_get_span_id_with_value(self):
        """获取已设置的 span_id"""
        span_id_var.set("def456")
        assert get_span_id() == "def456"

    def test_get_context_complete(self):
        """获取完整上下文"""
        trace_id_var.set("tid123")
        span_id_var.set("sid456")
        extra_context_var.set({"user_id": "user-1", "action": "login"})
        
        ctx = get_context()
        assert ctx["trace_id"] == "tid123"
        assert ctx["span_id"] == "sid456"
        assert ctx["user_id"] == "user-1"
        assert ctx["action"] == "login"


class TestTraceContextManager:
    """测试追踪上下文管理器"""

    def setup_method(self):
        """清空上下文"""
        trace_id_var.set("")
        span_id_var.set("")
        extra_context_var.set({})

    def test_trace_context_basic(self):
        """基础上下文"""
        with trace_context() as ctx:
            assert "trace_id" in ctx
            assert "span_id" in ctx
            assert get_trace_id() == ctx["trace_id"]
            assert get_span_id() == ctx["span_id"]

    def test_trace_context_custom_ids(self):
        """自定义 ID"""
        custom_tid = "custom-trace"
        custom_sid = "custom-span"
        
        with trace_context(trace_id=custom_tid, span_id=custom_sid) as ctx:
            assert ctx["trace_id"] == custom_tid
            assert ctx["span_id"] == custom_sid

    def test_trace_context_with_extra(self):
        """携带额外字段"""
        with trace_context(user_id="user-1", request_id="req-001") as ctx:
            # ctx 仅包含 trace_id 和 span_id
            assert ctx["trace_id"] is not None
            assert ctx["span_id"] is not None
            
            # extra 字段存储在 ContextVar 中，通过 get_context() 返回
            full_ctx = get_context()
            assert full_ctx["user_id"] == "user-1"
            assert full_ctx["request_id"] == "req-001"

    def test_trace_context_restore_after_exit(self):
        """退出后应该恢复上下文"""
        trace_id_var.set("outer-tid")
        span_id_var.set("outer-sid")
        
        with trace_context() as ctx:
            inner_tid = get_trace_id()
            inner_sid = get_span_id()
            assert inner_tid != ""
            assert inner_sid != ""
        
        # 退出后应该恢复外层
        assert get_trace_id() == "outer-tid"
        assert get_span_id() == "outer-sid"

    def test_trace_context_reuse_trace_id(self):
        """复用 trace_id"""
        with trace_context() as ctx1:
            outer_tid = ctx1["trace_id"]
            
            with trace_context() as ctx2:
                # 内层应该复用外层 trace_id
                assert ctx2["trace_id"] == outer_tid

    def test_trace_context_new_span_id(self):
        """每层产生新的 span_id"""
        with trace_context() as ctx1:
            sid1 = ctx1["span_id"]
            
            with trace_context() as ctx2:
                sid2 = ctx2["span_id"]
                assert sid1 != sid2


class TestSetTraceContext:
    """测试直接设置追踪上下文"""

    def setup_method(self):
        """清空上下文"""
        trace_id_var.set("")
        span_id_var.set("")
        extra_context_var.set({})

    def test_set_trace_id(self):
        """设置 trace_id"""
        set_trace_context(trace_id="my-trace")
        assert get_trace_id() == "my-trace"

    def test_set_span_id(self):
        """设置 span_id"""
        set_trace_context(span_id="my-span")
        assert get_span_id() == "my-span"

    def test_set_extra_fields(self):
        """设置额外字段"""
        set_trace_context(user_id="user-1", request_id="req-001")
        ctx = get_context()
        assert ctx["user_id"] == "user-1"
        assert ctx["request_id"] == "req-001"

    def test_set_merge_extra_fields(self):
        """合并额外字段"""
        set_trace_context(user_id="user-1")
        set_trace_context(request_id="req-001")
        
        ctx = get_context()
        assert ctx["user_id"] == "user-1"
        assert ctx["request_id"] == "req-001"

    def test_set_override_fields(self):
        """覆盖字段"""
        set_trace_context(user_id="user-1")
        set_trace_context(user_id="user-2")
        
        assert get_context()["user_id"] == "user-2"


class TestClearTraceContext:
    """测试清除追踪上下文"""

    def test_clear_all_context(self):
        """清除所有上下文"""
        set_trace_context(trace_id="tid", span_id="sid", user_id="user-1")
        clear_trace_context()
        
        assert get_trace_id() == ""
        assert get_span_id() == ""
        assert get_context() == {}

    def test_clear_empty_context(self):
        """清除空上下文（不应该报错）"""
        clear_trace_context()
        assert get_context() == {}


class TestTracingIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """完整工作流"""
        # 1. 生成新的追踪
        trace_id = generate_trace_id()
        set_trace_context(trace_id=trace_id, user_id="user-123")
        
        # 2. 验证上下文
        ctx = get_context()
        assert ctx["trace_id"] == trace_id
        assert ctx["user_id"] == "user-123"
        
        # 3. 使用 with 管理器添加新的 span
        with trace_context() as span_ctx:
            # 应该继承 trace_id
            assert get_trace_id() == trace_id
            # 但产生新的 span_id
            assert get_span_id() == span_ctx["span_id"]
        
        # 4. 清理
        clear_trace_context()
        assert get_context() == {}

    def test_multi_level_trace(self):
        """多层级追踪"""
        with trace_context(operation="outer") as ctx1:
            tid1 = ctx1["trace_id"]
            
            with trace_context(operation="middle") as ctx2:
                assert get_trace_id() == tid1  # 复用 trace_id
                
                with trace_context(operation="inner") as ctx3:
                    assert get_trace_id() == tid1  # 继续复用
            
            # 回到外层
            assert get_trace_id() == tid1
