# tests/core/test_output.py
import pytest
from gecko.core.output import (
    AgentOutput,
    TokenUsage,
    create_text_output,
    create_tool_output,
    merge_outputs
)


class TestTokenUsage:
    """TokenUsage 测试"""
    
    def test_basic_usage(self):
        """测试基本 usage"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_auto_total(self):
        """测试自动计算总数"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert usage.total_tokens == 150
    
    def test_cost_estimate(self):
        """测试成本估算"""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        cost = usage.get_cost_estimate(
            prompt_price_per_1k=0.01,
            completion_price_per_1k=0.02
        )
        
        # 1000 * 0.01/1000 + 500 * 0.02/1000 = 0.01 + 0.01 = 0.02
        assert cost == pytest.approx(0.02)
    
    def test_str(self):
        """测试字符串表示"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50
        )
        
        str_repr = str(usage)
        assert "100" in str_repr
        assert "50" in str_repr


class TestAgentOutput:
    """AgentOutput 测试"""
    
    # ===== 基础测试 =====
    
    def test_empty_output(self):
        """测试空输出"""
        output = AgentOutput()
        
        assert output.content == ""
        assert output.tool_calls == []
        assert output.is_empty()
        assert not output.has_content()
        assert not output.has_tool_calls()
    
    def test_text_output(self):
        """测试文本输出"""
        output = AgentOutput(content="Hello")
        
        assert output.content == "Hello"
        assert output.has_content()
        assert not output.is_empty()
    
    def test_tool_calls_output(self):
        """测试工具调用输出"""
        output = AgentOutput(
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": "{}"
                    }
                }
            ]
        )
        
        assert output.has_tool_calls()
        assert output.tool_call_count() == 1
    
    # ===== 验证器测试 =====
    
    def test_ensure_tool_calls(self):
        """测试 tool_calls 自动转换"""
        output = AgentOutput(tool_calls=None)
        assert output.tool_calls == []
    
    def test_ensure_content(self):
        """测试 content 自动转换"""
        output = AgentOutput(content=None)
        assert output.content == ""
    
    # ===== 检查方法测试 =====
    
    def test_has_usage(self):
        """测试 usage 检查"""
        output1 = AgentOutput()
        assert not output1.has_usage()
        
        output2 = AgentOutput(usage=TokenUsage(prompt_tokens=10))
        assert output2.has_usage()
    
    def test_bool_conversion(self):
        """测试布尔值转换"""
        assert not bool(AgentOutput())
        assert bool(AgentOutput(content="Hello"))
        assert bool(AgentOutput(tool_calls=[{"id": "1"}]))
    
    # ===== 提取方法测试 =====
    
    def test_get_tool_names(self):
        """测试提取工具名称"""
        output = AgentOutput(
            tool_calls=[
                {"function": {"name": "search"}},
                {"function": {"name": "calculator"}},
            ]
        )
        
        names = output.get_tool_names()
        assert names == ["search", "calculator"]
    
    def test_get_tool_call_by_id(self):
        """测试根据 ID 获取工具调用"""
        call = {"id": "call_123", "function": {"name": "search"}}
        output = AgentOutput(tool_calls=[call])
        
        result = output.get_tool_call_by_id("call_123")
        assert result == call
        
        not_found = output.get_tool_call_by_id("not_exist")
        assert not_found is None
    
    def test_get_text_preview(self):
        """测试文本预览"""
        output = AgentOutput(content="A" * 200)
        preview = output.get_text_preview(50)
        
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")
    
    # ===== 转换方法测试 =====
    
    def test_to_dict(self):
        """测试转换为字典"""
        output = AgentOutput(
            content="Hello",
            usage=TokenUsage(prompt_tokens=10)
        )
        
        data = output.to_dict()
        
        assert data["content"] == "Hello"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 10
    
    def test_to_message_dict(self):
        """测试转换为消息格式"""
        output = AgentOutput(
            content="Hello",
            tool_calls=[{"id": "1"}]
        )
        
        msg = output.to_message_dict()
        
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"
        assert "tool_calls" in msg
    
    # ===== 格式化测试 =====
    
    def test_format(self):
        """测试格式化输出"""
        output = AgentOutput(
            content="Hello",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5)
        )
        
        formatted = output.format()
        
        assert "Hello" in formatted
        assert "Token" in formatted
    
    def test_summary(self):
        """测试摘要"""
        output = AgentOutput(
            content="Hello world",
            tool_calls=[{"id": "1"}],
            usage=TokenUsage(total_tokens=100)
        )
        
        summary = output.summary()
        
        assert "Hello" in summary
        assert "工具调用: 1" in summary
        assert "100" in summary
    
    # ===== 统计测试 =====
    
    def test_get_stats(self):
        """测试统计信息"""
        output = AgentOutput(
            content="Hello",
            tool_calls=[{"function": {"name": "search"}}]
        )
        
        stats = output.get_stats()
        
        assert stats["content_length"] == 5
        assert stats["has_content"] is True
        assert stats["tool_call_count"] == 1
        assert "search" in stats["tool_names"]


class TestToolFunctions:
    """工具函数测试"""
    
    def test_create_text_output(self):
        """测试快速创建文本输出"""
        output = create_text_output(
            "Hello",
            usage=TokenUsage(prompt_tokens=10),
            source="test"
        )
        
        assert output.content == "Hello"
        assert output.has_usage()
        assert output.metadata["source"] == "test"
    
    def test_create_tool_output(self):
        """测试快速创建工具输出"""
        output = create_tool_output(
            tool_calls=[{"id": "1"}],
            content="Calling tools..."
        )
        
        assert output.has_tool_calls()
        assert output.content == "Calling tools..."
    
    def test_merge_outputs(self):
        """测试合并输出"""
        out1 = AgentOutput(
            content="Part 1",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5)
        )
        out2 = AgentOutput(
            content="Part 2",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10)
        )
        
        merged = merge_outputs([out1, out2])
        
        assert "Part 1" in merged.content
        assert "Part 2" in merged.content
        assert merged.usage.prompt_tokens == 30
        assert merged.usage.completion_tokens == 15
    
    def test_merge_empty_list(self):
        """测试合并空列表"""
        merged = merge_outputs([])
        assert merged.is_empty()
    
    def test_merge_single_output(self):
        """测试合并单个输出"""
        output = AgentOutput(content="Hello")
        merged = merge_outputs([output])
        
        assert merged.content == "Hello"