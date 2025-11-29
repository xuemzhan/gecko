# tests/core/test_output.py
import pytest

from gecko.core.output import (
    AgentOutput,
    TokenUsage,
    create_text_output,
    create_tool_output,
    create_json_output,
    merge_outputs,
    JsonOutput,
    StreamingOutput,
    StreamingChunk,
)


class TestTokenUsage:
    """TokenUsage 测试"""

    def test_basic_usage(self):
        """测试基本 usage"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_auto_total(self):
        """测试自动计算总数（total_tokens 未显式提供）"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
        )

        # 新实现：total_tokens 默认为 0 且 prompt+completion > 0 时会自动补全
        assert usage.total_tokens == 150

    def test_do_not_override_non_zero_total(self):
        """
        测试当 total_tokens 非 0 且与计算值不一致时：
        - 不会被自动覆盖
        - 只会记录 warning（这里不检查日志，只验证数值未被改写）
        """
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=999,  # 故意填错
        )

        # 按设计：保留 provider 提供的 total_tokens
        assert usage.total_tokens == 999

    def test_zero_all_tokens_kept(self):
        """测试三者都是 0 时保持为 0，不做额外处理"""
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_cost_estimate(self):
        """测试成本估算"""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        cost = usage.get_cost_estimate(
            prompt_price_per_1k=0.01,
            completion_price_per_1k=0.02,
        )

        # 1000 * 0.01/1000 + 500 * 0.02/1000 = 0.01 + 0.01 = 0.02
        assert cost == pytest.approx(0.02)

    def test_str(self):
        """测试字符串表示"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
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
                        "arguments": "{}",
                    },
                }
            ]
        )

        assert output.has_tool_calls()
        assert output.tool_call_count() == 1

    # ===== 验证器测试 =====

    def test_ensure_tool_calls_none(self):
        """测试 tool_calls=None 时自动转换为空列表"""
        output = AgentOutput(tool_calls=None) # type: ignore
        assert output.tool_calls == []

    def test_ensure_tool_calls_single_dict(self):
        """测试 tool_calls 为单个 dict 时自动包装为列表"""
        single_call = {"id": "1", "function": {"name": "search"}}
        output = AgentOutput(tool_calls=single_call) # type: ignore

        assert isinstance(output.tool_calls, list)
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0]["id"] == "1"

    def test_ensure_tool_calls_tuple(self):
        """测试 tool_calls 为 tuple 时自动转换为 list"""
        calls = (
            {"id": "1"},
            {"id": "2"},
        )
        output = AgentOutput(tool_calls=calls)  # type: ignore

        assert isinstance(output.tool_calls, list)
        assert len(output.tool_calls) == 2
        assert output.tool_calls[0]["id"] == "1"
        assert output.tool_calls[1]["id"] == "2"

    def test_ensure_tool_calls_invalid_type(self):
        """测试 tool_calls 为非法类型时强制为空列表"""
        output = AgentOutput(tool_calls="not-a-list") # type: ignore
        assert output.tool_calls == []

    def test_ensure_content(self):
        """测试 content 自动转换"""
        output = AgentOutput(content=None) # type: ignore
        assert output.content == ""

        output2 = AgentOutput(content=123) # type: ignore
        assert output2.content == "123"

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

        # 预览长度：最多 50 + "..." = 53
        assert len(preview) <= 53
        assert preview.endswith("...")

    # ===== 转换方法测试 =====

    def test_to_dict_basic(self):
        """测试转换为字典（基础字段）"""
        output = AgentOutput(
            content="Hello",
            usage=TokenUsage(prompt_tokens=10),
        )

        data = output.to_dict()

        assert data["content"] == "Hello"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 10
        # 未设置 include_raw，默认不包含 raw 字段
        assert "raw" not in data

    def test_to_dict_include_raw_with_structured_raw(self):
        """测试 to_dict 在 include_raw=True 且 raw 为结构化数据时的行为"""
        raw = {"foo": "bar"}
        output = AgentOutput(
            content="Hello",
            raw=raw,
            metadata={"include_raw": True},
        )

        data = output.to_dict()
        assert "raw" in data
        # 新实现会尽量保留原始结构
        assert isinstance(data["raw"], dict)
        assert data["raw"]["foo"] == "bar"

    def test_to_dict_exclude_raw_when_flag_false(self):
        """测试未设置 include_raw 时不会导出 raw 字段"""
        raw = {"foo": "bar"}
        output = AgentOutput(
            content="Hello",
            raw=raw,
            metadata={"include_raw": False},
        )

        data = output.to_dict()
        assert "raw" not in data

    def test_to_message_dict(self):
        """测试转换为消息格式"""
        output = AgentOutput(
            content="Hello",
            tool_calls=[{"id": "1"}],
        )

        msg = output.to_message_dict()

        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["id"] == "1"

    def test_to_message_dict_only_tools(self):
        """测试只有工具调用时 content 为 None 的情况"""
        output = AgentOutput(
            content="",
            tool_calls=[{"id": "1"}],
        )

        msg = output.to_message_dict()
        assert msg["content"] is None
        assert "tool_calls" in msg

    # ===== 格式化测试 =====

    def test_format(self):
        """测试格式化输出"""
        output = AgentOutput(
            content="Hello",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

        formatted = output.format()

        assert "Hello" in formatted
        assert "Token" in formatted
        assert "输入" in formatted
        assert "输出" in formatted

    def test_format_with_metadata(self):
        """测试格式化输出时包含 metadata"""
        output = AgentOutput(
            content="Hello",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            metadata={"model": "glm-4-flash"},
        )

        formatted = output.format(include_metadata=True)

        assert "Hello" in formatted
        assert "Token" in formatted
        assert "model: glm-4-flash" in formatted

    def test_summary(self):
        """测试摘要"""
        output = AgentOutput(
            content="Hello world",
            tool_calls=[{"id": "1"}],
            usage=TokenUsage(total_tokens=100),
        )

        summary = output.summary()

        assert "Hello" in summary
        assert "工具调用: 1" in summary
        assert "100" in summary

    def test_summary_empty(self):
        """测试空输出的摘要"""
        summary = AgentOutput().summary()
        assert summary == "空输出"

    # ===== 统计测试 =====

    def test_get_stats(self):
        """测试统计信息"""
        output = AgentOutput(
            content="Hello",
            tool_calls=[{"function": {"name": "search"}}],
        )

        stats = output.get_stats()

        assert stats["content_length"] == 5
        assert stats["has_content"] is True
        assert stats["tool_call_count"] == 1
        assert "search" in stats["tool_names"]
        assert stats["is_empty"] is False


class TestToolFunctions:
    """工具函数测试"""

    def test_create_text_output(self):
        """测试快速创建文本输出"""
        output = create_text_output(
            "Hello",
            usage=TokenUsage(prompt_tokens=10),
            source="test",
        )

        assert output.content == "Hello"
        assert output.has_usage()
        assert output.metadata["source"] == "test"

    def test_create_tool_output(self):
        """测试快速创建工具输出"""
        output = create_tool_output(
            tool_calls=[{"id": "1"}],
            content="Calling tools...",
        )

        assert output.has_tool_calls()
        assert output.content == "Calling tools..."

    def test_merge_outputs(self):
        """测试合并输出"""
        out1 = AgentOutput(
            content="Part 1",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        out2 = AgentOutput(
            content="Part 2",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
        )

        merged = merge_outputs([out1, out2])

        assert "Part 1" in merged.content
        assert "Part 2" in merged.content
        assert merged.usage.prompt_tokens == 30 # type: ignore
        assert merged.usage.completion_tokens == 15 # type: ignore

    def test_merge_empty_list(self):
        """测试合并空列表"""
        merged = merge_outputs([])
        assert merged.is_empty()
        assert merged.content == ""
        assert merged.tool_calls == []

    def test_merge_single_output(self):
        """测试合并单个输出"""
        output = AgentOutput(content="Hello")
        merged = merge_outputs([output])

        assert merged.content == "Hello"
        # 应直接返回原对象，不做拷贝（不做强制断言，但行为预期）
        assert merged is output


class TestJsonOutput:
    """JsonOutput 测试"""

    def test_create_json_output_factory(self):
        """测试通过工厂方法创建 JsonOutput"""
        data = {"status": "ok", "items": [1, 2, 3]}
        json_output = create_json_output(
            data=data,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10),
            schema_version="v1",
        )

        assert isinstance(json_output, JsonOutput)
        assert json_output.data["status"] == "ok"
        assert json_output.metadata["schema_version"] == "v1"
        assert json_output.usage.total_tokens == 15 # type: ignore

    def test_json_output_to_dict(self):
        """测试 JsonOutput.to_dict"""
        data = {"foo": "bar"}
        json_output = JsonOutput(
            data=data,
            usage=TokenUsage(prompt_tokens=5),
            metadata={"source": "test"},
        )

        d = json_output.to_dict()
        assert d["data"]["foo"] == "bar"
        assert d["metadata"]["source"] == "test"
        assert d["usage"]["prompt_tokens"] == 5

    def test_json_output_summary(self):
        """测试 JsonOutput.summary"""
        data = {"foo": "bar", "baz": [1, 2, 3]}
        json_output = JsonOutput(
            data=data,
            usage=TokenUsage(total_tokens=42),
        )

        summary = json_output.summary()
        assert "JSON:" in summary
        assert "Tokens: 42" in summary

    def test_json_output_to_agent_output(self):
        """测试 JsonOutput 转换为 AgentOutput"""
        data = {"foo": "bar"}
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        json_output = JsonOutput(
            data=data,
            usage=usage,
            metadata={"schema": "v1"},
        )

        agent_output = json_output.to_agent_output(pretty=True)
        assert isinstance(agent_output, AgentOutput)
        assert '"foo": "bar"' in agent_output.content
        assert agent_output.usage.total_tokens == 15 # type: ignore
        assert agent_output.metadata["from"] == "JsonOutput"
        assert agent_output.metadata["schema"] == "v1"


class TestStreamingOutput:
    """StreamingOutput / StreamingChunk 测试"""

    def test_streaming_basic_aggregation(self):
        """测试基础流式内容聚合"""
        stream = StreamingOutput(metadata={"model": "glm-4-flash"})

        stream.append_chunk(
            StreamingChunk(index=1, content_delta=" world", tool_calls_delta=[])
        )
        stream.append_chunk(
            StreamingChunk(index=0, content_delta="Hello", tool_calls_delta=[])
        )

        # iter_contents 应按追加顺序输出
        collected = "".join(stream.iter_contents())
        assert collected == " worldHello"  # 注意：顺序为 append 顺序

        # finalize 时会按 index 排序拼接
        final = stream.finalize()
        assert final.content == "Hello world"
        assert final.metadata["from"] == "StreamingOutput"
        assert final.metadata["model"] == "glm-4-flash"

    def test_streaming_aggregate_tool_calls(self):
        """测试流式工具调用聚合"""
        stream = StreamingOutput()

        stream.append_chunk(
            StreamingChunk(
                index=0,
                content_delta="",
                tool_calls_delta=[{"id": "t1"}],
            )
        )
        stream.append_chunk(
            StreamingChunk(
                index=1,
                content_delta="",
                tool_calls_delta=[{"id": "t2"}],
            )
        )

        final = stream.finalize()
        assert final.tool_call_count() == 2
        ids = [c.get("id") for c in final.tool_calls]
        assert ids == ["t1", "t2"]

    def test_streaming_usage_from_deltas(self):
        """测试从各个 chunk 的 usage_delta 汇总 usage"""
        stream = StreamingOutput()

        stream.append_chunk(
            StreamingChunk(
                index=0,
                content_delta="A",
                usage_delta=TokenUsage(prompt_tokens=5, completion_tokens=3),
            )
        )
        stream.append_chunk(
            StreamingChunk(
                index=1,
                content_delta="B",
                usage_delta=TokenUsage(prompt_tokens=0, completion_tokens=2),
            )
        )

        final = stream.finalize()
        assert final.usage.prompt_tokens == 5 # type: ignore
        assert final.usage.completion_tokens == 5 # type: ignore
        assert final.usage.total_tokens == 10 # type: ignore

    def test_streaming_usage_prefers_overall_usage(self):
        """测试当 StreamingOutput.usage 已存在时优先使用整体 usage"""
        stream = StreamingOutput(
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50)
        )

        stream.append_chunk(
            StreamingChunk(
                index=0,
                content_delta="Hello",
                usage_delta=TokenUsage(prompt_tokens=1, completion_tokens=1),
            )
        )

        final = stream.finalize()
        # 应直接使用整体 usage，而不是累加 delta
        assert final.usage.prompt_tokens == 100 # type: ignore
        assert final.usage.completion_tokens == 50 # type: ignore
        assert final.usage.total_tokens == 150 # type: ignore

    def test_streaming_get_stats(self):
        """测试 StreamingOutput.get_stats"""
        stream = StreamingOutput()

        stream.append_chunk(StreamingChunk(index=0, content_delta="Hel"))
        stream.append_chunk(StreamingChunk(index=1, content_delta="lo"))

        stats = stream.get_stats()
        assert stats["chunk_count"] == 2
        assert stats["total_content_length"] == 5
        assert stats["has_usage"] is False

    def test_streaming_str_repr(self):
        """测试 StreamingOutput 字符串表示"""
        stream = StreamingOutput()
        stream.append_chunk(StreamingChunk(index=0, content_delta="Hi"))
        s = str(stream)
        assert "StreamingOutput" in s
        assert "chunk" in s.lower()
