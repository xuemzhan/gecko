import pytest
import threading
from unittest.mock import MagicMock

from gecko.core.engine.buffer import StreamBuffer
from gecko.core.message import Message


def create_chunk(content=None, tool_calls=None):
    """
    Module-level helper for creating mock stream chunks.
    
    This function is exported for use by other test modules.
    """
    chunk = MagicMock()
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk.delta = delta
    return chunk


class TestStreamBuffer:
    def test_initialization(self):
        buffer = StreamBuffer()
        assert buffer._content_len == 0
        assert len(buffer.content_parts) == 0
        assert len(buffer.tool_calls_map) == 0
        assert buffer._max_tool_index == -1

    def test_initialization_with_limits(self):
        buffer = StreamBuffer(
            max_content_chars=50000,
            max_argument_chars=25000,
            max_tool_index=500
        )
        assert buffer._max_content_chars == 50000
        assert buffer._max_argument_chars == 25000
        assert buffer._max_tool_index_limit == 500

    def test_add_chunk_with_content(self):
        buffer = StreamBuffer()
        
        chunk = MagicMock()
        chunk.delta = {"content": "Hello"}
        
        result = buffer.add_chunk(chunk)
        
        assert result == "Hello"
        assert buffer.get_current_content() == "Hello"

    def test_add_chunk_with_multiple_contents(self):
        buffer = StreamBuffer()
        
        chunks = [
            create_chunk(content="Hello"),
            create_chunk(content=" "),
            create_chunk(content="World"),
        ]
        
        for chunk in chunks:
            buffer.add_chunk(chunk)
        
        assert buffer.get_current_content() == "Hello World"

    def test_add_chunk_with_tool_call(self):
        buffer = StreamBuffer()
        
        chunk = create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_123",
            "function": {
                "name": "test_tool",
                "arguments": '{"param": "value"}'
            }
        }])
        
        buffer.add_chunk(chunk)
        
        assert 0 in buffer.tool_calls_map
        assert buffer.tool_calls_map[0]["id"] == "call_123"
        assert buffer.tool_calls_map[0]["function"]["name"] == "test_tool"

    def test_add_chunk_incremental_tool_arguments(self):
        buffer = StreamBuffer()
        
        chunks = [
            create_chunk(tool_calls=[{
                "index": 0,
                "id": "call_1",
                "function": {"name": "tool", "arguments": '{"ke'}
            }]),
            create_chunk(tool_calls=[{
                "index": 0,
                "function": {"arguments": 'y": '}
            }]),
            create_chunk(tool_calls=[{
                "index": 0,
                "function": {"arguments": '"val"}'}
            }]),
        ]
        
        for chunk in chunks:
            buffer.add_chunk(chunk)
        
        assert buffer.tool_calls_map[0]["function"]["arguments"] == '{"key": "val"}'

    def test_add_chunk_multiple_tool_calls(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool1", "arguments": "{}"}
        }]))
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 1,
            "id": "call_2",
            "function": {"name": "tool2", "arguments": "{}"}
        }]))
        
        assert len(buffer.tool_calls_map) == 2
        assert buffer.get_tool_call_count() == 2

    def test_add_chunk_content_limit(self):
        buffer = StreamBuffer(max_content_chars=10)
        
        buffer.add_chunk(create_chunk(content="12345"))
        result = buffer.add_chunk(create_chunk(content="67890ABCDE"))
        
        assert buffer._content_len == 10
        assert result == "67890"

    def test_add_chunk_argument_limit(self):
        buffer = StreamBuffer(max_argument_chars=20)
        
        long_args = '{"key": "' + "A" * 100 + '"}'
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": long_args}
        }]))
        
        assert buffer._args_len_map[0] == 20

    def test_add_chunk_negative_index(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": -1,
            "id": "call_1",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        assert len(buffer.tool_calls_map) == 0

    def test_add_chunk_index_exceeds_limit(self):
        buffer = StreamBuffer(max_tool_index=10)
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 100,
            "id": "call_1",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        assert len(buffer.tool_calls_map) == 0

    def test_add_chunk_tool_id_change_warning(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_2",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        assert buffer.tool_calls_map[0]["id"] == "call_2"

    def test_extract_delta_from_dict(self):
        buffer = StreamBuffer()
        
        chunk = {"delta": {"content": "test"}}
        delta = buffer._extract_delta(chunk)
        
        assert delta == {"content": "test"}

    def test_extract_delta_from_object(self):
        buffer = StreamBuffer()
        
        chunk = MagicMock()
        chunk.delta = {"content": "test"}
        delta = buffer._extract_delta(chunk)
        
        assert delta == {"content": "test"}

    def test_extract_delta_from_choices(self):
        buffer = StreamBuffer()
        
        chunk = MagicMock()
        chunk.delta = None
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = {"content": "test"}
        
        delta = buffer._extract_delta(chunk)
        assert delta == {"content": "test"}

    def test_extract_delta_from_dict_choices(self):
        buffer = StreamBuffer()
        
        chunk = {
            "choices": [
                {"delta": {"content": "test"}}
            ]
        }
        
        delta = buffer._extract_delta(chunk)
        assert delta == {"content": "test"}

    def test_extract_delta_returns_none(self):
        buffer = StreamBuffer()
        
        chunk = {"invalid": "structure"}
        delta = buffer._extract_delta(chunk)
        
        assert delta is None

    def test_build_message_with_content_only(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(content="Hello World"))
        
        message = buffer.build_message()
        
        assert message.role == "assistant"
        assert message.content == "Hello World"
        assert message.tool_calls is None

    def test_build_message_with_tool_calls(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "NYC"}'
            }
        }]))
        
        message = buffer.build_message()
        
        assert message.role == "assistant"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["function"]["name"] == "get_weather"

    def test_build_message_skips_tool_without_name(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "", "arguments": "{}"}
        }]))
        
        message = buffer.build_message()
        
        assert len(message.tool_calls or []) == 0

    def test_build_message_with_empty_arguments(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": ""}
        }]))
        
        message = buffer.build_message()
        
        assert message.tool_calls[0]["function"]["arguments"] == "{}"

    def test_clean_arguments_valid_json(self):
        buffer = StreamBuffer()
        
        valid = '{"key": "value"}'
        result = buffer._clean_arguments(valid)
        
        assert result == valid

    def test_clean_arguments_empty_string(self):
        buffer = StreamBuffer()
        
        result = buffer._clean_arguments("")
        assert result == "{}"

    def test_clean_arguments_markdown_wrapped(self):
        buffer = StreamBuffer()
        
        markdown = '```json\n{"key": "value"}\n```'
        result = buffer._clean_arguments(markdown)
        
        assert result == '{"key": "value"}'

    def test_clean_arguments_quoted_json(self):
        buffer = StreamBuffer()
        
        quoted = '"{"key": "value"}"'
        result = buffer._clean_arguments(quoted)
        
        assert result == '{"key": "value"}'

    def test_clean_arguments_trailing_comma_object(self):
        buffer = StreamBuffer()
        
        trailing = '{"key": "value",}'
        result = buffer._clean_arguments(trailing)
        
        assert result == '{"key": "value"}'

    def test_clean_arguments_trailing_comma_array(self):
        buffer = StreamBuffer()
        
        trailing = '["a", "b",]'
        result = buffer._clean_arguments(trailing)
        
        assert result == '["a", "b"]'

    def test_clean_arguments_single_quotes(self):
        buffer = StreamBuffer()
        
        single_quotes = "{'key': 'value'}"
        result = buffer._clean_arguments(single_quotes)
        
        assert '"key"' in result
        assert '"value"' in result

    def test_clean_arguments_invalid_returns_empty_object(self):
        buffer = StreamBuffer()
        
        invalid = '{invalid json syntax!!!'
        result = buffer._clean_arguments(invalid)
        
        assert result == "{}"

    def test_safe_literal_eval_dict(self):
        buffer = StreamBuffer()
        
        result = buffer._safe_literal_eval("{'a': 1, 'b': 2}")
        assert result == {'a': 1, 'b': 2}

    def test_safe_literal_eval_list(self):
        buffer = StreamBuffer()
        
        result = buffer._safe_literal_eval("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_safe_literal_eval_nested(self):
        buffer = StreamBuffer()
        
        result = buffer._safe_literal_eval("{'a': [1, 2], 'b': {'c': 3}}")
        assert result == {'a': [1, 2], 'b': {'c': 3}}

    def test_safe_literal_eval_negative_numbers(self):
        buffer = StreamBuffer()
        
        result = buffer._safe_literal_eval("{'num': -42}")
        assert result == {'num': -42}

    def test_safe_literal_eval_rejects_function_calls(self):
        buffer = StreamBuffer()
        
        with pytest.raises(ValueError):
            buffer._safe_literal_eval("os.system('ls')")

    def test_safe_literal_eval_rejects_imports(self):
        buffer = StreamBuffer()
        
        with pytest.raises(ValueError):
            buffer._safe_literal_eval("__import__('os')")

    def test_safe_literal_eval_rejects_invalid_types(self):
        buffer = StreamBuffer()
        
        with pytest.raises(ValueError):
            buffer._safe_literal_eval("object()")

    def test_reset(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(content="test"))
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        buffer.reset()
        
        assert len(buffer.content_parts) == 0
        assert len(buffer.tool_calls_map) == 0
        assert buffer._content_len == 0
        assert buffer._max_tool_index == -1

    def test_get_current_content(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(content="Hello"))
        buffer.add_chunk(create_chunk(content=" World"))
        
        assert buffer.get_current_content() == "Hello World"

    def test_get_tool_call_count(self):
        buffer = StreamBuffer()
        
        for i in range(3):
            buffer.add_chunk(create_chunk(tool_calls=[{
                "index": i,
                "id": f"call_{i}",
                "function": {"name": "tool", "arguments": "{}"}
            }]))
        
        assert buffer.get_tool_call_count() == 3

    def test_repr(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(content="test"))
        
        repr_str = repr(buffer)
        assert "StreamBuffer" in repr_str
        assert "content_length" in repr_str

    def test_thread_safety(self):
        buffer = StreamBuffer()
        
        def worker(thread_id):
            for i in range(10):
                buffer.add_chunk(create_chunk(content=f"T{thread_id}-{i}"))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        content = buffer.get_current_content()
        assert len(content) > 0

    def test_tool_calls_sorted_by_index(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 2,
            "id": "call_2",
            "function": {"name": "tool2", "arguments": "{}"}
        }]))
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_0",
            "function": {"name": "tool0", "arguments": "{}"}
        }]))
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 1,
            "id": "call_1",
            "function": {"name": "tool1", "arguments": "{}"}
        }]))
        
        message = buffer.build_message()
        
        assert message.tool_calls[0]["function"]["name"] == "tool0"
        assert message.tool_calls[1]["function"]["name"] == "tool1"
        assert message.tool_calls[2]["function"]["name"] == "tool2"

    def test_large_content_handling(self):
        buffer = StreamBuffer(max_content_chars=1000)
        
        large_content = "A" * 500
        buffer.add_chunk(create_chunk(content=large_content))
        buffer.add_chunk(create_chunk(content=large_content))
        
        assert buffer._content_len == 1000

    def test_large_tool_arguments_handling(self):
        buffer = StreamBuffer(max_argument_chars=100)
        
        large_args = '{"data": "' + "X" * 200 + '"}'
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": large_args}
        }]))
        
        assert buffer._args_len_map[0] == 100

    def test_mixed_content_and_tools(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(content="Thinking..."))
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        buffer.add_chunk(create_chunk(content=" Done."))
        
        message = buffer.build_message()
        
        assert message.content == "Thinking... Done."
        assert len(message.tool_calls) == 1

    def test_tool_call_without_id(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        message = buffer.build_message()
        
        assert message.tool_calls[0]["id"] == ""

    def test_tool_call_type_preservation(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        message = buffer.build_message()
        
        assert message.tool_calls[0]["type"] == "function"

    def test_malformed_tool_call_structure(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": None
        }]))
        
        assert 0 in buffer.tool_calls_map

    def test_clean_arguments_preserves_unicode(self):
        buffer = StreamBuffer()
        
        unicode_json = '{"text": "你好世界"}'
        result = buffer._clean_arguments(unicode_json)
        
        assert "你好世界" in result

    def test_add_chunk_gap_warning(self):
        buffer = StreamBuffer()
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 0,
            "id": "call_0",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        buffer.add_chunk(create_chunk(tool_calls=[{
            "index": 600,
            "id": "call_600",
            "function": {"name": "tool", "arguments": "{}"}
        }]))
        
        assert buffer._max_tool_index == 600


# Export for use by other test modules
__all__ = ["create_chunk", "TestStreamBuffer"]