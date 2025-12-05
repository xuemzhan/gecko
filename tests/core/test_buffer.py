import json
from gecko.core.engine.buffer import StreamBuffer
from gecko.core.protocols.model import StreamChunk


def make_chunk(content=None, tool_calls=None):
    choice = {}
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choice["delta"] = delta
    return StreamChunk(choices=[choice])


def test_content_aggregation():
    buf = StreamBuffer()
    buf.add_chunk(make_chunk(content="Hello "))
    buf.add_chunk(make_chunk(content="World"))

    msg = buf.build_message()
    assert msg.get_text_content() == "Hello World"


def test_tool_calls_order_and_cleaning():
    buf = StreamBuffer()
    # out-of-order: idx 1 then idx 0
    raw_args0 = "```json\n{\"q\": \"hello\",}\n```"
    raw_args1 = "'{\"x\': 1, }'"

    tc0 = {"index": 0, "function": {"name": "search", "arguments": raw_args0}}
    tc1 = {"index": 1, "function": {"name": "parse", "arguments": raw_args1}}

    buf.add_chunk(make_chunk(tool_calls=[tc1]))
    buf.add_chunk(make_chunk(tool_calls=[tc0]))

    msg = buf.build_message()
    tool_calls = msg.safe_tool_calls

    # should be sorted by index 0,1
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "search"
    assert tool_calls[1]["function"]["name"] == "parse"

    # cleaned args should not contain markdown fences
    assert "```" not in tool_calls[0]["function"]["arguments"]
    # and should be valid JSON or fallback {}
    try:
        json.loads(tool_calls[0]["function"]["arguments"])
    except Exception:
        assert tool_calls[0]["function"]["arguments"] == "{}"


def test_content_truncation_small_limit():
    buf = StreamBuffer()
    # lower the limit for test
    buf._max_content_chars = 10
    buf.add_chunk(make_chunk(content="abcdefghijklmnopqrstuvwxyz"))
    msg = buf.build_message()
    assert len(msg.get_text_content()) <= 10


def test_tool_arguments_truncation_trigger():
    buf = StreamBuffer()
    # construct a very large argument >100000 to force truncation branch
    big_arg = "a" * 100_005
    tc = {"index": 0, "function": {"name": "bigtool", "arguments": big_arg}}
    buf.add_chunk(make_chunk(tool_calls=[tc]))
    msg = buf.build_message()
    args = msg.safe_tool_calls[0]["function"]["arguments"]
    # should have been truncated to 100000 chars
    assert len(args) == 100_000 or args == "{}"
