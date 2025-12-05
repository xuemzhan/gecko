from gecko.core.engine.buffer import StreamBuffer
from gecko.core.protocols.model import StreamChunk


def make_chunk(tool_calls=None):
    choice = {"delta": {}}
    if tool_calls is not None:
        choice["delta"]["tool_calls"] = tool_calls
    return StreamChunk(choices=[choice])


def test_sparse_index_skipped():
    buf = StreamBuffer()
    # first valid small index
    tc0 = {"index": 0, "function": {"name": "a", "arguments": "{}"}}
    buf.add_chunk(make_chunk(tool_calls=[tc0]))
    # now send wildly sparse index
    tc_big = {"index": 10000, "function": {"name": "b", "arguments": "{}"}}
    buf.add_chunk(make_chunk(tool_calls=[tc_big]))

    msg = buf.build_message()
    # only index 0 should be present
    assert len(msg.safe_tool_calls) == 1
    assert msg.safe_tool_calls[0]["function"]["name"] == "a"


def test_out_of_order_acceptance_within_gap():
    buf = StreamBuffer()
    # set initial max index to 0
    tc0 = {"index": 0, "function": {"name": "a", "arguments": "{}"}}
    buf.add_chunk(make_chunk(tool_calls=[tc0]))
    # index within allowed gap (<=100) should be accepted
    tc_next = {"index": 50, "function": {"name": "c", "arguments": "{}"}}
    buf.add_chunk(make_chunk(tool_calls=[tc_next]))
    msg = buf.build_message()
    assert any(tc["function"]["name"] == "c" for tc in msg.safe_tool_calls)
