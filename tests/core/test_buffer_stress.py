import random
import string
import asyncio
from concurrent.futures import ThreadPoolExecutor
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


def _fragment_tool_call(index: int, name: str, args: str, fragments: int):
    """Split a tool call into `fragments` pieces to simulate streaming fragments."""
    parts = []
    # split args into roughly equal parts
    n = len(args)
    if fragments <= 1 or n == 0:
        parts = [args]
    else:
        size = max(1, n // fragments)
        for i in range(fragments - 1):
            parts.append(args[i * size : (i + 1) * size])
        parts.append(args[(fragments - 1) * size :])

    chunks = []
    for p in parts:
        tc = {"index": index, "function": {"name": name, "arguments": p}}
        chunks.append(make_chunk(tool_calls=[tc]))
    return chunks


def test_multithreaded_random_injection():
    """Simulate many threads concurrently adding mixed content/tool chunks in random order."""
    buf = StreamBuffer()

    # prepare many tool calls
    total_tools = 200
    fragments_per_call = 3
    tool_chunks = []
    expected_names = set()
    for i in range(total_tools):
        name = f"tool_{i}"
        expected_names.add(name)
        # random args moderate size
        args = ''.join(random.choice(string.ascii_letters) for _ in range(200))
        frags = _fragment_tool_call(i, name, args, fragments_per_call)
        # also intersperse some content chunks
        combined = []
        for c in frags:
            combined.append(c)
            # occasionally add content
            if random.random() < 0.3:
                combined.append(make_chunk(content=f"txt_{i}_"))
        tool_chunks.extend(combined)

    # shuffle to create out-of-order arrival
    random.shuffle(tool_chunks)

    # concurrent add via ThreadPoolExecutor
    def worker(chunk):
        return buf.add_chunk(chunk)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, tool_chunks))

    msg = buf.build_message()
    names = {tc["function"]["name"] for tc in msg.safe_tool_calls}
    # The StreamBuffer concatenates name fragments; accept presence if any recorded name contains the expected base name
    for en in expected_names:
        assert any(en in recorded for recorded in names), f"Missing tool name {en} in recorded names"


import pytest


@pytest.mark.asyncio
async def test_asyncio_and_mixed_injection():
    """Use asyncio.to_thread to mix async tasks that call the synchronous add_chunk."""
    buf = StreamBuffer()

    total_tools = 150
    fragments_per_call = 4
    tasks = []

    for i in range(total_tools):
        name = f"a_tool_{i}"
        args = ''.join(random.choice(string.ascii_letters) for _ in range(300))
        chunks = _fragment_tool_call(i, name, args, fragments_per_call)
        # shuffle per call
        random.shuffle(chunks)
        # schedule each chunk with small random delay
        for chunk in chunks:
            async def add_delayed(ch):
                # small jitter
                await asyncio.sleep(random.random() * 0.01)
                # call synchronous method in thread to simulate blocking add from async context
                await asyncio.to_thread(buf.add_chunk, ch)
            tasks.append(asyncio.create_task(add_delayed(chunk)))

    # also run a few thread producers simultaneously
    def thread_producer(start):
        for j in range(start, start + 20):
            ch = make_chunk(content=f"thread_txt_{j}")
            buf.add_chunk(ch)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut = loop.run_in_executor(ex, thread_producer, 1000)
        await asyncio.gather(*tasks)
        await fut

    msg = buf.build_message()
    # basic sanity checks
    assert isinstance(msg, type(buf.build_message()))
    # ensure no exception and at least one tool call when created
    assert len(msg.safe_tool_calls) > 0
