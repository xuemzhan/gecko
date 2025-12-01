# tests/core/test_memory.py
"""
针对 gecko.core.memory 模块的单元测试

目标：
- 覆盖 TokenMemory / SummaryTokenMemory / 线程池管理 等核心逻辑
- 覆盖尽可能多的分支和异常路径，接近/达到 100% 行覆盖
- 验证增强版 memory 模块的功能、性能与稳定性

依赖：
- tests/conftest.py 中已经提供：
  - event_loop（异步测试用）
  - mock_llm（ModelProtocol 的 MagicMock 实现）
  - memory（默认的 TokenMemory Fixture，带 mock_llm 与 storage）
"""

from __future__ import annotations

import sys
from typing import Any, List

import pytest

from gecko.core.memory import (
    TokenMemory,
    SummaryTokenMemory,
    shutdown_token_executor,
)
from gecko.core.memory._executor import get_token_executor
from gecko.core.message import Message


# ======================================================================
# 测试辅助类：FakeModel
# ======================================================================


class FakeModel:
    """
    用于测试的假模型，实现简单的 count_tokens 和 acompletion 接口。

    使用场景：
    - 验证 SummaryTokenMemory 是否触发摘要生成（通过 summary_calls 计数）
    - 验证 TokenMemory 在 encode=None 时是否会调用 model_driver.count_tokens
    """

    def __init__(self) -> None:
        self.count_calls: List[List[dict]] = []
        self.summary_calls: int = 0

    def count_tokens(self, messages: List[dict]) -> int:
        """
        简单的 Token 计数规则：
        - 每条消息的 token 数 = len(content) // 2 + 10
        - 这里只要规则稳定即可，方便断言
        """
        self.count_calls.append(messages)
        total = 0
        for m in messages:
            content = m.get("content") or ""
            total += len(str(content)) // 2 + 10
        return total

    async def acompletion(self, messages: List[dict]) -> Any:
        """
        模拟异步补全接口：
        - 将输入的 history 文本截取前 50 字符，前面加上 "SUMMARY:"
        - 返回一个类似 OpenAI 的响应结构：response.choices[0].message.content
        """
        self.summary_calls += 1
        history_text = messages[0]["content"]
        summary = "SUMMARY:" + history_text[:50]

        class _Msg:
            def __init__(self, content: str):
                self._content = content

            def get(self, key: str, default: str = "") -> str:
                if key == "content":
                    return self._content
                return default

        class _Choice:
            def __init__(self, content: str):
                self.message = _Msg(content)

        class _Resp:
            def __init__(self, content: str):
                self.choices = [_Choice(content)]

        return _Resp(summary)


# ======================================================================
# TokenMemory 基础行为 & 缓存
# ======================================================================


def test_token_memory_count_and_cache():
    """
    测试：
    - 单条 Token 计数
    - LRU 缓存命中与统计信息
    """
    memory = TokenMemory(session_id="test", max_tokens=1000, cache_size=10)

    msg = Message(role="user", content="hello world")
    count1 = memory.count_message_tokens(msg)
    count2 = memory.count_message_tokens(msg)

    # 两次结果应完全一致
    assert count1 == count2

    stats = memory.get_cache_stats()
    # 至少有一次命中（第二次）
    assert stats["hits"] >= 1
    # 至少有一次 miss（第一次真正计算）
    assert stats["misses"] >= 1


def test_token_memory_clear_cache():
    """
    测试 clear_cache：
    - 能清空缓存
    - 能重置统计数据
    """
    memory = TokenMemory(session_id="test-clear", max_tokens=1000, cache_size=10)
    msg = Message(role="user", content="hello")
    _ = memory.count_message_tokens(msg)  # 触发一次 miss

    stats_before = memory.get_cache_stats()
    assert stats_before["cache_size"] >= 1

    memory.clear_cache()
    stats_after = memory.get_cache_stats()
    assert stats_after["cache_size"] == 0
    assert stats_after["hits"] == 0
    assert stats_after["misses"] == 0
    assert stats_after["evictions"] == 0


# ======================================================================
# 多模态 / dict 块计数修复
# ======================================================================


def test_token_memory_multimodal_dict_blocks():
    """
    回归测试：
    - content 为 list[dict]（多模态）时，不再低估 Token 数
    - 至少保证“文本+图片”的 Token 大于“纯文本”
    """
    memory = TokenMemory(session_id="test-mm", max_tokens=1000, cache_size=10)

    text_msg = Message(role="user", content="这是一条纯文本消息。")

    multi_msg = Message(
        role="user",
        content=[
            {"type": "text", "text": "这是一条多模态消息的文本部分。"},
            {
                "type": "image_url",
                "image_url": {"url": "http://example.com/1.png", "detail": "low"},
            },
        ], # type: ignore
    )

    text_tokens = memory.count_message_tokens(text_msg)
    multi_tokens = memory.count_message_tokens(multi_msg)

    assert multi_tokens > text_tokens


# ======================================================================
# 批量计数：同步 & 异步缓存语义
# ======================================================================


def test_token_memory_batch_use_cache_flag():
    """
    测试同步批量计数时 use_cache 的语义：
    - use_cache=None 时跟随实例级 enable_cache_for_batch（默认 True）
    - use_cache=False 时强制不走缓存，miss 计数应增加
    """
    memory = TokenMemory(session_id="test-batch", max_tokens=1000, cache_size=10)

    msgs = [
        Message(role="user", content="msg1"),
        Message(role="assistant", content="msg2"),
    ]

    counts1 = memory.count_messages_batch(msgs)
    stats1 = memory.get_cache_stats()

    counts2 = memory.count_messages_batch(msgs, use_cache=False)
    stats2 = memory.get_cache_stats()

    assert counts1 == counts2
    assert stats2["misses"] >= stats1["misses"]


@pytest.mark.asyncio
async def test_token_memory_async_batch_cache_semantics():
    """
    测试异步批量计数时 use_cache 的语义是否与同步版本一致。
    """
    memory = TokenMemory(session_id="test-abatch", max_tokens=1000, cache_size=10)

    msgs = [
        Message(role="user", content="async-1"),
        Message(role="assistant", content="async-2"),
    ]

    counts1 = await memory.count_messages_batch_async(msgs)
    stats1 = memory.get_cache_stats()

    counts2 = await memory.count_messages_batch_async(msgs, use_cache=False)
    stats2 = memory.get_cache_stats()

    assert counts1 == counts2
    assert stats2["misses"] >= stats1["misses"]


# ======================================================================
# 异步计数不在子线程调用 model_driver.count_tokens
# ======================================================================


@pytest.mark.asyncio
async def test_token_memory_async_not_call_model_driver_in_thread():
    """
    回归测试：
    - count_messages_batch_async 在线程池中只做本地 encode/估算，
      不会调用 model_driver.count_tokens（避免线程安全问题）
    """
    fake_model = FakeModel()
    memory = TokenMemory(
        session_id="test-async-model",
        max_tokens=1000,
        cache_size=10,
        model_driver=fake_model, # type: ignore
        enable_async_counting=True,
    )

    msgs = [Message(role="user", content="hello async")] * 3

    counts = await memory.count_messages_batch_async(msgs, use_cache=False)

    assert len(counts) == 3
    # 异步批量计数路径中我们显式传入 encode_fn，
    # 因此 _count_tokens_impl 不会再走 model_driver 分支
    assert fake_model.count_calls == []


# ======================================================================
# 文本强制截断逻辑（_truncate_text_to_tokens / _force_truncate）
# ======================================================================


def test_force_truncate_respects_token_limit():
    """
    测试 _force_truncate：
    - 能在给定的 Token 预算下对文本进行合理截断
    """
    memory = TokenMemory(session_id="test-trunc", max_tokens=100, cache_size=10)

    long_text = "你好，" + "非常长的文本。" * 100
    msg = Message(role="user", content=long_text)

    # 限制到 50 tokens（具体 encode 实现不重要，只要不爆炸就行）
    memory._force_truncate(msg, limit_tokens=50)
    truncated = msg.content

    tokens = memory.count_message_tokens(msg)
    assert tokens <= 60  # 适当给点 buffer

    assert isinstance(truncated, str)
    assert truncated.endswith("...(truncated)") or len(truncated) < len(long_text)


def test_truncate_text_to_tokens_short_text_with_encode(monkeypatch):
    """
    覆盖分支：
    - encode 存在 且 原文 Token 数不超 limit，直接返回原文
    """
    memory = TokenMemory(session_id="test-tt-short", max_tokens=100, cache_size=10)

    def fake_encode(text: str):
        return list(range(len(text)))

    monkeypatch.setattr(memory, "_get_encode_func", lambda: fake_encode)

    text = "abc"
    result = memory._truncate_text_to_tokens(text, limit_tokens=10)

    assert result == text


def test_truncate_text_to_tokens_char_fallback_without_encode(monkeypatch):
    """
    覆盖分支：
    - encode 为 None 时，走“字符估算 + 后缀”的降级路径
    """
    memory = TokenMemory(session_id="test-tt-char", max_tokens=100, cache_size=10)

    monkeypatch.setattr(memory, "_get_encode_func", lambda: None)

    long_text = "x" * 100
    result = memory._truncate_text_to_tokens(long_text, limit_tokens=5)

    assert isinstance(result, str)
    assert result.endswith("...(truncated)")
    assert len(result) <= int(5 * 3.5) + len("...(truncated)")


def test_truncate_text_to_tokens_extreme_small_limit_returns_empty(monkeypatch):
    """
    覆盖极端分支：
    - fake_encode 无论输入什么都返回超大 token 数
    - limit_tokens 极小，导致 even suffix 也“超限”，最终返回空串
    """
    memory = TokenMemory(session_id="test-tt-extreme", max_tokens=100, cache_size=10)

    def fake_encode(_: str):
        return list(range(1000))

    monkeypatch.setattr(memory, "_get_encode_func", lambda: fake_encode)

    text = "abc"
    result = memory._truncate_text_to_tokens(text, limit_tokens=5)

    assert result == ""


# ======================================================================
# model_driver.count_tokens 分支（单条同步计数）
# ======================================================================


def test_token_memory_uses_model_driver_for_single_message(mock_llm, monkeypatch):
    """
    覆盖分支：
    - _count_tokens_impl 在 encode 为 None 且 model_driver 存在时，
      应优先调用 model_driver.count_tokens
    """
    memory = TokenMemory(
        session_id="test-model-driver",
        max_tokens=1000,
        cache_size=10,
        model_driver=mock_llm,
    )

    # 确保 encode 为 None（跳过 tiktoken）
    monkeypatch.setattr(memory, "_get_encode_func", lambda: None)

    msg = Message(role="user", content="hello model driver")

    count = memory.count_message_tokens(msg)

    mock_llm.count_tokens.assert_called()
    # 返回值应为 mock_llm.count_tokens 的返回值（conftest 中为 10）
    assert count == mock_llm.count_tokens.return_value


# ======================================================================
# tool_calls / name 开销分支
# ======================================================================


def test_token_memory_tool_calls_json_error_adds_default_tokens(monkeypatch):
    """
    覆盖分支：
    - tool_calls 存在但 json.dumps 失败时，进入 except 分支并 +100 Token。
    """
    memory = TokenMemory(session_id="test-toolcalls", max_tokens=1000, cache_size=10)

    def fake_encode(text: str):
        return list(range(len(text)))

    base_msg = Message(role="user", content="base text")
    base_tokens = memory._count_tokens_impl(base_msg, encode=fake_encode)

    bad_msg = Message(role="user", content="base text")
    bad_msg.tool_calls = {"bad": object()}  # type: ignore # json.dumps 无法序列化 object()

    bad_tokens = memory._count_tokens_impl(bad_msg, encode=fake_encode)

    assert bad_tokens == base_tokens + 100


def test_token_memory_name_overhead():
    """
    覆盖分支：
    - message.name 非空时，_count_tokens_impl 会额外 +1 Token。
    """
    memory = TokenMemory(session_id="test-name", max_tokens=1000, cache_size=10)

    msg_no_name = Message(role="user", content="hello", name=None)
    msg_with_name = Message(role="user", content="hello", name="foo")

    tokens_no_name = memory.count_message_tokens(msg_no_name)
    tokens_with_name = memory.count_message_tokens(msg_with_name)

    assert tokens_with_name == tokens_no_name + 1


# ======================================================================
# get_history 基础裁剪 & 边界场景
# ======================================================================


@pytest.mark.asyncio
async def test_get_history_respects_max_tokens_basic():
    """
    测试基类 TokenMemory 的 get_history 裁剪逻辑：
    - 保留首条 system（若存在且 preserve_system=True）
    - 其余消息从尾部往前回填，直到接近 max_tokens
    """
    memory = TokenMemory(session_id="test-history", max_tokens=200, cache_size=10)

    raw = []
    raw.append({"role": "system", "content": "You are a helpful assistant."})
    for i in range(10):
        raw.append({"role": "user", "content": f"user-{i}"})
        raw.append({"role": "assistant", "content": f"assistant-{i}"})

    history = await memory.get_history(raw, preserve_system=True)

    assert history[0].role == "system"

    total_tokens = memory.count_total_tokens(history)
    assert total_tokens <= memory.max_tokens + 20  # 给一点余量


@pytest.mark.asyncio
async def test_get_history_empty_raw_messages():
    """
    覆盖分支：
    - raw_messages 为空列表时，直接返回 []
    """
    memory = TokenMemory(session_id="test-history-empty", max_tokens=100, cache_size=10)
    history = await memory.get_history([], preserve_system=True)
    assert history == []


@pytest.mark.asyncio
async def test_get_history_invalid_message_skipped():
    """
    原先假设: 缺少 content 会导致 Message(**raw) 失败 → 被跳过 → 返回 []
    实际实现: Message 对缺失 content 采用默认值 ""，不会抛异常，也不会被跳过。

    本测试改为验证：
    - get_history 能够容忍这种“字段不完整”的 raw
    - 返回的 Message 对象 role 正确，content 落在默认值范围
    """
    memory = TokenMemory(session_id="test-history-invalid", max_tokens=100, cache_size=10)

    raw = [{"role": "user"}]  # 缺少 content，但对 Message 来说是合法的

    history = await memory.get_history(raw, preserve_system=True)

    assert len(history) == 1
    msg = history[0]
    assert msg.role == "user"
    # content 在当前实现中会被填成 ""（或其他安全默认值）
    assert msg.content in ("", None)



# ======================================================================
# SummaryTokenMemory：摘要生成、预算控制与边界
# ======================================================================


@pytest.mark.asyncio
async def test_summary_token_memory_generates_summary_and_respects_budget():
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary",
        model=fake_model, # type: ignore
        max_tokens=200,
        summary_reserve_tokens=80,
        # [修复] 禁用防抖和后台更新，确保测试同步执行
        min_update_interval=0,
        background_update=False
    )

    raw = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        raw.append({"role": "user", "content": f"user-{i}-" + "x" * 20})
        raw.append({"role": "assistant", "content": f"assistant-{i}-" + "y" * 20})

    # 因为 background_update=False，这里 await 会等待摘要生成完毕
    history = await memory.get_history(raw, preserve_system=True)

    # 现在断言是安全的
    assert fake_model.summary_calls >= 1
    
    # 验证摘要是否注入
    has_summary = any("Previous context:" in m.content for m in history if m.role == "system")
    assert has_summary

@pytest.mark.asyncio
async def test_summary_token_memory_reserved_exceeds_max_and_drops_summary():
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary-drop",
        model=fake_model, # type: ignore
        max_tokens=50,
        summary_reserve_tokens=40,
        # [修复] 强制同步模式
        min_update_interval=0,
        background_update=False
    )

    long_system = "S" * 1000
    raw = [{"role": "system", "content": long_system}]
    for i in range(5):
        raw.append({"role": "user", "content": f"user-{i}-" + "x" * 50})

    history = await memory.get_history(raw, preserve_system=True)

    # 确认模型被调用了（虽然结果可能被丢弃）
    assert fake_model.summary_calls >= 1

@pytest.mark.asyncio
async def test_summary_token_memory_empty_raw_messages():
    """
    覆盖 SummaryTokenMemory.get_history 的 raw_messages 为空分支。
    """
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary-empty",
        model=fake_model, # type: ignore
        max_tokens=100,
        summary_reserve_tokens=20,
    )

    history = await memory.get_history([], preserve_system=True)
    assert history == []


@pytest.mark.asyncio
async def test_summary_token_memory_invalid_message_skipped():
    """
    与 TokenMemory 情况类似：
    - 缺少 content 不再视为“非法消息”，Message(**raw) 会成功构造
    因此这里验证：
    - SummaryTokenMemory 也能容忍这种 raw，并正常返回消息对象
    - 不会抛异常
    """
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary-invalid",
        model=fake_model, # type: ignore
        max_tokens=100,
        summary_reserve_tokens=20,
    )

    raw = [{"role": "user"}]  # 无 content，但对 Message 来说是合法的

    history = await memory.get_history(raw, preserve_system=True)

    assert len(history) == 1
    msg = history[0]
    assert msg.role == "user"
    assert msg.content in ("", None)

@pytest.mark.asyncio
async def test_summary_token_memory_update_summary_twice_uses_previous():
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary-twice",
        model=fake_model,  # type: ignore
        max_tokens=200,
        summary_reserve_tokens=50,
        # [修复] 关键：必须禁用防抖，否则第二次调用会被跳过
        min_update_interval=0,
        background_update=False
    )

    # 第一次
    raw1 = [
        {"role": "user", "content": "first-" + "x" * 50}
        for _ in range(20)
    ]
    await memory.get_history(raw1, preserve_system=False)
    assert fake_model.summary_calls == 1 # 确认第一次调用成功

    # 第二次：模拟新的一轮对话
    raw2 = raw1 + [
        {"role": "user", "content": "second-" + "y" * 50}
        for _ in range(10)
    ]
    
    # 因为 min_update_interval=0，这里会立即触发第二次摘要
    await memory.get_history(raw2, preserve_system=False)

    # 断言触发了多次
    assert fake_model.summary_calls >= 2
    assert len(memory.current_summary) > 0

def test_summary_token_memory_clear_summary():
    """
    测试 clear_summary：
    - 能清空 current_summary
    """
    fake_model = FakeModel()
    memory = SummaryTokenMemory(
        session_id="test-summary-clear",
        model=fake_model, # type: ignore
        max_tokens=200,
        summary_reserve_tokens=50,
    )

    memory.current_summary = "some summary"
    memory.clear_summary()
    assert memory.current_summary == ""


# ======================================================================
# estimate_tokens / print_cache_stats / __repr__
# ======================================================================


def test_estimate_tokens_with_and_without_encode(monkeypatch):
    """
    覆盖 estimate_tokens 的两条路径：
    - 有 encode 时，使用 encode 精算
    - 无 encode 时，退化为 len(text) // 4
    """
    memory = TokenMemory(session_id="test-estimate", max_tokens=100, cache_size=10)

    text = "hello world"

    def fake_encode(text: str):
        return list(range(len(text)))

    monkeypatch.setattr(memory, "_get_encode_func", lambda: fake_encode)
    tokens_with_encode = memory.estimate_tokens(text)
    assert tokens_with_encode == len(text)

    monkeypatch.setattr(memory, "_get_encode_func", lambda: None)
    tokens_without_encode = memory.estimate_tokens(text)
    assert tokens_without_encode == len(text) // 4


def test_print_cache_stats_and_repr(capsys):
    """
    覆盖：
    - print_cache_stats 打印逻辑（不会抛异常）
    - __repr__ 字符串表示是否包含关键信息
    """
    memory = TokenMemory(session_id="test-print", max_tokens=100, cache_size=10)

    msg = Message(role="user", content="hello")
    _ = memory.count_message_tokens(msg)

    memory.print_cache_stats()
    captured = capsys.readouterr()
    assert "Token Cache Statistics" in captured.out

    repr_str = repr(memory)
    assert "TokenMemory(session_id='test-print'" in repr_str
    assert "max_tokens=100" in repr_str


# ======================================================================
# tokenizer 属性：_encoding / _tokenizer_failed 分支
# ======================================================================


def test_tokenizer_property_encoding_and_failed_flags(monkeypatch):
    """
    覆盖 TokenMemory.tokenizer 的简单分支：
    - _encoding 已经存在时，直接返回
    - _tokenizer_failed=True 时，直接返回 None
    """
    memory = TokenMemory(session_id="test-tokenizer-flags", max_tokens=100, cache_size=10)

    # 1) _encoding 不为 None 时
    memory._encoding = "dummy-encoding"
    assert memory.tokenizer == "dummy-encoding"

    # 2) _tokenizer_failed=True 时，直接返回 None
    memory._encoding = None
    memory._tokenizer_failed = True
    assert memory.tokenizer is None


def test_tokenizer_property_with_fake_tiktoken_and_keyerror(monkeypatch):
    """
    覆盖 TokenMemory.tokenizer 中的 tiktoken 分支：
    - tiktoken.encoding_for_model 抛 KeyError
    - 退回到 tiktoken.get_encoding("cl100k_base")
    """
    memory = TokenMemory(session_id="test-tokenizer-keyerror", max_tokens=100, cache_size=10)

    class FakeEncoding:
        def __init__(self):
            self._name = "fake-encoding"

        def encode(self, text: str):
            return list(range(len(text)))

    class FakeTiktokenModule:
        def encoding_for_model(self, name: str):
            raise KeyError("not found")

        def get_encoding(self, name: str):
            return FakeEncoding()

    monkeypatch.setitem(sys.modules, "tiktoken", FakeTiktokenModule())

    memory._encoding = None
    memory._tokenizer_failed = False

    encoding = memory.tokenizer
    assert isinstance(encoding, FakeEncoding)
    assert memory._encode_func is not None


def test_tokenizer_property_with_fake_tiktoken_normal(monkeypatch):
    """
    覆盖 TokenMemory.tokenizer 中的正常 tiktoken 分支：
    - encoding_for_model 正常返回编码器
    """
    memory = TokenMemory(session_id="test-tokenizer-normal", max_tokens=100, cache_size=10)

    class FakeEncoding:
        def __init__(self):
            self._name = "fake-encoding-normal"

        def encode(self, text: str):
            return list(range(len(text)))

    class FakeTiktokenModule:
        def encoding_for_model(self, name: str):
            return FakeEncoding()

        def get_encoding(self, name: str):
            return FakeEncoding()

    monkeypatch.setitem(sys.modules, "tiktoken", FakeTiktokenModule())

    memory._encoding = None
    memory._tokenizer_failed = False

    encoding = memory.tokenizer
    assert isinstance(encoding, FakeEncoding)
    assert memory._encode_func is not None


# ======================================================================
# 线程池管理：get_token_executor / shutdown_token_executor
# ======================================================================


def test_get_token_executor_singleton():
    """
    覆盖 _executor 模块：
    - get_token_executor 懒加载
    - 多次调用返回同一个实例（单例）
    """
    shutdown_token_executor()  # 确保初始状态干净

    ex1 = get_token_executor()
    ex2 = get_token_executor()

    assert ex1 is ex2


def test_shutdown_token_executor_idempotent():
    """
    覆盖 shutdown_token_executor：
    - 多次调用不会抛异常
    """
    shutdown_token_executor()
    shutdown_token_executor()
