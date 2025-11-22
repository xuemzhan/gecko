# tests/core/test_memory.py
import pytest
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from unittest.mock import AsyncMock, MagicMock
from gecko.core.memory import SummaryTokenMemory


class TestTokenMemory:
    """TokenMemory 单元测试"""
    
    @pytest.fixture
    def memory(self):
        """创建测试用 TokenMemory"""
        return TokenMemory(
            session_id="test_session",
            storage=None,
            max_tokens=1000,
            model_name="gpt-3.5-turbo",
            cache_size=100
        )
    
    # ===== 基础功能测试 =====
    
    def test_initialization(self, memory):
        """测试初始化"""
        assert memory.session_id == "test_session"
        assert memory.max_tokens == 1000
        assert memory.cache_size == 100
    
    def test_invalid_max_tokens(self):
        """测试无效的 max_tokens"""
        with pytest.raises(ValueError):
            TokenMemory(
                session_id="test",
                max_tokens=0  # 无效
            )
    
    # ===== Token 计数测试 =====
    
    def test_count_simple_message(self, memory):
        """测试简单消息计数"""
        msg = Message.user("Hello")
        count = memory.count_message_tokens(msg)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_message_with_tool_calls(self, memory):
        """测试带 tool_calls 的消息"""
        msg = Message(
            role="assistant",
            content="I'll search for that",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        )
        
        count = memory.count_message_tokens(msg)
        assert count > len("I'll search for that")
    
    def test_batch_counting(self, memory):
        """测试批量计数"""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi"),
            Message.user("How are you?"),
        ]
        
        counts = memory.count_messages_batch(messages)
        
        assert len(counts) == 3
        assert all(c > 0 for c in counts)
    
    # ===== 缓存测试 =====
    
    def test_cache_hit(self, memory):
        """测试缓存命中"""
        msg = Message.user("Test message")
        
        # 首次计数（缓存未命中）
        count1 = memory.count_message_tokens(msg)
        stats1 = memory.get_cache_stats()
        
        # 第二次计数（缓存命中）
        count2 = memory.count_message_tokens(msg)
        stats2 = memory.get_cache_stats()
        
        assert count1 == count2
        assert stats2['hits'] > stats1['hits']
    
    def test_cache_clear(self, memory):
        """测试缓存清空"""
        msg = Message.user("Test")
        memory.count_message_tokens(msg)
        
        # 清空前
        stats_before = memory.get_cache_stats()
        assert stats_before['cache_size'] > 0
        
        # 清空
        memory.clear_cache()
        
        # 清空后
        stats_after = memory.get_cache_stats()
        assert stats_after['cache_size'] == 0
        assert stats_after['hits'] == 0
    
    def test_cache_eviction(self):
        """测试缓存淘汰"""
        # 创建小容量缓存
        memory = TokenMemory(
            session_id="test",
            cache_size=2  # 只能缓存 2 条
        )
        
        # 添加 3 条消息
        msg1 = Message.user("Message 1")
        msg2 = Message.user("Message 2")
        msg3 = Message.user("Message 3")
        
        memory.count_message_tokens(msg1)
        memory.count_message_tokens(msg2)
        memory.count_message_tokens(msg3)  # 触发淘汰
        
        stats = memory.get_cache_stats()
        assert stats['cache_size'] == 2
        assert stats['evictions'] > 0
    
    # ===== 历史加载测试 =====
    
    @pytest.mark.asyncio
    async def test_get_history_empty(self, memory):
        """测试空历史"""
        history = await memory.get_history([])
        assert history == []
    
    @pytest.mark.asyncio
    async def test_get_history_with_system(self, memory):
        """测试保留 system 消息"""
        raw_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        
        history = await memory.get_history(raw_messages)
        
        assert len(history) > 0
        assert history[0].role == "system"
    
    @pytest.mark.asyncio
    async def test_get_history_truncation(self):
        """测试历史截断"""
        memory = TokenMemory(
            session_id="test",
            max_tokens=100  # 很小的限制
        )
        
        # 创建大量消息
        raw_messages = []
        for i in range(20):
            raw_messages.append({
                "role": "user",
                "content": f"This is a long message number {i} with some content"
            })
        
        history = await memory.get_history(raw_messages)
        
        # 应该被截断
        assert len(history) < len(raw_messages)
        
        # 总 tokens 应该不超过限制
        total_tokens = sum(memory.count_message_tokens(m) for m in history)
        assert total_tokens <= memory.max_tokens
    
    @pytest.mark.asyncio
    async def test_get_history_invalid_messages(self, memory):
        """测试处理无效消息"""
        raw_messages = [
            {"role": "user", "content": "Valid"},
            {"invalid": "message"},  # 无效
            {"role": "assistant", "content": "Also valid"},
        ]
        
        history = await memory.get_history(raw_messages)
        
        # 应该跳过无效消息
        assert len(history) == 2
    
    # ===== 边缘情况测试 =====
    
    def test_very_long_message(self, memory):
        """测试超长消息"""
        long_text = "x" * 50000  # 50k 字符
        msg = Message.user(long_text)
        
        count = memory.count_message_tokens(msg)
        assert count > 0
    
    @pytest.mark.asyncio
    async def test_message_length_limit(self):
        """测试消息长度限制"""
        memory = TokenMemory(
            session_id="test",
            max_message_length=100  # 限制 100 字符
        )
        
        raw_messages = [
            {"role": "user", "content": "x" * 200}  # 超长
        ]
        
        history = await memory.get_history(raw_messages)
        
        # 应该被截断
        assert len(history) == 1
        assert len(history[0].content) <= 100
    
    # ===== 工具方法测试 =====
    
    def test_estimate_tokens(self, memory):
        """测试快速估算"""
        text = "This is a test"
        tokens = memory.estimate_tokens(text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_repr(self, memory):
        """测试字符串表示"""
        repr_str = repr(memory)
        
        assert "TokenMemory" in repr_str
        assert "test_session" in repr_str

class TestSummaryTokenMemory:
    """SummaryTokenMemory 测试"""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.acompletion = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message={"content": "Summary of old messages"})]
        ))
        return model

    @pytest.fixture
    def summary_memory(self, mock_model):
        return SummaryTokenMemory(
            session_id="summary_sess",
            model=mock_model,
            max_tokens=100  # 设置很小的阈值以触发摘要
        )

    @pytest.mark.asyncio
    async def test_get_history_triggers_summary(self, summary_memory, mock_model):
        """测试历史记录过长触发摘要"""
        # 构造 3 条消息，假设每条约 20 tokens (User: Msg X)
        # max_tokens=100, reserved=500 (defaut logic might need adjustment for test)
        # Wait, SummaryTokenMemory implementation reserves 500 tokens for system/summary.
        # If max_tokens is 100, available for history is negative?
        # 我们需要调整 memory 的 max_tokens 或者 mock count_message_tokens
        
        summary_memory.max_tokens = 1000
        # Mock tokenizer to return large length
        summary_memory.count_message_tokens = MagicMock(return_value=300)
        
        raw_messages = [
            {"role": "user", "content": "Msg 1"}, # Should be summarized
            {"role": "assistant", "content": "Msg 2"}, # Should be summarized
            {"role": "user", "content": "Msg 3"}, # Kept
        ]
        
        # Logic: 
        # Msg 3 (300) + Reserved (500) = 800 < 1000. OK.
        # Msg 2 (300) + 800 = 1100 > 1000. Msg 2 & Msg 1 go to summary.
        
        history = await summary_memory.get_history(raw_messages, preserve_system=False)
        
        # 验证
        assert len(history) == 2 # 1 Summary System Msg + 1 Recent Msg (Msg 3)
        assert history[0].role == "system"
        assert "Summary of old messages" in history[0].content
        assert history[1].content == "Msg 3"
        
        # 验证模型调用
        assert mock_model.acompletion.called
        call_args = mock_model.acompletion.call_args[0][0]
        prompt = call_args[0]["content"]
        assert "Msg 1" in prompt
        assert "Msg 2" in prompt

    @pytest.mark.asyncio
    async def test_get_history_no_summary_needed(self, summary_memory, mock_model):
        """测试不需要摘要的情况"""
        summary_memory.max_tokens = 2000
        summary_memory.count_message_tokens = MagicMock(return_value=10)
        
        raw_messages = [{"role": "user", "content": "Short"}]
        
        history = await summary_memory.get_history(raw_messages)
        
        assert len(history) == 1
        assert not mock_model.acompletion.called