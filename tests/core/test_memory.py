# tests/core/test_memory.py
import pytest
from gecko.core.message import Message
from gecko.core.memory import TokenMemory

def test_token_caching():
    mem = TokenMemory(session_id="test", cache_size=2) # 小缓存测试 LRU
    msg1 = Message.user("hello world")
    msg2 = Message.user("foo bar")
    msg3 = Message.user("baz qux")
    
    # 1. 第一次计算 (Miss) -> Misses: 1
    c1 = mem.count_message_tokens(msg1)
    assert mem._cache_misses == 1
    
    # 2. 第二次计算 (Hit) -> Misses: 1
    c1_again = mem.count_message_tokens(msg1)
    assert mem._cache_hits == 1
    assert c1 == c1_again
    
    # 3. 填满缓存 (msg2) -> Misses: 2
    mem.count_message_tokens(msg2)
    assert len(mem._token_cache) == 2
    
    # 4. 挤出 msg1 (LRU) -> Misses: 3
    # msg3 也是新内容，会导致一次 miss
    mem.count_message_tokens(msg3)
    assert len(mem._token_cache) == 2
    
    # 5. 再次访问 msg1 (应该 Miss) -> Misses: 4
    # 因为 msg1 在第4步被淘汰了，所以这里重新计算会导致 miss 增加
    mem.count_message_tokens(msg1)
    
    # 修复：累积的 miss 次数应该是 4 (msg1 + msg2 + msg3 + msg1_retry)
    assert mem._cache_misses == 4