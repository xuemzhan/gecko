# examples/memory_demo.py
import asyncio
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.plugins.storage.sqlite import SQLiteSessionStorage


async def main():
    # 1. 创建存储
    storage = SQLiteSessionStorage("sqlite://./test.db")
    
    # 2. 创建 TokenMemory
    memory = TokenMemory(
        session_id="user_123",
        storage=storage,
        max_tokens=4000,
        model_name="gpt-4",
        cache_size=1000,
        max_message_length=10000
    )
    
    print(memory)
    
    # 3. 计算单条消息
    msg = Message.user("What's the weather today?")
    tokens = memory.count_message_tokens(msg)
    print(f"\n单条消息 tokens: {tokens}")
    
    # 4. 批量计算
    messages = [
        Message.user("Hello"),
        Message.assistant("Hi there!"),
        Message.user("How are you?"),
    ]
    
    # 使用缓存
    counts1 = memory.count_messages_batch(messages, use_cache=True)
    print(f"\n批量计数（使用缓存）: {counts1}")
    
    # 不使用缓存（更快，但不缓存结果）
    counts2 = memory.count_messages_batch(messages, use_cache=False)
    print(f"批量计数（不使用缓存）: {counts2}")
    
    # 5. 测试缓存性能
    print("\n=== 缓存性能测试 ===")
    
    # 首次计数（缓存未命中）
    for _ in range(3):
        memory.count_message_tokens(msg)
    
    # 查看统计
    memory.print_cache_stats()
    
    # 6. 历史加载测试
    print("\n=== 历史加载测试 ===")
    
    # 构造大量历史消息
    raw_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    
    for i in range(100):
        raw_messages.append({
            "role": "user",
            "content": f"Question {i}: Can you help me?"
        })
        raw_messages.append({
            "role": "assistant",
            "content": f"Answer {i}: Of course! I'm here to help."
        })
    
    # 加载历史（自动裁剪）
    history = await memory.get_history(raw_messages)
    
    print(f"原始消息数: {len(raw_messages)}")
    print(f"加载后消息数: {len(history)}")
    print(f"总 tokens: {sum(memory.count_message_tokens(m) for m in history)}")
    
    # 7. 清空缓存
    memory.clear_cache()
    print("\n缓存已清空")
    memory.print_cache_stats()


if __name__ == "__main__":
    asyncio.run(main())