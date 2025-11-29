# examples/memory_demo.py
"""
Memory 模块使用示例（基于增强版 TokenMemory / SummaryTokenMemory）

本文件包含两个 Demo：

1. basic_token_memory_demo
   - 使用 TokenMemory + SQLiteStorage + Mock 模型驱动
   - 演示：
     - 单条 Token 计数
     - 同步 / 异步批量计数
     - LRU 缓存统计
     - 历史消息裁剪 (get_history)

2. zhipu_summary_memory_demo
   - 使用真实的 ZhipuChat 作为模型 + SummaryTokenMemory
   - 演示：
     - 长对话自动摘要
     - 历史上下文在 max_tokens 限制内进行「系统 + 摘要 + 最近消息」重排

注意：
- 第二个 Demo 需要环境变量 ZHIPU_API_KEY，如果未配置，会自动跳过并给出提示。
"""

import asyncio
import os
from unittest.mock import MagicMock

from gecko.core.memory import TokenMemory, SummaryTokenMemory
from gecko.core.message import Message
from gecko.plugins.models import ZhipuChat
from gecko.plugins.storage.backends.sqlite import SQLiteStorage


# ======================================================================
# 1. 基础 Demo：TokenMemory + SQLiteStorage + Mock 模型驱动
# ======================================================================


async def basic_token_memory_demo() -> None:
    """
    基础 Demo：
    - 不访问真实 LLM，完全本地运行
    - 展示 TokenMemory 的主要能力
    """

    print("\n================ 基础 TokenMemory Demo ================\n")

    # 1. 创建存储（当前 TokenMemory 尚未实际使用 storage，但预留接口）
    storage = SQLiteStorage("sqlite://./test_memory_demo.db")

    # 2. 创建一个简单的 Mock Driver，用于演示「模型驱动计数」
    #    在实际使用中，这里可以换成 ZhipuChat / OpenAI 等真实模型
    mock_driver = MagicMock()
    mock_driver.count_tokens.return_value = 10  # 模拟任意输入都返回 10 tokens

    # 3. 创建 TokenMemory 实例
    memory = TokenMemory(
        session_id="user_123",
        storage=storage, # type: ignore
        max_tokens=4000,
        model_name="gpt-4",     # 用于 tiktoken 编码器选择
        model_driver=mock_driver,  # 注入模型驱动（优先用于计数）
        cache_size=1000,
        max_message_length=10000,
        enable_async_counting=True,  # 启用线程池异步计数
    )

    print("TokenMemory 实例：", memory)

    # 4. 计算单条消息 Token 数
    msg = Message.user("What's the weather today?")
    tokens = memory.count_message_tokens(msg)
    print(f"\n[单条消息] tokens: {tokens}（来自 mock_driver.count_tokens，固定返回 10）")

    # 5. 批量计算（同步）
    messages = [
        Message.user("Hello"),
        Message.assistant("Hi there!"),
        Message.user("How are you?"),
    ]

    counts_sync_cache = memory.count_messages_batch(messages, use_cache=True)
    counts_sync_no_cache = memory.count_messages_batch(messages, use_cache=False)

    print(f"\n[批量计数 - 同步, 使用缓存]  : {counts_sync_cache}")
    print(f"[批量计数 - 同步, 不使用缓存]: {counts_sync_no_cache}")

    # 6. 批量计算（异步）
    counts_async_cache = await memory.count_messages_batch_async(messages, use_cache=True)
    counts_async_no_cache = await memory.count_messages_batch_async(messages, use_cache=False)

    print(f"\n[批量计数 - 异步, 使用缓存]  : {counts_async_cache}")
    print(f"[批量计数 - 异步, 不使用缓存]: {counts_async_no_cache}")

    # 7. 缓存性能 & 统计信息
    print("\n=== 缓存性能 / 统计信息 ===")

    # 多次重复计数，增加命中率
    for _ in range(3):
        memory.count_message_tokens(msg)

    memory.print_cache_stats()

    # 8. 历史加载 & 裁剪示例
    print("\n=== 历史加载 / 裁剪示例 ===")

    raw_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    # 构造大量历史消息，模拟长对话
    for i in range(100):
        raw_messages.append(
            {
                "role": "user",
                "content": f"Question {i}: Can you help me with something very important?",
            }
        )
        raw_messages.append(
            {
                "role": "assistant",
                "content": f"Answer {i}: Of course! I'm always here to help you.",
            }
        )

    history = await memory.get_history(raw_messages)

    print(f"原始消息数: {len(raw_messages)}")
    print(f"裁剪后消息数: {len(history)}")
    print(f"裁剪后总 tokens（基于 TokenMemory）: {memory.count_total_tokens(history)}")

    # 9. 清空缓存
    memory.clear_cache()
    print("\n缓存已清空：")
    memory.print_cache_stats()


# ======================================================================
# 2. 高级 Demo：ZhipuChat + SummaryTokenMemory 自动摘要
# ======================================================================


async def zhipu_summary_memory_demo() -> None:
    """
    高级 Demo：
    - 使用真实的 ZhipuChat 作为模型
    - 使用 SummaryTokenMemory 管理长对话上下文，并在超限时自动摘要

    运行前提：
    - 环境变量 ZHIPU_API_KEY 已配置
    """

    print("\n================ Zhipu + SummaryTokenMemory Demo ================\n")

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("⚠️ 未检测到环境变量 ZHIPU_API_KEY，跳过 Zhipu Summary Demo。")
        print("   如需运行该 Demo，请先在环境中设置 ZHIPU_API_KEY。")
        return

    # 1. 创建 ZhipuChat 模型（实现了 ModelProtocol）
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")

    # 2. 创建 SummaryTokenMemory
    #    - max_tokens：整体上下文窗口预算（含系统 + 摘要 + 最近消息）
    #    - summary_reserve_tokens：预留给摘要消息的 Token 预算
    memory = SummaryTokenMemory(
        session_id="zhipu_summary_demo",
        model=model,
        max_tokens=512,
        summary_reserve_tokens=128,
    )

    # 3. 构造一个较长的对话历史，用于触发「自动摘要」
    raw_messages = [
        {
            "role": "system",
            "content": "你是 Gecko 框架中的智能助手，擅长解释多智能体、工具调用和上下文记忆。",
        }
    ]

    # 模拟用户与助手的长对话，内容稍微长一点以便触发摘要
    for i in range(20):
        raw_messages.append(
            {
                "role": "user",
                "content": f"第 {i} 个问题：请用详细一点的方式，解释一下 Gecko 在多智能体编排上的优势？",
            }
        )
        raw_messages.append(
            {
                "role": "assistant",
                "content": (
                    f"第 {i} 个回答：Gecko 通过统一的消息总线、工具箱和内置内存模块，"
                    f"让智能体之间可以灵活编排、共享上下文，并基于 Token 感知进行裁剪。"
                    f"这使得复杂场景下的协作更加稳定和高效。"
                ),
            }
        )

    # 4. 使用 SummaryTokenMemory 进行历史加载
    history = await memory.get_history(raw_messages)

    # 5. 输出基本统计信息
    print(f"原始消息数: {len(raw_messages)}")
    print(f"SummaryTokenMemory 裁剪后消息数: {len(history)}")

    total_tokens = memory.count_total_tokens(history)
    print(f"裁剪后总 tokens: {total_tokens}（预算 max_tokens={memory.max_tokens}）")

    # 6. 检查是否插入了摘要 System 消息
    summary_msgs = [
        m for m in history if m.role == "system" and "Previous context:" in (m.content or "")
    ]

    if summary_msgs:
        print("\n✅ 已插入摘要 System 消息，内容示例：\n")
        # 由于摘要可能较长，这里只展示前 200 字
        print(summary_msgs[0].content[:200])  # type: ignore
    else:
        print("\n⚠️ 当前样例未触发摘要插入，可以尝试调小 max_tokens 或增加历史长度。")

    # 7. 还可以基于裁剪后的 history，继续调用 ZhipuChat 进行对话
    #    下面演示一个简单的「基于总结后的上下文再问一个问题」的调用流程：

    user_question = "基于以上历史，请用一句话总结 Gecko 的核心优势。"
    history_msgs = [m.to_openai_format() for m in history]
    history_msgs.append({"role": "user", "content": user_question})

    print("\n=== 使用裁剪后的上下文 + 摘要，向 Zhipu 提问 ===\n")
    print("用户问题：", user_question, "\n")

    response = await model.acompletion(history_msgs)
    # 按 ModelProtocol / OpenAI 风格取内容
    answer = response.choices[0].message.get("content", "")  # type: ignore
    print("Zhipu 回复：", answer)


# ======================================================================
# 3. 主入口
# ======================================================================


async def main() -> None:
    """
    统一入口：
    - 先跑本地 TokenMemory Demo（无网络、无真实 LLM）
    - 再尝试跑 Zhipu + SummaryTokenMemory Demo（有 API Key 时）
    """
    await basic_token_memory_demo()
    await zhipu_summary_memory_demo()


if __name__ == "__main__":
    asyncio.run(main())
