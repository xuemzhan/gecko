# examples/message_demo.py
import asyncio
from gecko.core.message import Message, MediaResource


async def main():
    print("=== Gecko Message 示例 ===\n")
    
    # 1. 简单文本消息
    print("1. 简单文本消息")
    user_msg = Message.user("Hello, how are you?")
    print(f"   {user_msg}")
    print(f"   OpenAI 格式: {user_msg.to_openai_format()}\n")
    
    # 2. 助手消息
    print("2. 助手消息")
    assistant_msg = Message.assistant("I'm doing great, thanks!")
    print(f"   {assistant_msg}\n")
    
    # 3. 系统消息
    print("3. 系统消息")
    system_msg = Message.system("You are a helpful assistant.")
    print(f"   {system_msg}\n")
    
    # 4. 多模态消息（同步）
    print("4. 多模态消息（同步）")
    # 假设有本地图片
    # multimodal_msg = Message.user(
    #     text="What's in this image?",
    #     images=["./test_image.jpg"]
    # )
    # print(f"   {multimodal_msg}")
    # print(f"   包含 {multimodal_msg.get_image_count()} 张图片\n")
    
    # 5. 多模态消息（异步）
    print("5. 多模态消息（异步）")
    # async_msg = await Message.user_async(
    #     text="Analyze these images",
    #     images=["./image1.jpg", "./image2.jpg"]
    # )
    # print(f"   {async_msg}\n")
    
    # 6. 工具返回消息
    print("6. 工具返回消息")
    tool_msg = Message.tool_result(
        tool_call_id="call_123",
        content={"result": "Search completed", "count": 42},
        tool_name="search"
    )
    print(f"   {tool_msg}")
    print(f"   内容: {tool_msg.get_text_content()[:50]}\n")
    
    # 7. 从 OpenAI 格式解析
    print("7. 从 OpenAI 格式解析")
    openai_format = {
        "role": "assistant",
        "content": "Here's what I found...",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}'
                }
            }
        ]
    }
    parsed_msg = Message.from_openai(openai_format)
    print(f"   {parsed_msg}\n")
    
    # 8. 消息工具方法
    print("8. 消息工具方法")
    long_msg = Message.user("This is a very long message " * 20)
    print(f"   原始长度: {len(long_msg.get_text_content())}")
    
    truncated = long_msg.truncate_content(50)
    print(f"   截断后: {truncated.get_text_content()}")
    
    print(f"   是否为空: {long_msg.is_empty()}")
    print(f"   是否有图片: {long_msg.has_images()}\n")
    
    # 9. 消息克隆
    print("9. 消息克隆")
    cloned = user_msg.clone()
    print(f"   原始: {user_msg}")
    print(f"   克隆: {cloned}")
    print(f"   是否相同对象: {cloned is user_msg}\n")
    
    # 10. MediaResource 示例
    print("10. MediaResource 示例")
    # 从 URL
    url_resource = MediaResource(
        url="https://example.com/image.jpg",
        detail="high"
    )
    print(f"   URL 资源: {url_resource.to_openai_image_url()}")
    
    # 从 base64
    base64_resource = MediaResource(
        base64_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        mime_type="image/png"
    )
    print(f"   Base64 大小估算: {base64_resource.get_size_estimate()} bytes\n")


if __name__ == "__main__":
    asyncio.run(main())