# examples/models/models_demo.py
import asyncio
import os
import sys
from dotenv import load_dotenv

# 确保 gecko 包可导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gecko.plugins.models.presets.zhipu import ZhipuChat
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.factory import create_model
from gecko.core.message import Message

load_dotenv()

async def demo_mock():
    """演示：Mock 模式"""
    print("\n--- Mock Demo ---")
    # [修复] 使用标准的模型名称，以便 LiteLLM 识别 Provider (即便是在 Mock 模式下)
    config = ModelConfig(
        model_name="gpt-3.5-turbo",  # 改为标准名称
        api_key="mock",
        extra_kwargs={"mock_response": "This is a mocked response."}
    )
    # 使用工厂创建
    model = create_model(config)
    
    resp = await model.acompletion([{"role": "user", "content": "hi"}])
    print(f"AI: {resp.choices[0].message['content']}")

async def demo_zhipu():
    """演示：智谱真实调用"""
    print("\n--- Zhipu Live Demo ---")
    key = os.getenv("ZHIPU_API_KEY")
    if not key:
        print("Skipping: No ZHIPU_API_KEY found.")
        return

    model = ZhipuChat(api_key=key, model="glm-4-flash")
    
    # 1. 文本
    msg = Message.user("简述AI智能体的优势")
    resp = await model.acompletion([msg.to_openai_format()])
    # 注意：LiteLLM 有时返回 content 为 None (如果被过滤)，需防御
    content = resp.choices[0].message.get("content", "") or ""
    print(f"AI: {content[:50]}...")

    # 2. 流式
    print("Streaming: ", end="")
    async for chunk in model.astream([msg.to_openai_format()]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

async def main():
    await demo_mock()
    await demo_zhipu()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass