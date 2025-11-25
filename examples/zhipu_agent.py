# examples/zhipu_agent.py
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Fix] 使用新的 Presets 类
from gecko.plugins.models.presets.zhipu import ZhipuChat

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Error: ZHIPU_API_KEY not found.")
        return

    # [Fix] 实例化类
    model = ZhipuChat(api_key=api_key, model="glm-4-flash", temperature=0.8)

    agent = AgentBuilder().with_model(model).build()

    output = await agent.run([
        Message(role="user", content="用中文介绍一下 Gecko 框架的插件化设计优势")
    ])
    print(output.content) # type: ignore

if __name__ == "__main__":
    asyncio.run(main())