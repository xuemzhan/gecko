import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air  # 一行导入

async def main():
    agent = AgentBuilder()\
        .with_model(glm_4_5_air(temperature=0.8)).build()

    output = await agent.run([
        Message(role="user", content="用中文介绍一下 Gecko 框架的插件化设计优势")
    ])
    print(output)

if __name__ == "__main__":
    asyncio.run(main())