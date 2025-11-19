import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message

async def main():
    agent = AgentBuilder()\
        .with_model(
            model="kimi-k2-thinking",
            base_url="http://172.19.37.104:8095/v1",
            api_key="test"
        ).build()

    output = await agent.run([
        Message(role="user", content="你好，Gecko 框架现在彻底运行成功了吗？")
    ])
    print("✅ Gecko 核心闭环完成！")
    print(output.content)

if __name__ == "__main__":
    asyncio.run(main())