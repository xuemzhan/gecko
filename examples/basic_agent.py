import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.litellm import LiteLLMModel  # [新增] 显式引入模型适配器

async def main():
    # [修改] 显式实例化模型对象
    model = LiteLLMModel(
        model="kimi-k2-thinking",
        base_url="http://172.19.37.104:8095/v1",
        api_key="test",
        temperature=0.7
    )

    agent = AgentBuilder()\
        .with_model(model)\
        .build()

    output = await agent.run([
        Message(role="user", content="你好，Gecko v2.0 框架运行成功了吗？")
    ])
    print("✅ Gecko 核心闭环完成！")
    print(output.content)

if __name__ == "__main__":
    asyncio.run(main())