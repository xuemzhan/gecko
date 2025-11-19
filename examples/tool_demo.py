# examples/tool_demo.py
import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air

# 自动加载所有 @tool 装饰的工具
from gecko.plugins.tools import calculator, duckduckgo  # noqa: F401

async def main():
    agent = AgentBuilder()\
        .with_model(glm_4_5_air())\
        .build()

    output = await agent.run([
        Message(role="user", content="计算 123 * 456，然后搜索一下 Gecko AI 框架")
    ])
    print(output.content)

if __name__ == "__main__":
    asyncio.run(main())