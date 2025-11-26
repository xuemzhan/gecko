# examples/zhipu_agent.py
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Fix] Correct import
from gecko.plugins.models import ZhipuChat 

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key: return

    agent = AgentBuilder()\
        .with_model(ZhipuChat(api_key=api_key, model="glm-4-flash"))\
        .build()

    output = await agent.run("用中文介绍一下 Gecko 框架的优势")
    print(output.content) # type: ignore

if __name__ == "__main__":
    asyncio.run(main())