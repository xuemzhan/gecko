# examples/full_tool_test.py
import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air

# 自动加载所有工具（只需 import）
from gecko.plugins.tools.standard.calculator import CalculatorTool  # noqa: F401
from gecko.plugins.tools.standard.duckduckgo import DuckDuckGoSearch  # noqa: F401

async def main():
    agent = AgentBuilder()\
        .with_model(glm_4_5_air(temperature=0.3))\
        .build()

    output = await agent.run([
        Message(role="user", content="请同时完成两件事："
                 "1. 计算 (12345 + 67890) * 2.5"
                 "2. 搜索今天北京的天气预报"
                 "最后用中文总结")
    ])
    print("\n=== 最终回答 ===\n")
    print(output.content)

if __name__ == "__main__":
    asyncio.run(main())