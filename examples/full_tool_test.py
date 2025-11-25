# examples/full_tool_test.py
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Fix] Import ZhipuChat
from gecko.plugins.models.presets.zhipu import ZhipuChat

# 自动加载所有工具（只需 import）
from gecko.plugins.tools.standard.calculator import CalculatorTool  # noqa: F401
from gecko.plugins.tools.standard.duckduckgo import DuckDuckGoSearchTool  # noqa: F401

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Skipping: ZHIPU_API_KEY not set")
        return

    # [Fix] Model Instance
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")

    agent = AgentBuilder()\
        .with_model(model)\
        .build() # Builder 会自动扫描并加载已 import 的 registered tools

    print("🚀 Running Full Tool Test...")
    output = await agent.run([
        Message(role="user", content="请同时完成两件事："
                 "1. 计算 (12345 + 67890) * 2.5 "
                 "2. 搜索今天北京的天气预报 "
                 "最后用中文总结")
    ])
    print("\n=== 最终回答 ===\n")
    print(output.content) # type: ignore

if __name__ == "__main__":
    asyncio.run(main())