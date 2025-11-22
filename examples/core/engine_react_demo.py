# examples/core/engine_react_demo.py
import asyncio
import os
from typing import Any, Dict

from pydantic import BaseModel, Field

# 导入核心组件
from gecko.core.agent import Agent
from gecko.core.message import Message
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox
from gecko.core.engine.react import ReActEngine
from gecko.plugins.tools.base import BaseTool
# [修改] 导入新的模型类
from gecko.plugins.models import ZhipuChat

# ==========================================
# 1. 定义简单的工具 (保持不变)
# ==========================================

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful for performing basic arithmetic operations. Input should be a math expression string."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g., '2 + 2'"
            }
        },
        "required": ["expression"]
    }
    # 为了兼容新版 BaseTool，这里虽然没有用 args_schema，但手动实现了 parameters 属性
    # 如果使用新版 BaseTool，建议定义 Pydantic Model。
    # 这里为了最小化改动，我们通过覆盖 _run 并忽略类型检查来适配 Demo
    
    # 定义一个临时的 args schema 以满足 BaseTool 初始化检查
    class Args(BaseModel):
        expression: str
    args_schema: type[BaseModel] = Args

    async def _run(self, args: Args) -> str:
        expression = args.expression
        try:
            # 注意：eval 在生产环境中是不安全的，仅用于演示
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"

class WeatherTool(BaseTool):
    name: str = "get_current_weather"
    description: str = "Get the current weather in a given location"
    
    class Args(BaseModel):
        location: str
        unit: str = "celsius"
    args_schema: type[BaseModel] = Args

    async def _run(self, args: Args) -> str:
        return f"The weather in {args.location} is sunny and 25°C."

# ==========================================
# 2. 定义结构化输出模型 (保持不变)
# ==========================================

class AnalysisReport(BaseModel):
    """分析报告结构"""
    summary: str = Field(description="对用户问题的简短总结")
    action_items: list[str] = Field(description="建议采取的行动项列表")
    priority: str = Field(description="优先级 (High/Medium/Low)")

# ==========================================
# 3. 主演示流程
# ==========================================

async def main():
    # 1. 初始化模型 [修改]
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Please set ZHIPU_API_KEY environment variable.")
        return

    llm = ZhipuChat(api_key=api_key, model="glm-4-flash")

    # 2. 初始化工具箱
    toolbox = ToolBox(tools=[CalculatorTool(), WeatherTool()])

    # 3. 初始化记忆
    memory = TokenMemory(session_id="react_demo_session", max_tokens=2000)

    # 4. 构建 Agent (使用 ReActEngine)
    agent = Agent(
        model=llm,
        toolbox=toolbox,
        memory=memory,
        engine_cls=ReActEngine,
        max_turns=5 # 限制最大思考轮数
    )

    print("\n🚀 ReAct Agent Demo (Powered by ZhipuChat)\n")

    # --- 场景 1: 需要使用工具的复杂查询 ---
    query1 = "What is 123 * 45? Also, what's the weather in Beijing?"
    print(f"👤 User: {query1}")
    print("🤖 Agent (Thinking...):")
    
    # 使用 run() 方法 (非流式)
    response1 = await agent.run(query1)
    print(f"💡 Final Answer: {response1.content}\n")
    
    # 查看统计
    if agent.engine.stats:
        print(f"📊 Stats: Steps={agent.engine.stats.total_steps}, ToolCalls={agent.engine.stats.tool_calls}")

    print("-" * 50)

    # --- 场景 2: 结构化输出 ---
    query2 = "Based on the weather in Beijing, suggest a weekend plan."
    print(f"\n👤 User: {query2} (Requesting Structured Output)")
    
    # 使用 run() 并指定 response_model
    result = await agent.run(query2, response_model=AnalysisReport)
    print(f"📦 Structured Result:\n{result.model_dump_json(indent=2)}\n")

    print("-" * 50)

    # --- 场景 3: 流式输出 ---
    query3 = "Tell me a short story about a Gecko programmer."
    print(f"\n👤 User: {query3} (Streaming Mode)")
    print("🌊 Stream: ", end="", flush=True)
    
    async for chunk in agent.stream(query3):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())