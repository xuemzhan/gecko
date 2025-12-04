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
from gecko.core.events import EventBus, AgentRunEvent
from gecko.plugins.models.presets.zhipu import ZhipuChat

# ==========================================
# 1. 定义简单的工具 (保持不变)
# ==========================================
class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful for performing basic arithmetic operations. Input should be a math expression string."
    args_schema: type[BaseModel] = CalculatorArgs
    parameters: Dict[str, Any] = { # type: ignore
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

    async def _run(self, args: Args) -> str: # type: ignore
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

    async def _run(self, args: Args) -> str: # type: ignore
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
# Event Handler
# ==========================================
async def on_tool_event(event: AgentRunEvent):
    """监听工具执行事件，模拟前端 UI 更新"""
    if event.type == "tool_execution_start":
        tools = event.data.get("tools", [])
        names = ", ".join([t["name"] for t in tools])
        print(f"\n[UI Event] ⏳ 正在调用工具: {names} ...")
    elif event.type == "tool_execution_end":
        count = event.data.get("result_count", 0)
        print(f"[UI Event] ✅ 工具执行完成 ({count} 个结果)\n")

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

    # [New] 创建 EventBus 并注册监听器
    event_bus = EventBus()
    event_bus.subscribe("tool_execution_start", on_tool_event) # type: ignore
    event_bus.subscribe("tool_execution_end", on_tool_event) # type: ignore

    # [New] 注入 EventBus
    agent = Agent(
        model=llm,
        toolbox=toolbox,
        memory=memory,
        engine_cls=ReActEngine,
        event_bus=event_bus, # 注入
        max_turns=10
    )

    print("\n🚀 ReAct Agent Demo (Powered by ZhipuChat)\n")

    # --- 场景 1: 需要使用工具的复杂查询 ---
    query1 = "What is 123 * 45? Also, what's the weather in Beijing?"
    print(f"👤 User: {query1}")
    print("🤖 Agent (Thinking...):")
    
    # 使用 run() 方法 (非流式)
    response1 = await agent.run(query1)
    print(f"💡 Final Answer: {response1.content}\n") # type: ignore
    
    # 查看统计
    if agent.engine.stats:
        print(f"📊 Stats: Steps={agent.engine.stats.total_steps}, ToolCalls={agent.engine.stats.tool_calls}")

    print("-" * 50)

    # 修改 query2，增加极其明确的格式指令
    # 技巧：给出 JSON 示例的开头 "{"，诱导模型进入 JSON 补全模式
    query2 = (
        "Based on the weather in Beijing, suggest a weekend plan. "
        "You MUST output the result strictly in JSON format matching the schema. "
        "Do not output any conversational text."
    )
    print(f"\n👤 User: {query2} (Requesting Structured Output)")
    
    # 保持 max_retries=3
    result = await agent.run(
        query2, 
        response_model=AnalysisReport, 
        max_retries=3
    )
    print(f"📦 Structured Result:\n{result.model_dump_json(indent=2)}\n")

    print("-" * 50)

    # --- 场景 3: 流式输出 (长文本/多步推理) ---
    # [修改] 构造一个需要多步思考的问题，验证流式迭代
    query3 = "请先计算 50 的阶乘，然后搜索这个数字的位数，最后写一首关于这个数字的短诗。"
    print(f"\n👤 User: {query3} (Streaming Mode - Iterative)")
    print("🌊 Stream: ", end="", flush=True)
    
    try:
        async for chunk in agent.stream(query3):
            print(chunk, end="", flush=True)
        print("\n")
    except RecursionError:
        print("\n❌ Error: Recursion depth exceeded! (Optimization needed)")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())