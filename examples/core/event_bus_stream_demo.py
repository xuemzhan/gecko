# examples/core/event_bus_stream_demo.py
"""
流式体验优化演示 (Streaming UX Demo)

本示例演示了 Gecko v0.2 如何解决 ReAct 模式下工具执行期间的"静默期"问题。
通过 EventBus，前端可以接收到 `tool_execution_start` 事件并展示 Loading 动画。
"""
import asyncio
import os
import time
from typing import Type
from pydantic import BaseModel, Field

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.engine.react import ReActEngine 
from gecko.core.events import EventBus, AgentRunEvent
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.models.presets.zhipu import ZhipuChat

# 1. 定义一个慢速工具
class SlowCalculationArgs(BaseModel):
    number: int

class SlowTool(BaseTool):
    name: str = "slow_calculation"
    description: str = "一个模拟耗时计算的工具"
    args_schema: Type[BaseModel] = SlowCalculationArgs

    async def _run(self, args: SlowCalculationArgs) -> ToolResult: # type: ignore
        # 模拟 2 秒耗时
        await asyncio.sleep(2.0)
        return ToolResult(content=f"The result is {args.number * 2}")

# 2. 定义事件处理器 (模拟前端)
async def frontend_handler(event: AgentRunEvent):
    if event.type == "tool_execution_start":
        # 模拟前端收到事件，显示 Loading
        print("\n>>> [Frontend] 收到 'tool_start' 事件")
        print(">>> [Frontend] UI 更新: 🟢 显示 Spinner (正在思考中...)")
    
    elif event.type == "tool_execution_end":
        # 模拟前端收到事件，隐藏 Loading
        print(">>> [Frontend] 收到 'tool_end' 事件")
        print(">>> [Frontend] UI 更新: ⚫ 隐藏 Spinner\n")

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Skipping: ZHIPU_API_KEY not found.")
        return

    # 3. 设置环境
    event_bus = EventBus()
    # 订阅相关事件
    event_bus.subscribe("tool_execution_start", frontend_handler) # type: ignore
    event_bus.subscribe("tool_execution_end", frontend_handler) # type: ignore

    agent = (
        AgentBuilder()
        .with_model(ZhipuChat(api_key=api_key, model="glm-4-flash"))
        .with_tools([SlowTool()])
        .with_engine(
            engine_cls=ReActEngine,  # 默认 ReAct # type: ignore
            event_bus=event_bus # 注入 EventBus
        ) 
        # 注意：AgentBuilder.with_engine 参数处理逻辑可能需要根据 builder.py 实现调整
        # 这里我们直接在 build() 后手动覆盖或者使用支持注入的 Builder 模式
        # 检查 Builder 源码发现 with_engine 接收 **kwargs，会传给 Agent
        # 所以上面的写法是有效的
        .build()
    )

    print("用户: 请帮我计算 21 的两倍 (这个工具很慢)")
    print("Agent 流式输出开始:")
    print("-" * 40)

    # 4. 执行流式推理
    # 预期效果：
    # Text -> [UI: Start Spinner] -> (2s delay) -> [UI: Stop Spinner] -> Text
    async for chunk in agent.stream("请帮我计算 21 的两倍"):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("流式输出结束")
    
    await event_bus.shutdown()

if __name__ == "__main__":
    asyncio.run(main())