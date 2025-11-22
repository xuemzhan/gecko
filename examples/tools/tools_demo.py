# examples/tools/tools_demo.py
"""
Gecko Tool Demo (适配新版 Models 架构)

展示如何结合 ZhipuChat 模型与 ToolBox 构建具备工具调用能力的 Agent。
"""
import asyncio
import os
import sys
from typing import Type

from pydantic import BaseModel, Field

# 确保当前目录在 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from gecko.core.builder import AgentBuilder
from gecko.core.logging import setup_logging

# [修改 1] 从统一的 models 插件入口导入 ZhipuChat
from gecko.plugins.models import ZhipuChat
from gecko.plugins.tools.base import BaseTool, ToolResult
# [修改] 导入 load_tool 用于实例化工具
from gecko.plugins.tools.registry import register_tool, load_tool

# 导入标准工具库以触发自动注册
import gecko.plugins.tools.standard

setup_logging(level="INFO")


# ==========================================
# 自定义工具定义 (保持不变)
# ==========================================

class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称，例如: 'Beijing', 'Shanghai'")

@register_tool("weather_query")
class WeatherTool(BaseTool):
    name: str = "weather_query"
    description: str = "查询特定城市的当前天气状况。"
    args_schema: Type[BaseModel] = WeatherArgs

    async def _run(self, args: WeatherArgs) -> ToolResult: # type: ignore
        print(f"\n[Mock API] Querying weather for {args.city}...")
        mock_data = {
            "Beijing": "Sunny, 25°C, Wind: NW 3km/h",
            "Shanghai": "Rainy, 22°C, Wind: SE 5km/h",
            "New York": "Cloudy, 18°C, Wind: NE 10km/h"
        }
        result = mock_data.get(args.city, "Unknown location")
        return ToolResult(content=result)


# ==========================================
# 主程序
# ==========================================

async def main():
    print("🚀 初始化 Gecko Agent...")

    # 1. 获取 API Key
    api_key = os.environ.get("ZHIPU_API_KEY")
    
    if not api_key:
        print("❌ 未检测到 ZHIPU_API_KEY 环境变量")
        print("请运行: export ZHIPU_API_KEY='your_key_here'")
        # 为了演示继续运行，这里可以抛出异常或者硬编码测试Key(不推荐)
        return

    print(f"✅ 检测到 API Key: {api_key[:6]}******")

    # 2. 初始化模型 [修改 2]
    # 使用新的 ZhipuChat 类，显式传入 api_key 和 model
    try:
        llm = ZhipuChat(
            api_key=api_key,
            model="glm-4-air",
            temperature=0.1
        )
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 3. 构建 Agent
    try:
        # [修改] 显式加载工具实例，满足 AgentBuilder 的严格类型检查
        tools_list = [
            load_tool("calculator"),        # 从注册表加载标准工具
            load_tool("duckduckgo_search"), # 从注册表加载标准工具
            load_tool("weather_query")      # 加载刚刚注册的自定义工具
        ]

        agent = (
            AgentBuilder()
            .with_model(llm)
            .with_tools(tools_list)  # 现在传入的是 BaseTool 实例列表
            .with_max_tokens(4000)
            .build()
        )
    except Exception as e:
        print(f"❌ Agent 构建失败: {e}")
        return

    print(f"📦 已加载工具: {[t.name for t in agent.toolbox.list_tools()]}")
    
    # ==========================================
    # 执行测试
    # ==========================================
    
    # 场景 A: 数学计算
    query_math = "计算 (123 * 45) + sqrt(1024) 的结果是多少？"
    print(f"\nUser: {query_math}")
    try:
        response = await agent.run(query_math)
        print(f"Agent: {response.content}") # type: ignore
    except Exception as e:
        print(f"Execution failed: {e}")

    # 场景 B: 联网搜索 (可选)
    try:
        import duckduckgo_search
        query_search = "2024年巴黎奥运会金牌榜第一名是哪个国家？"
        print(f"\nUser: {query_search}")
        response = await agent.run(query_search)
        print(f"Agent: {response.content}") # type: ignore
    except ImportError:
        print("\n⚠️ 跳过搜索测试：未安装 duckduckgo-search")
    except Exception as e:
        print(f"\n⚠️ 搜索测试出错 (网络问题?): {e}")

    # 场景 C: 自定义工具
    query_weather = "北京和上海现在的天气怎么样？"
    print(f"\nUser: {query_weather}")
    await agent.run(query_weather)

if __name__ == "__main__":
    asyncio.run(main())