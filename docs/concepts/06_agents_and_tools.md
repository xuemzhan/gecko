# 智能体与工具箱

**Agent** 是 Gecko 的原子执行单元，而 **Tool** 是 Agent 与外部世界交互的手脚。

## Agent 的解剖

一个 Agent 由以下核心组件组装而成：

```python
agent = AgentBuilder()\
    .with_model(model)          # 大脑：负责推理
    .with_tools([tools])        # 手脚：负责行动
    .with_memory(memory)        # 记忆：负责上下文
    .with_system_prompt("...")  # 人设：负责指令
    .build()
```

### 认知引擎 (Cognitive Engine)

Agent 内部并不直接运行 LLM，而是委托给 **Engine**。默认的 `ReActEngine` 实现了经典的推理循环：

1.  **Think**: 将历史消息、System Prompt 和工具定义发送给 LLM。
2.  **Act**: 解析 LLM 返回的 Tool Calls，通过 `ToolBox` 并发执行工具。
3.  **Observe**: 将工具运行结果（截断后）追加到历史记录中。
4.  **Loop**: 重复上述步骤，直到 LLM 决定停止或达到 `max_turns`。

**鲁棒性特性**：
*   **死循环熔断**: Engine 会计算工具调用的 Hash，如果发现连续重复调用（参数相同），会自动中断并提示 LLM。
*   **观测截断**: 如果工具返回了巨大的文本（如网页 HTML），Engine 会自动截断，防止 Token 溢出。

---

## 工具箱 (ToolBox)

`ToolBox` 是管理工具注册与执行的容器。

### 定义工具

Gecko 强制使用 Pydantic 定义工具参数，这带来了双重好处：
1.  **类型安全**: 运行时自动校验参数。
2.  **Schema 生成**: 自动生成符合 OpenAI 规范的 JSON Schema。

```python
from pydantic import BaseModel, Field
from gecko.plugins.tools.base import BaseTool, ToolResult

# 1. 定义参数 Schema
class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称")
    unit: str = Field("celsius", description="温度单位")

# 2. 定义工具
class WeatherTool(BaseTool):
    name = "get_weather"
    description = "查询天气"
    args_schema = WeatherArgs

    async def _run(self, args: WeatherArgs) -> ToolResult:
        # 这里可以直接使用 args.city, args.unit (已有类型提示)
        return ToolResult(content=f"{args.city} 25°C")
```

### 工具执行流程

当 Engine 决定调用工具时，`ToolBox` 会接管执行：

1.  **解析**: 将 LLM 返回的 JSON 字符串解析为字典。
2.  **校验**: 使用 `args_schema` 校验参数，如果失败，直接返回错误信息给 LLM（让 LLM 重试）。
3.  **并发**: 如果 LLM 一次性调用了多个工具（Parallel Function Calling），ToolBox 会使用 `asyncio` 并发执行。
4.  **容错**: 捕获执行过程中的异常，封装为 `ToolResult(is_error=True)`。