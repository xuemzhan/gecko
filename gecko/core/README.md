# Gecko Core (`gecko.core`)

`gecko.core` 是 Gecko 框架的基础设施层。它定义了智能体（Agent）的核心生命周期、认知架构、记忆管理以及工具执行机制。该模块采用 **Async-first** 设计，并通过 **Protocol** 协议层实现了与具体模型和存储后端的解耦。

## 🌟 核心特性

*   **通用 Agent 门面**：统一的 `Agent` 类，支持单次推理 (`run`) 和流式输出 (`stream`)。
*   **ReAct 认知引擎**：内置稳健的 ReAct (Reason+Act) 循环，支持：
    *   自动死循环检测与中断。
    *   流式输出时的工具调用完整性保护。
    *   观测值（Observation）自动截断，防止 Context 溢出。
*   **智能记忆管理**：
    *   **TokenMemory**：基于滑动窗口的精确 Token 计数与裁剪。
    *   **SummaryTokenMemory** (New): 支持在上下文超限时自动调用 LLM 生成摘要，保留长期记忆。
*   **异步工具箱**：`ToolBox` 支持并发执行，并自动处理同步函数的线程卸载，防止阻塞事件循环。
*   **结构化输出**：原生支持将 LLM 响应解析为 Pydantic 对象，内置多策略修复机制。
*   **事件驱动**：内置 `EventBus`，支持生命周期 Hook 和中间件拦截。

## 📦 模块组成

| 模块 | 组件 | 职责 |
| :--- | :--- | :--- |
| **agent.py** | `Agent` | 智能体的主入口，协调 Engine、Memory 和 Toolbox。 |
| **engine/** | `ReActEngine`, `CognitiveEngine` | 实现了 "思考-行动-观察" 的推理循环。 |
| **memory.py** | `TokenMemory`, `SummaryTokenMemory` | 管理对话历史，负责上下文的加载、裁剪和摘要。 |
| **toolbox.py** | `ToolBox` | 工具的注册表与执行器，负责并发控制与参数校验。 |
| **message.py** | `Message` | 标准化的消息数据结构，支持多模态（文本/图片）。 |
| **output.py** | `AgentOutput` | 标准化的输出结构，包含内容、工具调用和 Token 统计。 |
| **protocols.py** | `ModelProtocol`, `StorageProtocol` | 定义核心接口，实现鸭子类型多态。 |
| **structure.py**| `StructureEngine` | 负责 JSON 提取与 Pydantic 模型转换。 |

## 🚀 快速开始

### 1. 基础 ReAct Agent

```python
import asyncio
import os
from gecko.core.agent import Agent
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox
from gecko.plugins.models.zhipu import ZhipuChat  # 假设使用智谱插件
from gecko.plugins.tools.standard import CalculatorTool

async def main():
    # 1. 初始化模型
    llm = ZhipuChat(api_key=os.getenv("ZHIPU_API_KEY"), model="glm-4-flash")

    # 2. 准备工具箱
    toolbox = ToolBox(tools=[CalculatorTool()])

    # 3. 初始化记忆 (基础滑动窗口)
    memory = TokenMemory(session_id="user_session_001", max_tokens=2000)

    # 4. 构建 Agent
    agent = Agent(model=llm, toolbox=toolbox, memory=memory)

    # 5. 运行
    response = await agent.run("计算 123 * 45，并告诉我结果的平方根是多少？")
    print(f"Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 使用摘要记忆 (Summary Memory)

当对话历史超过 `max_tokens` 时，`SummaryTokenMemory` 会自动调用模型将早期对话压缩为摘要。

```python
from gecko.core.memory import SummaryTokenMemory

# 初始化带有摘要能力的记忆组件
memory = SummaryTokenMemory(
    session_id="long_context_session",
    model=llm,  # 需要传入模型实例用于生成摘要
    max_tokens=1000,  # 设定较小的阈值以触发摘要
    summary_prompt="请简要总结上述对话的关键信息："  # 可选：自定义摘要提示词
)

agent = Agent(model=llm, toolbox=toolbox, memory=memory)
```

### 3. 结构化输出

强制 Agent 返回符合 Pydantic 定义的 JSON 数据。

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    sentiment: str = Field(description="情感倾向: positive/negative")
    keywords: list[str] = Field(description="提取的关键词列表")

result = await agent.run(
    "这款新手机的电池续航太棒了，但是拍照效果一般。",
    response_model=AnalysisResult
)

print(result.sentiment)  # positive
print(result.keywords)   # ['电池续航', '拍照效果']
```

## ⚙️ 核心机制详解

### ReAct 引擎流程
`ReActEngine` 的 `step` 方法执行以下循环：
1.  **Think**: 将当前上下文（System + History + Input）发送给 LLM。
2.  **Route**: 
    *   如果 LLM 返回纯文本 -> 结束循环，返回结果。
    *   如果 LLM 返回工具调用 -> 进入下一步。
3.  **Act**: `ToolBox` 并发执行所有工具调用。
4.  **Observe**: 将工具执行结果（Result/Error）追加到上下文。
5.  **Loop**: 回到第一步，LLM 根据观测结果生成新的回答。

*注：引擎内置了死循环检测机制，如果 Agent 连续以相同的参数调用同一个工具，循环将被强制中断。*

### 协议驱动 (Protocols)
Gecko Core 不依赖具体的模型 SDK 或数据库驱动。任何类只要实现了 `gecko.core.protocols` 中定义的 `Protocol`，即可被 Agent 使用。

例如，自定义一个简单的模型适配器：

```python
from gecko.core.protocols import ModelProtocol, CompletionResponse

class MyCustomModel(ModelProtocol):
    async def acompletion(self, messages, **kwargs) -> CompletionResponse:
        # 调用你的私有模型 API
        return CompletionResponse(...)
```

## 🛠️ 开发指南

*   **添加新工具**：继承 `gecko.plugins.tools.base.BaseTool` 并定义 `args_schema`。
*   **自定义存储**：继承 `gecko.plugins.storage.abc.AbstractStorage` 并实现 `SessionInterface`。
*   **调试**：设置 `LOG_LEVEL=DEBUG` 可以看到详细的推理步骤和工具调用日志。

## ⚠️ 注意事项

1.  **Token 计算**：默认依赖 `tiktoken`。如果未安装，将回退到字符估算模式，可能导致 Context 窗口计算不准。
2.  **异步上下文**：所有 IO 操作（工具执行、存储读写）均已在框架层面做了异步优化，但在编写自定义工具时，请尽量避免使用阻塞式 IO（如 `time.sleep` 或 `requests`），应使用 `asyncio.sleep` 或 `httpx`，或者使用 `gecko.core.utils.run_sync` 包装。