# 核心概念

## Agent (智能体)

`Agent` 是 Gecko 的原子执行单元。它封装了以下三个核心组件：
1.  **Model**: 负责生成文本和决策。
2.  **Memory**: 负责管理上下文历史。
3.  **ToolBox**: 负责执行外部工具。

## Cognitive Engine (认知引擎)

Gecko 将推理逻辑从 Agent 中剥离，称为 `Engine`。目前内置了 **ReActEngine**。

### ReAct Engine 特性
*   **死循环检测**: 基于 Hash 自动检测重复的工具调用参数，防止 Agent 陷入死循环。
*   **观测值截断**: 自动截断过长的工具输出（如爬虫抓取了 10MB 文本），防止 Context Window 溢出。
*   **结构化重试**: 如果 LLM 返回的 JSON 格式错误，Engine 会自动将错误信息反馈给 LLM 进行自我修正。

## Memory (记忆)

`TokenMemory` 是 Gecko 的默认记忆实现。

*   **滑动窗口**: 自动计算 Token 数，当超出 `max_tokens` 时，保留 System Prompt，并移除最早的历史消息。
*   **自动摘要**: 使用 `SummaryTokenMemory` 可以在移除历史消息前自动生成摘要并注入 Context。

## ToolBox (工具箱)

`ToolBox` 负责工具的注册、执行和并发控制。

*   **并发安全**: 内置线程锁，保证统计数据的安全性。
*   **Schema 生成**: 自动从 Pydantic 的 `args_schema` 生成 OpenAI 兼容的 JSON Schema。
*   **批量执行**: 支持 `execute_many`，利用 `anyio.TaskGroup` 并发执行多个工具调用。