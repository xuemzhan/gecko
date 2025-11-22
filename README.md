# 🦎 Gecko Agent Framework

> **工业级、异步优先、协议驱动的 Python AI 智能体开发框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![AsyncIO](https://img.shields.io/badge/Async-AnyIO-green.svg)](https://anyio.readthedocs.io/)

Gecko 是一个专为**生产环境**设计的 AI Agent 框架。它拒绝“魔法”和过度封装，强调**类型安全**、**并发控制**与**确定性执行**。与其他框架不同，Gecko 从底层构建了对异步 I/O、状态持久化和断点恢复的原生支持，非常适合构建高并发、长流程的复杂业务应用。

---

## 🌟 核心特性 (Core Features)

*   **🚀 原生异步 (Async-First)**
    *   基于 `anyio` 构建，核心链路全异步。
    *   内置 `ThreadOffloadMixin`，自动将同步 I/O（如 SQLite 写入、文件操作）卸载至线程池，杜绝阻塞事件循环。
*   **🛡️ 协议驱动 (Protocol-Driven)**
    *   通过 Python `Protocol` 定义接口（Model, Storage, Tool），而非强制继承。
    *   解耦具体实现，轻松替换底层组件（如从 OpenAI 切换至本地 Ollama，从 SQLite 切换至 Redis）。
*   **🔄 强大的 ReAct 引擎**
    *   **死循环检测**：基于 Hash 的工具调用指纹检测，自动熔断重复操作。
    *   **观测值截断**：智能截断过长的工具输出，防止 Context Window 爆炸。
    *   **自动重试**：内置指数退避重试机制，应对 LLM 幻觉和网络波动。
*   **💾 状态持久化与恢复 (Resumability)**
    *   **Workflow 引擎**：支持 DAG（有向无环图）编排，支持条件分支与循环。
    *   **断点续传**：支持 Step 级别的状态快照（Checkpoint）。系统崩溃重启后，可调用 `resume()` 无缝恢复执行，绝不丢失进度。
    *   **原子写入**：内置 `FileLock` 跨进程锁，确保 SQLite/文件存储在多进程（如 Gunicorn）环境下的数据安全。
*   **🧩 插件化架构**
    *   **Models**：基于 `LiteLLM` 适配 100+ 模型（OpenAI, Azure, ZhipuAI, Ollama 等）。
    *   **Storage**：支持 SQLite (WAL模式), Redis, ChromaDB, LanceDB 等。
    *   **Tools**：基于 Pydantic 的强类型工具定义，自动生成 OpenAI Schema。

---

## 📦 安装

*(注：项目尚未发布到 PyPI，目前建议源码安装)*

```bash
# 基础安装
pip install gecko-ai

# 安装所有可选依赖 (Redis, Vector DBs, etc.)
pip install "gecko-ai[all]"
```

---

## ⚡️ 快速开始

### 1. 基础 Agent (ZhipuAI 示例)

只需几行代码即可构建一个具备工具调用能力的 Agent。

```python
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models import ZhipuChat
from gecko.plugins.tools.standard import CalculatorTool

# 设置 API Key
os.environ["ZHIPU_API_KEY"] = "your_api_key"

async def main():
    # 1. 初始化模型
    model = ZhipuChat(api_key=os.environ["ZHIPU_API_KEY"], model="glm-4-flash")
    
    # 2. 构建 Agent
    agent = (AgentBuilder()
             .with_model(model)
             .with_tools([CalculatorTool()])  # 自动注册标准工具
             .with_session_id("quick_start_session")
             .build())

    # 3. 执行任务
    response = await agent.run("请计算 (123 * 45) 的结果，并写一首诗赞美它。")
    print(f"Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 可恢复的工作流 (Resumable Workflow)

展示 Gecko 最强大的特性：定义一个 DAG 工作流，模拟崩溃并恢复。

```python
from gecko.compose.workflow import Workflow, CheckpointStrategy
from gecko.compose.nodes import step
from gecko.plugins.storage.factory import create_storage

@step("Research")
async def research(topic: str):
    print(f"🔍 正在调研: {topic}...")
    return f"{topic} 的调研报告"

@step("Write")
async def write(context):
    data = context.get_last_output()
    print(f"✍️ 正在撰写关于 {data} 的文章...")
    # 模拟崩溃！
    # raise RuntimeError("系统崩溃！") 
    return "最终文章内容"

async def main():
    # 初始化持久化存储
    storage = await create_storage("sqlite:///./workflow_state.db")
    
    # 定义工作流
    wf = Workflow(
        name="ArticleFlow", 
        storage=storage, 
        checkpoint_strategy=CheckpointStrategy.ALWAYS # 每一步都保存
    )
    
    wf.add_node("Research", research)
    wf.add_node("Write", write)
    wf.add_edge("Research", "Write")
    wf.set_entry_point("Research")
    
    session_id = "session_001"
    
    try:
        # 首次运行
        await wf.execute("AI Agents", session_id=session_id)
    except Exception:
        print("❌ 检测到崩溃，正在恢复...")
        # 恢复运行：自动跳过已完成的 "Research" 节点，直接从 "Write" 重试
        result = await wf.resume(session_id)
        print(f"✅ 恢复并完成: {result}")

import asyncio
if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 自定义工具 (Pydantic 强类型)

```python
from pydantic import BaseModel, Field
from gecko.plugins.tools.base import BaseTool, ToolResult

class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "查询天气"
    args_schema: type[BaseModel] = WeatherArgs

    async def _run(self, args: WeatherArgs) -> ToolResult:
        # 这里可以进行异步 API 调用
        return ToolResult(content=f"{args.city} 天气晴朗，25℃")
```

---

## 🏗️ 系统架构

Gecko 采用清晰的分层架构：

1.  **Compose Layer (编排层)**:
    *   `Workflow`: DAG 调度，状态管理。
    *   `Team`: 并行多智能体协作 (Map-Reduce)。
2.  **Core Layer (核心层)**:
    *   `Engine`: ReAct / Chain 推理逻辑。
    *   `Memory`: Token 计数、滑动窗口、自动摘要。
    *   `Structure`: 结构化输出解析与修复。
3.  **Support Layer (支撑层)**:
    *   `ToolBox`: 工具注册、Schema 生成、并发限流。
    *   `EventBus`: 异步事件分发。
4.  **Plugin Layer (插件层)**:
    *   `Models`: 适配 OpenAI, Zhipu, Ollama (LiteLLM Driver)。
    *   `Storage`: 适配 SQLite, Redis, Chroma, LanceDB。

---

## 🔌 存储后端支持

Gecko 的存储层通过 URL Scheme 配置，支持即插即用：

| Scheme | Backend | 用途 | 特性 |
| :--- | :--- | :--- | :--- |
| `sqlite://` | SQLite | Session/State | WAL 模式，进程锁，无依赖 |
| `redis://` | Redis | Session/Cache | 高性能，TTL 支持 |
| `chroma://` | ChromaDB | Vector RAG | 本地/服务端模式 |
| `lancedb://` | LanceDB | Vector RAG | 高性能文件型向量库 |

---

## 🛣️ Roadmap

*   **v0.1 (Alpha)**: 核心 ReAct 引擎，基础 ToolBox。
*   **v0.2 (Current)**: 
    *   ✅ 引入 Workflow DAG 引擎与断点恢复。
    *   ✅ 引入 Storage Plugin 系统 (SQLite/Redis/Vector)。
    *   ✅ 增强 ReAct 稳定性 (死循环检测)。
    *   ✅ 完善测试覆盖率。
*   **v0.3 (Planned)**:
    *   🚧 **RAG 增强**: 完善 Knowledge Plugin，支持更多 Loader 和 Rerank 策略。
    *   🚧 **生态适配**: 提供 LangChain/LlamaIndex 适配器。
    *   🚧 **可观测性**: 集成 OpenTelemetry。

---

## 🤝 贡献

欢迎提交 Pull Request 或 Issue！

1.  Fork 本仓库。
2.  创建特性分支 (`git checkout -b feature/AmazingFeature`).
3.  提交更改 (`git commit -m 'Add some AmazingFeature'`).
4.  推送到分支 (`git push origin feature/AmazingFeature`).
5.  开启 Pull Request。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。