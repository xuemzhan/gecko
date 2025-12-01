# 🦎 Gecko Agent Framework (v0.3.1)

> **工业级、异步优先、协议驱动的 Python AI 智能体开发框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![AsyncIO](https://img.shields.io/badge/Async-AnyIO-green.svg)](https://anyio.readthedocs.io/)
[![Observability](https://img.shields.io/badge/Otel-OpenTelemetry-purple.svg)](https://opentelemetry.io/)

**Gecko v0.3.1** 是一个里程碑版本。我们在保持核心“异步优先”和“确定性执行”的基础上，对 Prompt、Output、Structure 等核心模块进行了深度重构与增强，并正式引入了 **RAG (检索增强生成)** 和 **OpenTelemetry 可观测性** 支持。

Gecko 专为构建**生产环境**下的高并发、长流程 AI 应用而生。

---

## 🌟 v0.3.1 核心特性 (Key Features)

### 🚀 1. 生产级 RAG 知识库
*   **Pipeline 入库**：提供 `IngestionPipeline`，支持从加载、切分 (Splitter)、向量化 (Embedding) 到入库 (Upsert) 的全流程。
*   **混合存储**：原生支持 **ChromaDB** 和 **LanceDB**，支持元数据过滤 (`metadata filtering`)。
*   **检索工具**：内置 `RetrievalTool`，一键赋予 Agent 查阅私有知识库的能力。

### 🧩 2. 模块化 Prompt 引擎
*   **Prompt Composer**：支持将 Prompt 拆分为多个 Section（如 System, Few-Shot, Task）进行动态组合。
*   **Prompt Registry**：内置版本管理中心，支持按 `name` + `version` 管理和回滚 Prompt 模板。
*   **Validator & Lint**：静态检查 Prompt 质量，发现未定义变量、禁用词汇或长度超限。

### 🛠️ 3. 增强型结构化输出 (Structure Engine 2.0)
*   **多策略解析**：不仅支持 OpenAI Tool Calls，还内置了 Markdown 提取、JSON 修复、YAML 解析（插件）等多种回退策略。
*   **自动修复**：自动处理常见的 JSON 错误（如尾部逗号、注释、Markdown 包裹）。
*   **类型安全**：深度集成 Pydantic v2，支持 `RootModel` 和嵌套结构的严格校验。

### 📊 4. 全链路可观测性
*   **OpenTelemetry 集成**：内置 Tracing 支持，自动追踪 Agent 运行、工具调用、LLM 请求及数据库操作。
*   **Guardrails**：提供 `InputSanitizer` 中间件，防御 Prompt Injection 攻击，支持威胁分级（Low/Medium/High）。

### 💾 5. 极致的工程鲁棒性
*   **原子写入**：Storage 层引入 `FileLock` 跨进程锁，确保 SQLite 在多 Worker 环境下的数据一致性。
*   **线程卸载**：Token 计算、JSON 序列化、数据库 IO 自动卸载至线程池，杜绝主事件循环阻塞。
*   **断点恢复**：Workflow 引擎支持 Step 级状态快照，系统崩溃后可无缝 `resume()`。

---

## 📦 安装

```bash
# 基础安装
pip install gecko-ai

# 安装 RAG 支持 (包含向量库依赖)
pip install "gecko-ai[rag]"

# 安装所有功能 (Redis, Otel, YAML, etc.)
pip install "gecko-ai[all]"
```

---

## ⚡️ 快速开始

### 1. 构建 RAG 增强的 Agent

```python
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.plugins.models import ZhipuChat
from gecko.plugins.models.embedding import LiteLLMEmbedder, ModelConfig
from gecko.plugins.storage.factory import create_storage
from gecko.plugins.knowledge import IngestionPipeline, RetrievalTool

async def main():
    # 1. 准备向量存储与 Embedder
    vector_store = await create_storage("chroma://./my_knowledge_db")
    embedder = LiteLLMEmbedder(
        config=ModelConfig(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")),
        dimension=1536
    )

    # 2. 知识入库 (仅需运行一次)
    # pipeline = IngestionPipeline(vector_store, embedder)
    # await pipeline.run(["company_policy.pdf", "api_docs.md"])

    # 3. 创建检索工具
    rag_tool = RetrievalTool(vector_store=vector_store, embedder=embedder, top_k=3)

    # 4. 构建 Agent
    llm = ZhipuChat(api_key=os.getenv("ZHIPU_API_KEY"), model="glm-4-flash")
    agent = (AgentBuilder()
             .with_model(llm)
             .with_tools([rag_tool])  # 注入 RAG 工具
             .with_system_prompt("你是一个助手，请优先查阅知识库回答问题。")
             .build())

    # 5. 提问
    response = await agent.run("公司的报销政策是怎样的？")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 结构化输出与 Prompt 管理

```python
from pydantic import BaseModel, Field
from gecko.core.structure import StructureEngine
from gecko.core.prompt import PromptTemplate, PromptValidator

# 1. 定义目标数据结构
class UserProfile(BaseModel):
    name: str = Field(description="用户姓名")
    tags: list[str] = Field(description="用户标签")

async def demo_structure():
    # 2. Prompt 模板与验证
    tpl = PromptTemplate(
        template="Extract info from: {{ text }}",
        input_variables=["text"]
    )
    # 静态检查 Prompt 质量
    issues = PromptValidator().validate(tpl)
    if not issues:
        print("Prompt check passed ✅")

    # 3. 模拟 LLM 输出的脏数据 (包含 Markdown 和 注释)
    llm_output = """
    Here is the JSON:
    ```json
    {
        "name": "Gecko",
        "tags": ["Async", "Robust"], // 这是一个注释
    }
    ```
    """

    # 4. 自动提取与修复
    user = await StructureEngine.parse(
        content=llm_output,
        model_class=UserProfile,
        auto_fix=True  # 自动修复尾部逗号和注释
    )
    print(f"Parsed: {user.name}, Tags: {user.tags}")

import asyncio
if __name__ == "__main__":
    asyncio.run(demo_structure())
```

### 3. 可恢复的工作流 (Resumable Workflow)

```python
from gecko.compose.workflow import Workflow, CheckpointStrategy
from gecko.compose.nodes import step, Next
from gecko.plugins.storage.factory import create_storage

@step("Step1")
async def step_one(ctx):
    print("Executing Step 1...")
    return "Data from Step 1"

@step("Step2")
async def step_two(ctx):
    data = ctx.get_last_output()
    print(f"Executing Step 2 with {data}")
    # 模拟崩溃
    # raise RuntimeError("Crash!")
    return "Finish"

async def main():
    # 使用 SQLite 持久化状态
    storage = await create_storage("sqlite:///./workflow.db")
    
    wf = Workflow(
        "MyFlow", 
        storage=storage, 
        checkpoint_strategy=CheckpointStrategy.ALWAYS # 每步保存
    )
    
    wf.add_node("A", step_one)
    wf.add_node("B", step_two)
    wf.add_edge("A", "B")
    wf.set_entry_point("A")
    
    session_id = "uniq_session_id"
    
    try:
        # 尝试恢复（如果是第一次运行，会自动从头开始）
        res = await wf.resume(session_id)
        print("Result:", res)
    except Exception as e:
        print(f"Workflow paused due to error: {e}")
        # 下次运行此代码将自动从 Step 2 重试，不会重跑 Step 1

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏗️ 核心架构

Gecko v0.3.1 架构进一步解耦，分为核心层与插件层：

| 层级 | 模块 | 功能描述 |
| :--- | :--- | :--- |
| **Compose** | `Workflow` | DAG 编排，支持循环、条件分支、状态持久化 |
| | `Team` | 并行多智能体协作，支持 Map-Reduce 和 Race 模式 |
| **Core** | `Agent` | 智能体门面，组装 Model/Memory/Tools |
| | `Engine` | ReAct 推理循环，流式缓冲，死循环熔断 |
| | `Memory` | `TokenMemory` (LRU缓存), `SummaryTokenMemory` (异步摘要) |
| | `Structure` | 结构化输出解析，Schema 生成，策略插件 |
| | `Prompt` | 模板管理，组合器 (Composer)，注册表 (Registry) |
| **Support** | `ToolBox` | 工具注册与执行，并发控制，参数校验 |
| | `Events` | 异步事件总线，支持中间件拦截 |
| | `Telemetry` | OpenTelemetry 链路追踪，Context 传播 |
| **Plugins** | `Models` | 基于 LiteLLM 适配 OpenAI, Zhipu, Ollama 等 |
| | `Storage` | SQLite (FileLock), Redis, ChromaDB, LanceDB |
| | `Knowledge` | RAG 流水线，文档加载，切分，向量化 |
| | `Guardrails`| 输入清洗，Prompt Injection 防御 |

---

## 🔌 存储后端矩阵

Gecko 存储层通过 URL Scheme 统一管理：

| Scheme | 后端 | 类型 | 用途 | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| `sqlite://` | SQLite | KV | Session/Workflow | WAL 模式，跨进程文件锁，无依赖 |
| `redis://` | Redis | KV | Session/Cache | 高性能，TTL 支持，分布式锁 |
| `chroma://` | ChromaDB | Vector | RAG | 元数据过滤，本地/远程模式 |
| `lancedb://` | LanceDB | Vector | RAG | 基于 Arrow 的高性能文件向量库 |

---

## 🛣️ 版本演进

*   **v0.1**: 基础 ReAct 引擎与工具箱。
*   **v0.2**: 引入 Workflow DAG，断点恢复，SQLite/Redis 存储插件。
*   **v0.3 (Current)**: 
    *   ✅ **RAG**: Knowledge Plugin (Ingestion/Retrieval)。
    *   ✅ **Refactor**: Prompt/Structure/Output 模块化重构。
    *   ✅ **Observability**: OpenTelemetry 集成。
    *   ✅ **Safety**: Guardrails 输入清洗。
*   **v0.4 (Planned)**:
    *   🚧 **Advanced RAG**: Rerank 策略，GraphRAG 支持。
    *   🚧 **Ecosystem**: LangChain/LlamaIndex 桥接器。
    *   🚧 **Deployment**: FastAPI Server 模板，Docker 镜像。

---

## 🤝 贡献

Gecko 是一个开源项目，欢迎通过 Issue 或 Pull Request 参与贡献。请遵循代码规范并确保通过所有单元测试 (`pytest tests/`)。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。