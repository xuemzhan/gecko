# 🦎 Gecko Agent Framework (v0.4.0)

> **工业级、异步优先、协议驱动的 Python AI 智能体开发框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![AsyncIO](https://img.shields.io/badge/Async-AnyIO-green.svg)](https://anyio.readthedocs.io/)
[![Observability](https://img.shields.io/badge/Otel-OpenTelemetry-purple.svg)](https://opentelemetry.io/)

**Gecko v0.4.0** 是在 v0.3.1 基础上的一次 **稳定性与并发能力增强版本**：

- ✅ 新一代 **并发 Workflow 引擎**：支持 DAG 分层并行执行、Copy-On-Write 状态隔离、动态跳转与断点恢复:contentReference[oaicite:0]{index=0}  
- ✅ **Team 多智能体并行引擎升级**：Race 模式原子获胜保护、全员失败可观测结果、超时控制与输入分片 (Sharding) 支持  
- ✅ **P0 级 Bug 全面修复**：Next 指令语义、条件跳过节点、history 清理策略等通过专门测试用例锁死回归  
- ✅ 新增 **高并发演示示例**：展示 Workflow + Team + SummaryMemory 的真实并发行为与锁机制:contentReference[oaicite:3]{index=3}  
- ✅ 继续维持 **v1.0 核心 API 稳定层 (L1)**：`Agent / AgentBuilder / Message / TokenMemory / StructureEngine / Workflow / Team / step / Next` 统一从 `gecko` 顶层导出:contentReference[oaicite:4]{index=4}  

Gecko 专为构建**生产环境**下的高并发、长流程 AI 应用而生。

---

## 🌟 v0.4.0 相比 v0.3.1 的关键改动

### 1. 并发 Workflow 引擎 2.0

v0.4.0 对 Workflow 做了核心重构，引入 **执行层 (Execution Layers)** 与 **Copy-On-Write 状态模型**：:contentReference[oaicite:5]{index=5}  

- 使用拓扑排序将 DAG 拆分为多层 `{node_set}`，每一层内部节点可并行执行。  
- 每个节点在执行时获得：
  - 共享的 `history`（避免深拷贝巨大上下文）；
  - 基于 `_COWDict` 的 **写时复制 state**，保证并发节点互不污染，并在层级结束后通过 diff 合并回主上下文。  
- 引擎支持：
  - `Next` 指令在执行过程中打断静态计划，动态跳转到任意节点；
  - `execute(..., start_node=..., _resume_context=...)`，方便实现断点恢复与自定义入口；
  - `timeout` 超时保护，防止长时间卡死整个工作流。  

同时新增 `_cleanup_history` 策略，对 `history` 做定期瘦身，在保留 `last_output` 的前提下限制历史长度，避免长流程场景内存无界增长。:contentReference[oaicite:6]{index=6}  

### 2. Team 多智能体并行引擎增强

在 v0.4.0 中，`Team` 作为多智能体/多任务并行执行引擎得到系统化增强：  

- **ExecutionStrategy.ALL / RACE**：
  - `ALL`：等待所有成员完成，支持 `max_concurrent` 控制并发度；
  - `RACE`：赛马模式，首个成功结果获胜，其余任务通过 `CancelScope` 取消。
- **Race 原子获胜修复 (P0-1)**：
  - 引入 `_winner_lock: anyio.Lock`，保证在极端并发场景下只有一个成员能成为“赢家”，避免竞争条件。  
- **全员失败可观测 (P0-2)**：
  - 若 Race 模式下所有成员都失败，不再返回空列表，而是返回每个成员的 `MemberResult`（`is_success=False` 且带 `error`），方便上层聚合与监控。  
- **超时控制**：
  - `Team.run(..., timeout=...)` 支持 ALL/RACE 模式下的整体超时，使用 `anyio.move_on_after` 包裹任务组；超时后返回已完成结果或抛出错误。  
- **输入分片 (Sharding)**：
  - 新增 `input_mapper(raw_input, idx)`，可针对每个成员自定义输入切分逻辑，用于分页检索、多段文本切分等高并发场景。  
- **结果模型标准化**：
  - 所有成员结果统一为 `MemberResult[result, error, member_index, is_success]`，并提供 `value` 便捷属性。  

### 3. Next 指令与条件分支语义完善

围绕 `Next` 和条件边，v0.4.0 做了多项语义修复：  

- **Next.input=None 语义 (P0-3)**：  
  - 当节点返回 `Next(node="X", input=None)` 时，`Workflow` 不再用 `None` 覆盖 `last_output`，而是保持之前的输出，真正实现“只跳转、不改输入”的 pass-through 语义。
- **条件跳过节点不污染 history (P0-4)**：  
  - 引入 `NodeStatus.SKIPPED`，条件不满足的节点会被标记为跳过，并不会写入 `history`，保证下游节点只看到实际执行过的历史。  
- 条件函数支持同步/异步形式，异常视为条件失败（Fail Safe），保证执行稳定性。  

### 4. 并发与内存安全验证测试

v0.4.0 引入多组专门测试用例，覆盖并发语义与 P0 Bug 修复：  

- `test_cow_state_is_per_node_and_history_shared`：验证并行层内部每个节点有独立的 state（不同 id），共享 history（相同 id），并在层结束后将键值合并回主上下文。  
- `TestP0_RaceBehavior` / `TestP0_NextAndHistory` / `TestP0_SkippedNodes`：  
  - Race 获胜原子性、全员失败返回结构化错误；
  - Next.input=None 语义保留 `last_output`；
  - 条件跳过节点不污染 history。  
- `test_team_input_sharding` / `test_team_race_strategy` / `test_team_race_all_fail`：  
  - 验证输入分片、Race 模式性能与全失败场景行为。  

这些测试确保 v0.4.0 在高并发与复杂流控场景下具有 **可依赖的语义与行为边界**。

### 5. 新增/更新示例

v0.4.0 在 `examples/` 下新增和更新了多份示例代码：:contentReference[oaicite:10]{index=10}  

- `advanced/concurrent_workflow_demo.py`：  
  - 使用 `Workflow + Team + SummaryTokenMemory` 模拟多 Agent 并发访问同一记忆体，验证 `_summary_lock` 在高并发下只触发一次真实摘要调用，其余协程等待并复用结果。  
- `compose/team_demo.py`：  
  - 使用新版 `Team` 与 `MemberResult` 编写“专家评审团”，展示部分失败容错与结果聚合。  
- `compose/workflow_demo.py` / `compose/workflow_dag_demo.py`：  
  - 更新为使用 v0.4 Workflow，引入 `allow_cycles`、`Next.update_state` 等特性，演示循环与动态跳转。  

---

## 📦 安装

```bash
# 基础安装
pip install gecko-ai

# 安装 RAG 支持 (包含向量库依赖)
pip install "gecko-ai[rag]"

# 安装所有功能 (Redis, Otel, YAML, etc.)
pip install "gecko-ai[all]"
````

---

## ⚡️ 快速开始

### 1. 构建 RAG 增强的 Agent（与 v0.3.1 兼容）

v0.4.0 完全兼容 v0.3.1 中的 RAG 能力，包括 `IngestionPipeline`、`RetrievalTool`、Chroma/LanceDB 向量库等。

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
        config=ModelConfig(
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        dimension=1536,
    )

    # 2. 知识入库 (仅需运行一次)
    # pipeline = IngestionPipeline(vector_store, embedder)
    # await pipeline.run(["company_policy.pdf", "api_docs.md"])

    # 3. 创建检索工具
    rag_tool = RetrievalTool(
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
    )

    # 4. 构建 Agent
    llm = ZhipuChat(
        api_key=os.getenv("ZHIPU_API_KEY"),
        model="glm-4-flash",
    )
    agent = (
        AgentBuilder()
        .with_model(llm)
        .with_tools([rag_tool])
        .with_system_prompt("你是一个助手，请优先查阅知识库回答问题。")
        .build()
    )

    # 5. 提问
    response = await agent.run("公司的报销政策是怎样的？")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 使用 Workflow 构建可恢复 & 并发流程

v0.4.0 的 Workflow 具备并发执行与断点恢复能力，基本用法保持与 v0.3.1 一致，只是内部执行模型更强大：

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
    return "Finish"

async def main():
    storage = await create_storage("sqlite:///./workflow.db")
    
    wf = Workflow(
        "MyFlow",
        storage=storage,
        checkpoint_strategy=CheckpointStrategy.ALWAYS,  # 每步保存
        enable_parallel=True,
    )
    
    wf.add_node("A", step_one)
    wf.add_node("B", step_two)
    wf.add_edge("A", "B")
    wf.set_entry_point("A")
    
    session_id = "uniq_session_id"
    
    # 首次执行（或发生 Crash 后）始终可以通过 execute / resume 恢复
    res = await wf.execute("init_input", session_id=session_id)
    print("Result:", res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3. 使用 Team 构建“专家评审团”

```python
import asyncio
import os
from typing import List
from gecko.compose.team import Team, MemberResult
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.plugins.models.presets.zhipu import ZhipuChat

def create_expert(role: str, prompt: str, api_key: str) -> Agent:
    model = ZhipuChat(api_key=api_key, model="glm-4-air", temperature=0.8)
    return (
        AgentBuilder()
        .with_model(model)
        .with_session_id(f"expert_{role}")
        .with_system_prompt(f"你是一位{role}。{prompt} 请简短回答（50字以内）。")
        .build()
    )

async def aggregate_results(results: List[MemberResult]) -> str:
    lines = []
    for res in results:
        idx = res.member_index + 1
        if res.is_success:
            lines.append(f"专家 {idx}: {res.result}")
        else:
            lines.append(f"专家 {idx}: [缺席] (Error: {res.error})")
    return "\n".join(lines)

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("请设置 ZHIPU_API_KEY")
        return

    optimist = create_expert("乐观主义未来学家", "请对未来的 AI 发展给出一个极其乐观的预测。", api_key)
    pessimist = create_expert("悲观主义安全专家", "请警告人类 AI 可能带来的最大生存风险。", api_key)
    realist = create_expert("务实工程师", "请从技术落地角度评估未来 5 年 AI 的实际应用。", api_key)

    team = Team(
        members=[optimist, pessimist, realist],
        name="AI_Review_Board",
        max_concurrent=2,
    )

    topic = "我们应该如何看待 AGI 的到来？"
    results = await team.run(topic)
    print(await aggregate_results(results))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏗️ 核心架构（v0.4.0）

在 v0.4.0 中，架构继续沿用 v0.3.1 的分层设计，并强化了 Compose 层的并发执行能力：

| 层级          | 模块           | 功能描述                                                       |
| :---------- | :----------- | :--------------------------------------------------------- |
| **Compose** | `Workflow`   | DAG 编排，支持并行层级执行、条件分支、循环、动态跳转、状态持久化与断点恢复                    |
|             | `Team`       | 多智能体并行执行，支持 ALL/RACE 策略、并发控制、超时、输入分片                       |
|             | `nodes`      | `step` 装饰器（统一同步/异步函数），`Next` 控制流指令（含 `update_state`）       |
| **Core**    | `Agent`      | 智能体门面，组装 Model / Memory / Tools                            |
|             | `Engine`     | ReAct 推理循环，流式缓冲，死循环熔断                                      |
|             | `Memory`     | `TokenMemory` (LRU缓存), `SummaryTokenMemory` (异步摘要 & 并发锁保护) |
|             | `Structure`  | 结构化输出解析，Schema 生成，多策略解析与自动修复                               |
|             | `Prompt`     | 模板管理，组合器 (Composer)，注册表 (Registry)，静态验证                    |
| **Support** | `ToolBox`    | 工具注册与执行，并发控制，参数校验                                          |
|             | `Events`     | 异步事件总线，支持中间件拦截                                             |
|             | `Telemetry`  | OpenTelemetry 链路追踪，Context 传播                              |
| **Plugins** | `Models`     | 基于 LiteLLM 适配 OpenAI, Zhipu, Ollama 等                      |
|             | `Storage`    | SQLite (FileLock), Redis, ChromaDB, LanceDB                |
|             | `Knowledge`  | RAG 流水线，文档加载、切分、向量化                                        |
|             | `Guardrails` | 输入清洗，Prompt Injection 防御                                   |

顶层包 `gecko/__init__.py` 暴露了 v1.0 核心稳定 API（L1）：

```python
from gecko import (
    __version__,
    Agent,
    AgentBuilder,
    Message,
    Role,
    AgentOutput,
    TokenUsage,
    TokenMemory,
    SummaryTokenMemory,
    StructureEngine,
    Workflow,
    step,
    Next,
    Team,
)
```

---

## 🔌 存储后端矩阵

存储层继续复用 v0.3.1 的 URL Scheme 设计：

| Scheme       | 后端       | 类型     | 用途                 | 特性                 |
| :----------- | :------- | :----- | :----------------- | :----------------- |
| `sqlite://`  | SQLite   | KV     | Session / Workflow | WAL 模式，跨进程文件锁，无依赖  |
| `redis://`   | Redis    | KV     | Session / Cache    | 高性能，TTL 支持，分布式锁    |
| `chroma://`  | ChromaDB | Vector | RAG                | 元数据过滤，本地/远程模式      |
| `lancedb://` | LanceDB  | Vector | RAG                | 基于 Arrow 的高性能文件向量库 |

---

## 🛣️ 版本演进

> 以下是从 v0.1 到 v0.4 的能力演进脉络：

* **v0.1**

  * 基础 ReAct 引擎与工具箱。
* **v0.2**

  * 引入 Workflow DAG，断点恢复，SQLite/Redis 存储插件。
* **v0.3**

  * ✅ **RAG**：Knowledge Plugin (Ingestion/Retrieval)。
  * ✅ **Refactor**：Prompt / Structure / Output 模块化重构。
  * ✅ **Observability**：OpenTelemetry 集成。
  * ✅ **Safety**：Guardrails 输入清洗。
* **v0.4 (Current)**

  * ✅ 并发 Workflow 引擎：执行层级 + Copy-On-Write 状态管理 + 动态跳转 + history 清理。
  * ✅ Team 并行引擎优化：Race 原子获胜、全员失败可观测、超时控制、输入分片。
  * ✅ P0 Bugfix 测试矩阵：覆盖 Next 语义、条件跳过、COW 行为、并发 Race 行为。
  * ✅ 新示例：并发 Workflow Demo、多专家 Team Demo 等。

（后续版本规划会在独立的 Roadmap 文档中给出）

---

## 🤝 贡献

Gecko 是一个开源项目，欢迎通过 Issue 或 Pull Request 参与贡献。
如果你在使用过程中遇到问题、发现 Bug，或希望补充新的示例与存储/模型插件，都非常期待你的反馈 🙌

```text
1. Fork 本仓库
2. 创建你的特性分支: git checkout -b feature/my-awesome-feature
3. 提交变更: git commit -am 'Add some feature'
4. 推送分支: git push origin feature/my-awesome-feature
5. 提交 Pull Request
```

---

> 如果你正在从 v0.3.1 升级到 v0.4.0：
>
> * 绝大多数 API 完全兼容；
> * Workflow & Team 的内部实现更强，但对外接口保持稳定；
> * 如需利用新的并发能力，建议阅读 `examples/compose/` 与 `examples/advanced/` 下的最新示例。
