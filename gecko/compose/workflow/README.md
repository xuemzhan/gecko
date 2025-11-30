# Gecko Compose: Workflow Engine (v0.3)

Gecko Workflow 是一个生产级、基于 DAG（有向无环图）的智能体编排引擎。v0.3 版本经过深度重构，采用了模块化架构，专注于**高并发**、**低 I/O 开销**和**断点续传**能力。

## 🌟 核心特性 (v0.3 新增)

*   **模块化架构**：将核心逻辑拆分为 `Graph`（拓扑）、`Executor`（执行）、`Persistence`（存储）和 `Models`（数据），解耦清晰。
*   **Context Slimming (上下文瘦身)**：
    *   在持久化时自动剥离监控数据（Traces）并裁剪冗余历史。
    *   在大规模长流程中，存储体积减少 **80%+**，显著降低 Redis/DB 的 I/O 压力。
*   **两阶段提交 (Two-Phase Commit)**：
    *   节点执行前保存 `RUNNING` 状态，执行后保存 `SUCCESS` 状态。
    *   确保即使系统在节点执行中崩溃，也能精确定位故障点并恢复。
*   **高级并行策略 (Team)**：
    *   支持 **Race (赛马模式)**：多个 Agent 并发执行，取最快结果，自动取消其他任务。
    *   支持 **Sharding (输入分片)**：支持 Map-Reduce 模式的大任务拆解。
*   **向后兼容**：保持 v0.2 的对外 API 签名不变，现有代码可无缝升级。

---

## 📂 目录结构

```text
gecko/compose/workflow/
├── __init__.py          # 外观接口 (Facade)，保持 API 兼容性
├── models.py            # 数据模型 (Context, NodeExecution, Slimming 逻辑)
├── graph.py             # DAG 拓扑管理 (节点, 边, 环检测, 层级构建)
├── executor.py          # 节点调度器 (参数注入, 重试机制, 结果标准化)
├── persistence.py       # 持久化管理器 (异步 IO, 序列化清洗)
└── engine.py            # 核心引擎 (主循环, 断点恢复)
```

---

## 🚀 快速开始

### 1. 基础线性工作流

```python
import asyncio
from gecko.compose import Workflow, step

# 定义节点 (可以是普通函数或异步函数)
@step(name="Step1")
def generate_number(seed: int):
    return seed * 10

@step(name="Step2")
async def process_number(num: int):
    await asyncio.sleep(0.1)
    return f"Processed: {num}"

async def main():
    # 初始化工作流
    wf = Workflow("SimpleFlow")
    
    # 构建 DAG
    wf.add_node("gen", generate_number)
    wf.add_node("proc", process_number)
    wf.add_edge("gen", "proc")
    wf.set_entry_point("gen")
    
    # 执行
    result = await wf.execute(input_data=5)
    print(result)  # Output: "Processed: 50"

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 带分支与控制流 (Next)

```python
from gecko.compose.nodes import Next

def router(score: int):
    if score >= 60:
        # 动态跳转并传递新输入
        return Next(node="Pass", input=f"Score {score} is good")
    else:
        return Next(node="Fail", input=f"Score {score} is bad")

wf.add_node("Check", router)
wf.add_node("Pass", lambda x: f"Congratz: {x}")
wf.add_node("Fail", lambda x: f"Retry: {x}")
# ...
```

---

## ⚙️ 生产级配置指南

### 1. 启用上下文瘦身 (Context Slimming)

在处理长流程（如 50+ 步骤）时，必须开启此功能以防止 Context 爆炸。

```python
from gecko.compose import Workflow, CheckpointStrategy

wf = Workflow(
    name="LongProcess",
    storage=redis_storage,
    # 策略: ALWAYS (每步保存) / FINAL (仅结束保存) / MANUAL
    checkpoint_strategy=CheckpointStrategy.ALWAYS,
    # [关键] 仅保留最近 10 步的历史输入输出，旧数据会被从 Checkpoint 中裁剪
    max_history_retention=10 
)
```

### 2. 高级并行执行 (Team Strategies)

利用 `Team` 模块实现并发优化。

```python
from gecko.compose import Team
from gecko.compose.team import ExecutionStrategy

# 场景：赛马模式 (降低长尾延迟)
# 同时请求 3 个模型，谁先返回用谁的，其他的自动 Cancel
fast_team = Team(
    members=[gpt4, claude3, llama3],
    strategy=ExecutionStrategy.RACE
)

wf.add_node("FastResponse", fast_team)

# 场景：Map-Reduce (文档分片处理)
def page_splitter(doc, index):
    return doc.pages[index] # 将大文档切分给不同的 Worker

map_team = Team(
    members=[agent_worker] * 5, # 5 个并发 Worker
    input_mapper=page_splitter, # 输入分片逻辑
    strategy=ExecutionStrategy.ALL
)

wf.add_node("ProcessPages", map_team)
```

### 3. 断点续传 (Resume)

当工作流因异常（如 API 超时、进程崩溃）中断时，可从最后一次成功的节点恢复。

```python
try:
    await wf.execute(data, session_id="session_123")
except Exception:
    # 稍后重试...
    # 自动加载上次的状态，并在失败的节点（或 Next 指向的节点）继续
    result = await wf.resume(session_id="session_123")
```

---

## 🔧 架构与调试

### 智能参数绑定 (Smart Binding)

节点函数支持灵活的参数签名，`Executor` 会自动注入所需对象：

| 参数名 | 注入内容 | 说明 |
| :--- | :--- | :--- |
| `context` | `WorkflowContext` | 完整的上下文对象（读写 State/History） |
| `workflow_context` | `WorkflowContext` | 同上（兼容旧版本命名） |
| *其他参数* | `input` | 上一个节点的输出（或 Next 传递的值） |

### 可视化

```python
# 打印 Mermaid 流程图代码
print(wf.to_mermaid())
```

---

## ⚠️ 常见问题 (FAQ)

**Q: 升级到 v0.3 后，旧的持久化数据还能读取吗？**
A: **可以**。`WorkflowContext.from_storage_payload` 包含兼容逻辑，可以自动补全旧版本数据中缺失的 `executions` 等字段。

**Q: 开启 Context Slimming 后，我无法访问很久以前的历史数据了吗？**
A: 在 `Checkpoint`（存储层）中无法访问，但在**内存运行态**中仍然可以访问全量历史。只有当进程重启并调用 `resume()` 时，过久的历史才会丢失。如果业务逻辑强依赖第一步的输入，建议将其显式存入 `context.state`（`state` 永远全量保存）。

**Q: 如何处理不可序列化的对象（如数据库连接、锁）？**
A: `PersistenceManager` 会在保存前自动扫描并清洗 `Context`。不可序列化的对象会被替换为 `{"__gecko_unserializable__": True}` 标记，防止程序崩溃，但该对象在 `resume` 后无法恢复。请勿将此类对象存入 `state`。

---

## 📅 版本历史

*   **v0.3.0**: 模块化重构；引入 Context Slimming；增强 Team 并行策略。
*   **v0.2.0**: 引入 DAG 支持；基础持久化。
*   **v0.1.0**: 简单的 Chain 模式。