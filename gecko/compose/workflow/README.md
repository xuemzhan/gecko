# Gecko Compose: Workflow Engine (v0.5)

Gecko Compose Workflow 引擎是一个生产级、基于 DAG（有向无环图）的智能体编排引擎。

**v0.5 版本 (Released: 2025-12-28)** 是一个里程碑式的**稳定版本**，专注于解决并发安全、状态一致性与高负载下的系统鲁棒性。通过引入严格的状态隔离机制（COW with Copy-on-Read）和异步资源管理策略，v0.5 彻底消除了前序版本中的数据竞争隐患，并大幅提升了 I/O 密集型任务的吞吐量。

---

## 🌟 核心特性与 v0.5 关键升级

### 1. 🛡️ 严格的并发状态隔离 (Fix P0)
在此版本中，我们彻底重构了状态管理逻辑，解决了并行执行时的“脏写”与“幽灵读”问题。
*   **Copy-on-Read (读取即隔离)**：并行节点在读取 `state` 中的可变对象（如 List/Dict）时，会自动触发深拷贝（DeepCopy）。这意味着节点 A 对列表的修改绝不会污染节点 B 的视图。
*   **墓碑机制 (Tombstone Deletion)**：支持在并行分支中执行 `del context.state["key"]`。删除操作会被记录为墓碑，并在层级合并时正确地从全局状态中移除该键。
*   **独立模块**：状态管理逻辑已独立拆分为 `gecko.compose.workflow.state` 模块。

### 2. ⚡ 高性能异步调度 (Fix P1/P2)
针对 Python `asyncio` 的单线程特性进行了深度优化，防止 Event Loop 被阻塞。
*   **同步函数自动卸载**：`NodeExecutor` 会智能识别普通的同步函数（如使用 `requests` 或 CPU 密集型计算），并自动将其卸载到 Worker 线程池中执行。主线程仅负责调度，确保高并发下的心跳稳定。
*   **异步序列化**：持久化过程中的大对象序列化（Pydantic Dump）和数据清洗操作也已移入线程池，消除了大 Context 保存时的瞬时卡顿。

### 3. 🧩 增强的组合能力
*   **Team 控制流穿透**：修复了 `Team`（多智能体组）无法控制工作流走向的问题。现在，如果 Team 中的某个 Router Agent 返回了 `Next` 指令，引擎能够递归解包并正确执行跳转。
*   **故障隔舱 (Bulkheading)**：`Team` 的输入映射（Input Mapper）现在具备容错能力。单个成员的映射失败只会导致该成员标记为 Failed，而不会炸毁整个 Team 的执行。

### 4. 💾 智能持久化与恢复
*   **Context Slimming (上下文瘦身)**：在持久化时自动剥离监控数据（Traces）并裁剪冗余历史，存储体积减少 **80%+**。
*   **长链路依赖保障**：移除了运行时内存中的强制历史清理，确保存储在内存中的 DAG 长链路依赖（如 Step 50 读取 Step 1 的输出）始终可用，仅在落盘时进行裁剪。

---

## 🔧 快速开始

### 1. 定义工作流

```python
from gecko.compose import Workflow, step, Next

# 使用 @step 装饰器 (支持 sync 和 async)
@step(name="Fetcher")
def fetch_data(url: str):
    # 同步阻塞 IO 会自动卸载到线程池，不卡 Loop
    import requests
    return requests.get(url).json()

@step(name="Analyzer")
async def analyze(data: dict):
    if data["status"] != "ok":
        return Next(node="ErrorHandler", input="Data invalid")
    return {"result": "pass"}

wf = Workflow(name="DataPipeline")
wf.add_node("Fetcher", fetch_data)
wf.add_node("Analyzer", analyze)
wf.add_node("ErrorHandler", lambda x: print(f"Error: {x}"))

wf.add_edge("Fetcher", "Analyzer")
wf.set_entry_point("Fetcher")
```

### 2. 执行与并发

```python
# 自动并行：如果图中存在分支，引擎会自动并行调度
# timeout: 全局超时保护 (秒)
result = await wf.execute(
    input_data="http://api.example.com", 
    session_id="sess_001", 
    timeout=30.0
)
```

### 3. 多智能体协作 (Team)

```python
from gecko.compose import Team, ExecutionStrategy

# Race 模式：多个模型竞争，取最快结果
# 具备原子性锁保护，确保 Winner 唯一
team = Team(
    members=[gpt4_agent, claude_agent, local_model],
    strategy=ExecutionStrategy.RACE
)

# Team 可直接嵌入 Workflow 作为一个节点
wf.add_node("ReasoningCluster", team)
```

---

## 📊 性能基准 (v0.5 Benchmarks)

基于 `benchmarks/compose_cow_benchmark.py` 的测试结果：

| 场景 (Nodes) | v0.4 (ms) | **v0.5 (ms)** | 提升 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Small (50)** | 82.1 | **5.2** | **15.8x** | 得益于 Copy-on-Read 避免了全量深拷贝 |
| **Medium (200)** | 262.0 | **18.3** | **14.3x** | 大幅降低内存分配开销 |
| **Large (500)** | 311.5 | **47.5** | **6.6x** | 在大规模 DAG 中保持线性增长 |

> **注**：v0.5 的 Copy-on-Read 策略意味着只有在节点真正修改数据时才付出拷贝成本，对于只读场景（Read-Heavy），性能接近原生引用传递。

---

## ✅ 升级指南 (v0.4 -> v0.5)

**本次升级包含底层行为变更，建议进行回归测试。**

1.  **检查可变对象的使用**：
    *   如果您依赖并行节点 A 修改 `list`，且希望节点 B 立即看到该修改（Side Effect），这在 v0.5 中将**不再生效**（被隔离了）。请改用 `return` 值传递或 Redis 共享存储。
2.  **同步函数无需 `async` 包装**：
    *   以前为了不阻塞 Loop，您可能手动将同步函数包装为 `async`。现在可以直接传入普通 `def` 函数，引擎会自动处理。
3.  **Team 返回值**：
    *   `Team` 的返回值始终是 `List[MemberResult]`。如果您之前依赖隐式的解包逻辑，请检查代码。但 Workflow 引擎已能够自动识别 Team 返回列表中的 `Next` 指令。

---

## 📦 模块结构

*   `gecko/compose/workflow/engine.py`: 核心调度器，DAG 遍历与任务分发。
*   `gecko/compose/workflow/state.py`: **[New]** 状态管理，COWDict 实现。
*   `gecko/compose/workflow/executor.py`: 节点执行器，智能参数注入与线程卸载。
*   `gecko/compose/workflow/persistence.py`: 持久化管理，异步序列化。
*   `gecko/compose/team.py`: 多智能体并行/赛马引擎。

---

## 贡献

提交 PR 前请确保所有单元测试通过：
```bash
rye run pytest tests/compose/
```
当前测试覆盖率要求：**100%**。

---
© 2025 Gecko Project