# Gecko Compose 编排引擎

`gecko.compose` 是 Gecko 框架的核心编排层，旨在构建复杂的 AI 应用。它提供了一套轻量级、异步优先（Async-first）的引擎，支持从简单的顺序链到复杂的自修正循环、并行协作以及生产级的断点恢复。

## ✨ 核心特性

*   **图编排 (Graph Orchestration)**: 支持 DAG（有向无环图）和 Cyclic Graph（有环图），轻松实现 ReAct、Reflexion 等高级模式。
*   **并行协作 (Team)**: 基于 `anyio` 的高效并行执行引擎，支持多 Agent 投票、赛马机制，内置并发限流与容错。
*   **生产级运维 (Ops-Ready)**:
    *   **断点恢复 (Resumability)**: 系统崩溃后，可从断点处恢复执行，不重跑已完成的步骤。
    *   **状态持久化**: 支持细粒度的状态保存策略（每步保存或结束保存）。
*   **类型安全**: 摒弃隐式魔法，提供强类型的上下文访问和结果封装。

## 📦 快速开始

### 1. 基础工作流

```python
import asyncio
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.compose.nodes import step

@step("Step1")
async def analyze(input_str: str):
    return f"Analyzed: {input_str}"

@step("Step2")
def summarize(context: WorkflowContext):
    # 类型安全地获取上一步输出
    prev = context.get_last_output_as(str)
    return f"Summary of [{prev}]"

async def main():
    wf = Workflow("SimpleFlow")
    wf.add_node("A", analyze)
    wf.add_node("B", summarize)
    
    wf.add_edge("A", "B")
    wf.set_entry_point("A")
    
    result = await wf.execute("Hello Gecko")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧩 核心组件

### 1. Workflow (工作流)

Workflow 是一个状态机容器，管理节点（Node）的执行顺序和上下文（Context）。

#### 显式循环与分支
V0.2+ 版本支持显式定义循环结构。

```python
# 允许定义有环图
wf = Workflow("LoopFlow", allow_cycles=True, max_steps=10)

# 定义分支条件
wf.add_edge("Analyze", "QuickReply", lambda ctx: len(ctx.input) < 5)
wf.add_edge("Analyze", "DeepThink", lambda ctx: len(ctx.input) >= 5)

# 定义循环 (DeepThink -> Check -> DeepThink)
wf.add_edge("Check", "DeepThink", lambda ctx: "Error" in ctx.get_last_output())
```

### 2. Nodes & Control Flow (节点与控制流)

使用 `@step` 装饰器定义节点。节点可以返回 `Next` 指令来动态控制流转。

```python
from gecko.compose.nodes import step, Next

@step("CheckResult")
async def check_quality(context: WorkflowContext):
    score = context.state.get("score", 0)
    
    if score < 60:
        # 动态跳转，并更新上下文状态
        return Next(
            node="Rewrite", 
            input="Please rewrite better.",
            update_state={"retry_count": context.state.get("retry_count", 0) + 1}
        )
    
    return "Quality Pass"
```

### 3. Team (并行协作组)

`Team` 用于并行执行多个任务（Agent 或 函数），并聚合结果。

> **⚠️ Breaking Change (V0.2)**: `Team.run` 现在返回 `List[MemberResult]` 对象，而不是混合了错误字符串的列表。请务必更新您的代码以适配新结构。

```python
from gecko.compose.team import Team, MemberResult

# 定义团队
team = Team(members=[agent_a, agent_b, agent_c], max_concurrent=2)

# 执行
results: List[MemberResult] = await team.run("Topic")

# 处理结果
for res in results:
    if res.is_success:
        print(f"✅ Member {res.member_index}: {res.result}")
    else:
        print(f"❌ Member {res.member_index} Failed: {res.error}")
```

---

## 🛡️ 健壮性与断点恢复 (New in V0.2)

Gecko 提供了生产级的状态管理能力，确保持久运行的任务不会因意外中断而前功尽弃。

### 持久化策略 (Checkpoint Strategy)

```python
from gecko.compose.workflow import CheckpointStrategy

wf = Workflow(
    name="LongRunningTask",
    storage=sqlite_storage,
    # 策略选项:
    # ALWAYS: 每执行完一个节点保存一次 (推荐，支持 Resume)
    # FINAL:  仅在工作流结束时保存 (高性能，不支持过程 Resume)
    # MANUAL: 不自动保存
    checkpoint_strategy=CheckpointStrategy.ALWAYS
)
```

### 断点恢复 (Resume)

当系统崩溃或重启后，可以使用 `resume` 方法从上次中断的地方继续执行。

```python
try:
    # 首次运行
    await wf.execute("Start Data", session_id="session_001")
except Exception:
    # 发生崩溃...
    pass

# ... 重启系统后 ...

# 恢复执行
# 引擎会自动加载 session_001 的状态，跳过已完成的节点，从中断点继续
final_result = await wf.resume(session_id="session_001")
```

---

## 📚 API 变更指南 (V0.1 -> V0.2)

如果您从旧版本升级，请注意以下变更：

1.  **移除隐式数据拆包**:
    *   **旧行为**: 如果节点返回 `{"content": "text", "meta": 1}`，下游节点会自动收到 `"text"`。
    *   **新行为**: 下游节点将收到完整的字典 `{"content": "text", "meta": 1}`。请在下游节点中使用 `context.get_last_output()["content"]` 显式获取。

2.  **Team 返回值变更**:
    *   **旧行为**: 返回 `["result", "Error: boom"]`。
    *   **新行为**: 返回 `[MemberResult(result="result"), MemberResult(error="boom", is_success=False)]`。

3.  **上下文类型安全**:
    *   推荐使用 `context.get_last_output_as(Type)` 来替代直接属性访问，以获得更好的类型提示和运行时检查。

## 🌟 最佳实践

1.  **总是配置 Storage**: 即使在开发环境，也建议配置 SQLite Storage，这将极大方便调试和状态回溯。
2.  **显式优于隐式**: 在节点间传递复杂数据时，建议使用 Pydantic 模型，并通过 `Next(input=model)` 传递，避免使用无结构的字典。
3.  **控制循环深度**: 开启 `allow_cycles=True` 时，务必在逻辑中设置退出条件（如重试次数限制），`max_steps` 是最后的安全网。