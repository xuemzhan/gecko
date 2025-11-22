# 编排与工作流

当单一 Agent 无法满足需求时，你需要使用 `Workflow` 将多个节点编排成有向无环图 (DAG)。

## Workflow 基础

Workflow 由 **Nodes (节点)** 和 **Edges (边)** 组成。

*   **Node**: 可以是普通函数、Agent 或 Team。
*   **Edge**: 定义节点间的流转，支持条件分支。

```python
from gecko.compose.workflow import Workflow
from gecko.compose.nodes import step, Next

@step("Check")
async def check_input(ctx):
    if len(ctx.input) > 10:
        return Next(node="ProcessLong")
    return Next(node="ProcessShort")

# 定义工作流
wf = Workflow("DemoFlow")
wf.add_node("Check", check_input)
# ... 添加其他节点 ...
wf.set_entry_point("Check")
```

## 状态持久化与断点恢复 (Resumability)

Gecko 支持在系统崩溃或重启后恢复工作流的执行状态。

### Checkpoint 策略
通过 `checkpoint_strategy` 控制保存频率：
*   `ALWAYS`: 每执行完一个节点就保存（最安全，默认）。
*   `FINAL`: 仅在工作流结束时保存。
*   `MANUAL`: 不自动保存。

### 如何恢复 (Resume)

```python
from gecko.plugins.storage.factory import create_storage

# 1. 必须配置持久化存储
storage = await create_storage("sqlite:///./workflow.db")
wf = Workflow(name="ResumableFlow", storage=storage)

# 2. 首次运行 (可能会崩溃)
try:
    await wf.execute(input_data, session_id="user_123")
except Exception:
    print("系统崩溃！")

# 3. 恢复运行
# Gecko 会自动加载 user_123 的状态，跳过已完成的节点
result = await wf.resume(session_id="user_123")
```

## Team (多智能体并行)

`Team` 引擎实现了 Map-Reduce 模式，用于并发执行任务。

```python
from gecko.compose.team import Team

# 创建评审团
team = Team(
    members=[agent_coder, agent_reviewer, agent_manager],
    max_concurrent=2  # 限制并发数，防止 API Rate Limit
)

results = await team.run("评审这段代码...")
```