# 工作流编排与恢复

当单一 Agent 无法满足复杂业务逻辑时，`Workflow` 提供了基于 DAG (有向无环图) 的编排能力。Gecko Workflow 的核心特性是 **Resumability (可恢复性)**。

## 基础编排

Workflow 由 **Node (节点)** 和 **Edge (边)** 组成。使用 `@step` 装饰器定义节点。

```python
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.compose.nodes import step, Next

@step("CheckInput")
async def check_input(query: str, context: WorkflowContext):
    # 根据输入内容决定下一步
    if "urgent" in query:
        return Next("FastTrack", input=query)
    return Next("NormalTrack", input=query)

@step("FastTrack")
async def fast_track(query: str):
    return f"⚡️ 处理加急请求: {query}"

@step("NormalTrack")
async def normal_track(query: str):
    return f"🐢 处理普通请求: {query}"

# 定义图
wf = Workflow("RequestRouter")
wf.add_node("CheckInput", check_input)
wf.add_node("FastTrack", fast_track)
wf.add_node("NormalTrack", normal_track)
wf.set_entry_point("CheckInput")
```

## 状态持久化与断点恢复 (Resume)

在生产环境中，服务器可能会重启或崩溃。Gecko 允许你配置持久化存储，使得 Workflow 可以从中断的节点继续执行。

### 1. 配置存储与策略

要启用恢复功能，必须提供 `storage` 并建议将策略设为 `ALWAYS`。

```python
from gecko.compose.workflow import CheckpointStrategy
from gecko.plugins.storage.factory import create_storage

# 使用 SQLite 持久化
storage = await create_storage("sqlite:///./workflow_state.db")

wf = Workflow(
    name="PaymentFlow", 
    storage=storage,
    # ALWAYS: 每执行完一个节点就保存一次快照 (最安全)
    checkpoint_strategy=CheckpointStrategy.ALWAYS
)
```

### 2. 执行与恢复

```python
session_id = "uniq_order_id_1001"

try:
    # 首次执行
    await wf.execute("user input", session_id=session_id)
except Exception:
    print("系统崩溃！正在尝试恢复...")
    
    # --- 模拟重启后 ---
    
    # 调用 resume() 而不是 execute()
    # Gecko 会从数据库加载该 session 的状态，跳过已完成的节点
    # 直接从上次失败或未执行的节点开始重试
    result = await wf.resume(session_id=session_id)
    print("恢复执行结果:", result)
```

## 并行与循环

*   **并行**: 使用 `Team` 节点可以实现多个 Agent 并行工作（Map-Reduce 模式）。
*   **循环**: 设置 `Workflow(allow_cycles=True)` 并在节点中返回指向前序节点的 `Next` 指令即可实现循环（如：审批不通过 -> 重写 -> 审批）。