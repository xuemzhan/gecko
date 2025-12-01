# 记忆管理与状态持久化

在 Gecko 中，我们需要区分两个概念：**Memory (短期上下文)** 和 **State (长期状态)**。

## Memory: 上下文窗口管理

Memory 负责管理发送给 LLM 的 `messages` 列表。由于 LLM 的 Context Window 是有限且昂贵的，Memory 必须智能地裁剪历史。

### TokenMemory (滑动窗口)
最基础的策略。
*   **机制**: 实时计算当前历史的 Token 总数 (基于 `tiktoken` 或模型估算)。
*   **裁剪**: 当超出 `max_tokens` 时，保留 System Prompt，从最早的 User/Assistant 消息开始丢弃。
*   **缓存**: 内置 LRU 缓存，避免对同一条消息重复计算 Token。

### SummaryTokenMemory (自动摘要)
适用于长对话场景。
*   **机制**: 当历史消息即将被丢弃时，触发一个后台异步任务，调用 LLM 对即将丢弃的内容生成摘要。
*   **注入**: 摘要会作为一个特殊的 System Message 插入到上下文头部。
*   **并发控制**: 内置防抖 (Debounce) 和锁机制，防止高频触发摘要生成导致 Token 浪费。

---

## State: 工作流状态持久化

当使用 `Workflow` 时，我们需要跨请求、跨重启保存整个流程的状态。

### WorkflowContext
这是状态的载体，包含：
*   `input`: 初始输入。
*   `history`: 节点执行结果的历史记录 (Dict)。
*   `state`: 用户自定义的全局变量 (Dict)。
*   `next_pointer`: 动态跳转指令。

### 状态瘦身 (State Slimming)
在持久化到数据库之前，`WorkflowContext.to_storage_payload` 会执行瘦身操作：
1.  **移除轨迹**: 移除 `executions` 等仅用于调试的庞大轨迹数据。
2.  **裁剪历史**: 仅保留最近 N 步的节点输出 (`workflow_history_retention`)，防止 `state_json` 随运行时间无限膨胀。

### 存储后端
Gecko 支持多种 KV 存储后端用于保存 State：

| 后端 | 特性 | 适用场景 |
| :--- | :--- | :--- |
| **SQLite** | WAL 模式，文件锁 | 单机部署，低运维成本 |
| **Redis** | TTL，高性能 | 分布式部署，高并发 |

### 原子性与一致性
为了防止并发写入导致的状态覆盖，Gecko 实现了 `AtomicWriteMixin`：
*   **协程锁**: 保证单进程内同一 Session 串行写入。
*   **快照**: 在写入 I/O 发生前，先在内存中创建状态深拷贝快照。