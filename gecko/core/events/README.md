# Gecko Core Events

`gecko.core.events` 是 Gecko 框架的异步事件总线系统。它旨在解耦系统中的各个组件（如 Agent、Workflow、Session），使它们能够通过发布/订阅模式进行通信，而无需直接相互引用。

## 🌟 核心特性

*   **异步优先**：基于 Python `asyncio` 构建，专为高并发场景设计。
*   **类型安全**：所有事件均基于 `Pydantic` 模型，提供自动的序列化和类型检查。
*   **灵活的处理器**：同时支持异步 (`async def`) 和同步 (`def`) 事件处理器。
*   **中间件支持**：支持拦截器模式，可用于全局日志记录、鉴权或事件过滤。
*   **后台任务管理**：在 Fire-and-forget 模式下发布事件时，自动追踪后台任务，防止垃圾回收并在关闭时优雅等待。

## 📂 模块结构

| 模块文件 | 类/组件 | 描述 |
| :--- | :--- | :--- |
| **`bus.py`** | `EventBus` | 事件总线的核心实现，负责订阅管理、事件分发和中间件执行。 |
| **`types.py`** | `BaseEvent` | 所有事件的基类，定义了 `type`, `timestamp`, `data` 等标准字段。 |
| **`presets.py`** | `AgentRunEvent`<br>`WorkflowEvent`... | 框架内置的标准事件定义。 |

## 🚀 使用指南

### 1. 定义事件
自定义事件只需继承 `BaseEvent`。由于它是 Pydantic 模型，你可以定义任意的数据字段。

```python
from gecko.core.events import BaseEvent

class UserLoginEvent(BaseEvent):
    type: str = "user.login"
    # data 字段默认是一个 dict，你也可以自定义额外的字段
```

### 2. 订阅事件 (Handlers)
你可以注册异步或同步函数作为处理器。支持通配符 `*` 订阅所有事件。

```python
async def send_welcome_email(event: BaseEvent):
    user_id = event.data.get("user_id")
    print(f"Sending email to {user_id}...")
    await asyncio.sleep(1)

def log_analytics(event: BaseEvent):
    print(f"[Analytics] Event {event.type} occurred at {event.timestamp}")

# 注册
bus = EventBus()
bus.subscribe("user.login", send_welcome_email)
bus.subscribe("*", log_analytics)
```

### 3. 发布事件
支持两种发布模式：**等待模式**（Wait）和**后台模式**（Fire-and-forget）。

```python
event = UserLoginEvent(data={"user_id": 123})

# 模式 A: 等待所有处理器执行完毕 (阻塞当前协程)
await bus.publish(event, wait=True)

# 模式 B: 后台执行 (立即返回，不阻塞)
# 适用于耗时的副作用操作，如发送通知、写日志
await bus.publish(event, wait=False)
```

### 4. 使用中间件 (Middleware)
中间件可以在事件到达处理器之前拦截、修改或阻止事件。

```python
async def filter_middleware(event: BaseEvent):
    # 示例：阻止特定 ID 的事件
    if event.data.get("is_blocked"):
        print(f"Blocked event: {event.type}")
        return None  # 返回 None 表示拦截事件，不再向下传递
    
    # 也可以修改事件数据
    event.data["processed_by"] = "middleware"
    return event

bus.add_middleware(filter_middleware)
```

### 5. 生命周期管理
推荐使用上下文管理器 (`async with`)，它会在退出时自动调用 `shutdown()`，确保所有后台任务执行完毕。

```python
async with EventBus() as bus:
    bus.subscribe("task", handler)
    await bus.publish(event, wait=False)
    # 退出时会自动等待后台 handler 执行完成
```

或者手动控制：

```python
bus = EventBus()
# ... 使用 bus ...
await bus.shutdown(wait=True) # 优雅关闭
```

## 内置事件

Gecko 预定义了一些系统级事件，位于 `presets.py`：

*   **`AgentRunEvent`**: Agent 开始思考、结束思考或发生错误时触发。
*   **`WorkflowEvent`**: 工作流节点状态变更时触发。
*   **`SessionEvent`**: 会话加载、保存或过期时触发。