# Gecko Plugins Tools

`gecko.plugins.tools` 是 Gecko 框架的工具执行层，负责连接 LLM 与外部世界。它提供了一套基于 Pydantic 的强类型工具定义标准，支持自动生成 OpenAI Function Calling Schema，并内置了异步并发执行和安全控制机制。

## ✨ 核心特性

*   **Type-Safe**: 基于 Pydantic V2 定义参数 Schema，自动进行运行时参数校验。
*   **Async-First**: 全异步设计，支持 `ToolBox` 并发批量执行工具调用。
*   **Auto Schema**: 自动生成符合 OpenAI 规范的 `function` 定义 JSON。
*   **Secure**: 内置安全计算器（AST 解析）和防 DoS 机制。
*   **Extensible**: 支持有状态（Stateful）工具和依赖注入。

## 📦 目录结构

```
gecko/plugins/tools/
├── base.py          # BaseTool 抽象基类与 ToolResult 定义
├── registry.py      # 工具注册与发现机制
├── standard/        # 标准工具库
│   ├── calculator.py    # 安全数学计算器
│   └── duckduckgo.py    # 联网搜索 (Async Thread-offloaded)
└── __init__.py
```

## 🚀 快速开始

### 1. 定义一个简单工具

继承 `BaseTool` 并定义 `args_schema` 即可创建一个工具。

```python
from typing import Type
from pydantic import BaseModel, Field
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import register_tool

# 1. 定义参数结构
class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称")
    unit: str = Field(default="celsius", description="温度单位")

# 2. 注册并实现工具
@register_tool("weather")
class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "查询指定城市的天气情况"
    args_schema: Type[BaseModel] = WeatherArgs

    async def _run(self, args: WeatherArgs) -> ToolResult:
        # 模拟 API 调用
        temp = 25
        return ToolResult(content=f"{args.city} 当前气温 {temp} {args.unit}")
```

### 2. 使用 ToolBox 管理工具

`ToolBox` 是工具的运行时容器，负责生命周期管理和执行统计。

```python
from gecko.core.toolbox import ToolBox
from gecko.plugins.tools.registry import load_tool

# 加载工具
toolbox = ToolBox([
    load_tool("weather"),           # 加载自定义工具
    load_tool("calculator"),        # 加载标准工具
    load_tool("duckduckgo_search")  # 加载标准工具
])

# 获取 OpenAI Schema (传给 LLM)
schemas = toolbox.to_openai_schema()

# 执行工具
result = await toolbox.execute(
    name="weather",
    arguments={"city": "Beijing"}
)
print(result)  # Output: Beijing 当前气温 25 celsius
```

---

## 🛠️ 高级用法

### 1. 有状态工具 (Stateful Tools) 与依赖注入

当工具需要访问数据库连接、API Client 或共享内存时，可以通过 `__init__` 注入依赖。

> **注意**: 这里的工具不能通过 `load_tool` 字符串加载，需要手动实例化。

```python
class OrderDatabase:
    ...

class PlaceOrderTool(BaseTool):
    name: str = "place_order"
    description: str = "下单"
    args_schema: Type[BaseModel] = PlaceOrderArgs

    def __init__(self, db: OrderDatabase):
        # 1. 必须先调用 super().__init__()
        super().__init__()
        # 2. 使用 object.__setattr__ 注入私有属性，避开 Pydantic 校验
        object.__setattr__(self, "_db", db)

    async def _run(self, args: PlaceOrderArgs) -> ToolResult:
        # 使用 self._db
        await self._db.save(args)
        return ToolResult(content="Success")

# 使用
db = OrderDatabase()
tool = PlaceOrderTool(db=db)
toolbox = ToolBox(tools=[tool])
```

### 2. 处理同步 I/O (Thread Offloading)

如果工具内部使用的是同步库（如 `requests`, `pandas`），必须将其卸载到线程池，否则会阻塞整个 Agent 的 Event Loop。

使用 `anyio.to_thread.run_sync`:

```python
from anyio.to_thread import run_sync

class SyncApiTool(BaseTool):
    ...
    async def _run(self, args: MyArgs) -> ToolResult:
        def _blocking_call():
            import time
            time.sleep(5) # 模拟阻塞 IO
            return "Done"
            
        # 在线程池中运行，不阻塞主线程
        result = await run_sync(_blocking_call)
        return ToolResult(content=result)
```

---

## 📚 标准工具库

Gecko 内置了一些开箱即用的高质量工具：

### 🧮 Calculator (`calculator`)
基于 Python AST 解析的安全计算器。
*   **安全性**: 拒绝 `__import__`, `os`, `sys` 等危险调用；禁止属性访问 (`__class__`)。
*   **防 DoS**: 限制了幂运算 (`**`) 的指数大小，防止 CPU 耗尽攻击。
*   **支持**: `+`, `-`, `*`, `/`, `sqrt`, `sin`, `log` 等。

### 🔍 DuckDuckGo Search (`duckduckgo_search`)
基于 `duckduckgo-search` 库的联网搜索工具。
*   **异步优化**: 内部已封装 `run_sync`，确保并发调用时不阻塞。
*   **隐私保护**: 无需 API Key，且不追踪用户。

---

## 🧪 测试与调试

### 单元测试
工具模块包含完整的测试套件。运行测试：

```bash
rye run pytest tests/plugins/tools/test_tools.py
```

### 常见问题
1.  **`Tool 必须继承 BaseTool`**: 确保在 `AgentBuilder.with_tools()` 中传入的是实例化的 `BaseTool` 对象，或者注册表中的字符串名称。
2.  **Pydantic Warning**: `LiteLLM` 可能会产生序列化警告，Gecko 已在 `logging.py` 中默认屏蔽，不影响使用。
3.  **`duckduckgo` 报错**: 确保安装了最新版库，Gecko 已适配了移除 `backend='api'` 的新接口。