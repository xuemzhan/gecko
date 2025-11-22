# Gecko Core Protocols

`gecko.core.protocols` 是 Gecko 框架的接口定义层。它定义了框架中核心组件（模型、存储、工具等）必须遵循的标准契约。

Gecko 采用 Python 的 `typing.Protocol` 机制实现 **Duck Typing（鸭子类型）**。这意味着只要你的类实现了协议规定的方法和属性，它就可以被框架使用，而无需显式继承特定的基类。

## 📂 模块结构

该包将核心协议按功能领域进行了拆分：

| 模块文件 | 主要协议/类 | 描述 |
| :--- | :--- | :--- |
| **`model.py`** | `ModelProtocol`<br>`StreamableModelProtocol`<br>`CompletionResponse` | 定义 LLM 模型的调用接口（同步/流式）及标准的响应数据结构。 |
| **`storage.py`** | `StorageProtocol` | 定义键值（KV）存储后端的增删改查接口，用于 Session 持久化。 |
| **`tool.py`** | `ToolProtocol` | 定义 Agent 工具的标准接口（名称、描述、参数定义、执行逻辑）。 |
| **`vector.py`** | `VectorStoreProtocol` | 定义向量数据库的接口（RAG 场景），支持向量的存储与检索。 |
| **`embedder.py`** | `EmbedderProtocol` | 定义文本嵌入模型的接口，用于生成 Vector Embedding。 |
| **`runnable.py`** | `RunnableProtocol` | 定义通用的可运行对象接口（如 Agent, Chain, Workflow Node）。 |
| **`base.py`** | `check_protocol`<br>`get_missing_methods` | 提供运行时协议检查和反射工具。 |

## 🛠️ 核心协议说明

### 1. ModelProtocol (模型)
所有 LLM 适配器必须实现此协议。

```python
class ModelProtocol(Protocol):
    async def acompletion(self, messages: List[Dict], **kwargs) -> CompletionResponse:
        ...
```

如果是流式模型，还需实现 `astream`：
```python
class StreamableModelProtocol(ModelProtocol, Protocol):
    async def astream(self, messages: List[Dict], **kwargs) -> AsyncIterator[StreamChunk]:
        ...
```

### 2. StorageProtocol (存储)
用于 Session Memory 的持久化后端。

```python
class StorageProtocol(Protocol):
    async def get(self, key: str) -> Optional[Dict]: ...
    async def set(self, key: str, value: Dict, ttl: int = None) -> None: ...
    async def delete(self, key: str) -> bool: ...
```

### 3. ToolProtocol (工具)
Agent 使用的工具必须具备以下属性和方法。

```python
class ToolProtocol(Protocol):
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    
    async def execute(self, arguments: Dict[str, Any]) -> str: ...
```

## 🚀 如何使用

### 实现一个协议
你不需要显式继承 `Protocol` 类，只需实现对应的方法即可。

```python
# 示例：实现一个简单的工具
class MyCalculator:
    name = "calculator"
    description = "Performs basic math"
    parameters = {
        "type": "object",
        "properties": {"expr": {"type": "string"}}
    }

    async def execute(self, arguments: dict) -> str:
        return str(eval(arguments["expr"]))

# MyCalculator 隐式地符合 ToolProtocol
```

### 运行时验证
Gecko 提供了一组 `validate_*` 函数，用于在运行时检查对象是否符合协议，如果不符合会抛出详细的 `TypeError` 或 `ValueError`。

```python
from gecko.core.protocols import validate_tool, check_protocol, ToolProtocol

my_tool = MyCalculator()

# 1. 简单检查 (返回 bool)
is_valid = check_protocol(my_tool, ToolProtocol)  # True

# 2. 严格验证 (抛出异常并提示缺失的方法/属性)
try:
    validate_tool(my_tool)
    print("工具验证通过 ✅")
except Exception as e:
    print(f"工具无效 ❌: {e}")
```

## 📦 数据模型
为了保证组件间的数据交换标准统一，本模块还定义了基于 Pydantic 的核心数据模型：

*   **`CompletionResponse`**: 标准的模型响应封装。
*   **`StreamChunk`**: 流式响应片段封装。
*   **`CompletionUsage`**: Token 消耗统计。