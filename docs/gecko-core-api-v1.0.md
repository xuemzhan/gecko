# Gecko 核心 API v1.0 稳定接口规范（草案）

> 文件路径建议：`docs/core-api-v1.0.md`  
> 本文档用于声明 Gecko v1.0 版本对外 **承诺稳定** 的核心接口范围，以及配套的版本管理与演进策略。

---

## 0. 前言

- **目标读者**：  
  - 使用 Gecko 构建智能体 / 工作流应用的业务开发者  
  - 基于 Gecko 扩展模型、工具、存储后端的插件开发者  
- **文档目的**：  
  - 明确 v1.0 中的「稳定 API」边界  
  - 区分应用层 API（L1）、扩展层 API（L2）与内部实现（L3）  
  - 为后续版本演进提供兼容性约束依据

> ⚠️ 注意：本文件为 v1.0 规范草案，正式发布前请与 CHANGELOG、源码注释保持一致。

---

## 1. API 分级与范围

### 1.1 API 分级说明

- **L1：应用开发者 API（强稳定）**
  - 面向：直接使用 Gecko 搭建智能体 / 工作流 / 多智能体应用的开发者
  - 要求：v1.x 生命周期内不做破坏性变更，必要变更需经过 deprecate 过渡期

- **L2：扩展开发者 API（相对稳定）**
  - 面向：开发模型适配器、Memory 策略、存储后端、工具插件等的扩展开发者
  - 要求：尽量保持兼容，重要变更需在文档和 CHANGELOG 中提前说明

- **L3：内部实现 / 实验性 API（不承诺稳定）**
  - 面向：框架内部演进与实验性特性
  - 要求：可在小版本中发生破坏性调整，不建议外部直接依赖

### 1.2 v1.0 稳定 API 范围概览

- L1（应用开发者）：
  - 顶层导出：`gecko.__version__`、`Agent`、`AgentBuilder`、`Message`、`Role`、`AgentOutput`、`TokenUsage`、`TokenMemory`、`SummaryTokenMemory`、`StructureEngine`、`Workflow`、`step`、`Next`、`Team`
- L2（扩展开发者）：
  - 模型扩展基类 / 协议
  - Memory 扩展基类
  - Storage 后端扩展基类
  - Tool 定义装饰器 / 结构
- L3（内部 / 实验性）：
  - Telemetry、Tracing、Events、Guardrails、Knowledge 等尚在演进中的模块

---

## 2. L1 应用开发者 API

### 2.1 顶层导出（`gecko` 包）

#### 2.1.1 顶层符号一览

自 v1.0 起，`gecko/__init__.py` 至少导出以下符号：

- `__version__: str`
- `Agent`
- `AgentBuilder`
- `Message`
- `Role`
- `AgentOutput`
- `TokenUsage`
- `TokenMemory`
- `SummaryTokenMemory`
- `StructureEngine`
- `Workflow`
- `step`
- `Next`
- `Team`

#### 2.1.2 使用示例（示意）

```python
from gecko import AgentBuilder, Message, Workflow, step, Next

async def main():
    agent = (
        AgentBuilder()
        .with_model(...)
        .with_system_prompt("You are a helpful assistant.")
        .build()
    )

    output = await agent.run("Hello Gecko!")
    print(output.content)
````

> TODO：补充完整的最小可运行示例。

---

### 2.2 Agent & Builder（`gecko.core.agent` / `gecko.core.builder`）

#### 2.2.1 Agent（`gecko.core.agent.Agent`）

**职责**：封装单智能体的推理流程，统一对外提供 `run()` / 流式接口。

* 核心方法（签名以实现为准，语义需稳定）：

  * `async def run(self, input, **kwargs) -> AgentOutput`
  * 可选：`async def astream(self, input, **kwargs) -> AsyncIterator[AgentOutput]`
* 关键属性（对外可见但建议只读）：

  * `model`
  * `memory`
  * `tools`

> TODO：在这里列出正式的方法签名与参数说明表格。

#### 2.2.2 AgentBuilder（`gecko.core.builder.AgentBuilder`）

**职责**：按 Builder 模式配置并构建 `Agent` 实例。

* 稳定链式方法：

  * `with_model(model)`
  * `with_prompt(prompt)`
  * `with_system_prompt(text: str)`
  * `with_memory(memory)`
  * `with_tools(tools | *tools)`
  * `with_events(event_bus | callbacks)`（如有）
  * `build() -> Agent`

```python
from gecko import AgentBuilder

agent = (
    AgentBuilder()
    .with_model(my_model)
    .with_system_prompt("You are a code assistant.")
    .with_memory(TokenMemory(max_tokens=2048))
    .with_tools([search_tool])
    .build()
)
```

---

### 2.3 消息与 Prompt（`gecko.core.message` / `gecko.core.prompt`）

#### 2.3.1 Message & Role（`gecko.core.message`）

* `class Message`

  * 典型字段：

    * `role: str`
    * `content: str | list | dict`
    * `name: str | None`
    * `tool_call_id: str | None`
  * 常用构造方法（如已实现）：

    * `Message.user(content)`
    * `Message.assistant(content)`
    * `Message.system(content)`

* `Role`（Enum 或常量集合）

  * `Role.USER`
  * `Role.ASSISTANT`
  * `Role.SYSTEM`
  * `Role.TOOL`
  * ...

```python
from gecko import Message, Role

msg = Message(role=Role.USER, content="Hello")
```

#### 2.3.2 Prompt 模板（`gecko.core.prompt`）

> ⚠️ 该小节为骨架，具体类名与方法需要与实际实现对齐。

* 主力类（示例名）：`PromptTemplate`

  * 功能：

    * 变量插值（`format` / `render`）
    * 条件判断（if/else）
    * 循环块（for）
    * 部分填充（partial）
  * 典型方法：

    * `render(**kwargs) -> str`
    * `format_safe(**kwargs) -> str`
    * `partial(**preset) -> PromptTemplate`

```python
from gecko.core.prompt import PromptTemplate

tmpl = PromptTemplate("Hello, {{ name }}!")
print(tmpl.render(name="Gecko"))
```

---

### 2.4 Memory（`gecko.core.memory`）

#### 2.4.1 TokenMemory

* 模块：`gecko.core.memory`
* 类：`TokenMemory`

  * 责任：在给定 `max_tokens` 约束下维护与裁剪会话上下文
  * 核心方法：

    * `append(message: Message) -> None`
    * `get_context() -> list[Message]`
    * `reset() -> None`

#### 2.4.2 SummaryTokenMemory

* 类：`SummaryTokenMemory`

  * 在 TokenMemory 基础上引入「自动摘要」机制，用于长会话压缩。

```python
from gecko import TokenMemory, Message

memory = TokenMemory(max_tokens=2048)
memory.append(Message.user("你好"))
context = memory.get_context()
```

> TODO：补充摘要策略相关的行为说明。

---

### 2.5 输出封装（`gecko.core.output`）

#### 2.5.1 AgentOutput

* 模块：`gecko.core.output`
* 类：`AgentOutput`

  * 典型字段：

    * `content`
    * `messages`
    * `tool_calls`
    * `usage: TokenUsage | None`
  * 方法：

    * `has_content() -> bool`
    * `is_empty() -> bool`

#### 2.5.2 TokenUsage

* 类：`TokenUsage`

  * 字段：

    * `prompt_tokens: int`
    * `completion_tokens: int`
    * `total_tokens: int`

#### 2.5.3 工具函数

* `create_text_output(content: str, *, usage: TokenUsage | None = None) -> AgentOutput`
* `create_tool_output(tool_name: str, arguments: dict, *, usage: TokenUsage | None = None) -> AgentOutput`
* `merge_outputs(*outputs: AgentOutput) -> AgentOutput`

```python
from gecko.core.output import AgentOutput, TokenUsage

usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
output = AgentOutput(content="OK", usage=usage)
```

---

### 2.6 结构化输出（`gecko.core.structure`）

* 模块：`gecko.core.structure`
* 类：`StructureEngine`

  * 典型接口（示意）：

    * `@staticmethod to_openai_tool(model_cls: type[BaseModel]) -> dict`
    * `@classmethod async parse(model_cls, raw) -> BaseModel`

```python
from pydantic import BaseModel, Field
from gecko import StructureEngine

class UserProfile(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(ge=0, le=150)

schema = StructureEngine.to_openai_tool(UserProfile)
```

> TODO：补充解析示例与错误处理说明。

---

### 2.7 Workflow & Team（`gecko.compose.workflow` / `gecko.compose.team`）

#### 2.7.1 Workflow / step / Next

* 模块：`gecko.compose.workflow`

  * `class Workflow`

    * `async def run(self, input, **kwargs) -> Any`
  * `def step(func) -> func`

    * 标记 Workflow 节点
  * `class Next`

    * 字段示意：`target: str`, `data: Any | None`, `terminate: bool = False`

```python
from gecko import Workflow, step, Next

wf = Workflow()

@step
async def start(ctx, data):
    # ...
    return Next(target="end", data=data)

@step
async def end(ctx, data):
    return data

result = await wf.run(input="hello")
```

#### 2.7.2 Team

* 模块：`gecko.compose.team`

  * `class Team`

    * `async def run(self, input, **kwargs) -> AgentOutput | dict[str, AgentOutput]`

```python
from gecko import Team, AgentBuilder

team = Team(
    experts=[
        AgentBuilder().with_model(...).build(),
        AgentBuilder().with_model(...).build(),
    ]
)

result = await team.run("分析一下这个需求")
```

> TODO：根据实际实现补充 Team 构造参数与协作策略说明。

---

## 3. L2 扩展开发者 API

> 本节列出 v1.0 中建议对外开放、相对稳定的扩展接口，适合开发模型适配器、Memory 策略、存储后端、工具插件等。

### 3.1 模型扩展接口（`gecko.plugins.models`）

* 抽象基类 / 协议（示意）：

  * `BaseChatModel` 或 `ModelProtocol`
  * 关键抽象方法：

    * `async def acompletion(self, messages: list[Message], **kwargs) -> AgentOutput | ModelRawOutput`

> TODO：贴出正式的抽象类定义与最小适配器示例（如 ZhipuChat）。

---

### 3.2 Memory 扩展接口（`gecko.core.memory.base`）

* `class BaseMemory`

  * 抽象方法：

    * `append(message: Message) -> None`
    * `get_context() -> list[Message]`
    * `reset() -> None`

> TODO：给出一个自定义 Memory（如 RedisMemory）的示例骨架。

---

### 3.3 Storage 扩展接口（`gecko.plugins.storage`）

* `class BaseStorageBackend`

  * 典型抽象方法：

    * `save_session(session_id: str, data: dict) -> None`
    * `load_session(session_id: str) -> dict | None`
    * `delete_session(session_id: str) -> None`

> TODO：补充 SQLite / 内存后端的参考实现说明。

---

### 3.4 Tool 定义接口（`gecko.plugins.tools` / `gecko.core.tools`）

* `@tool` 装饰器（示意）
* `class Tool` / `ToolSpec`

```python
from gecko.plugins.tools import tool

@tool
def search(query: str, max_results: int = 5) -> list[dict]:
    """搜索工具"""
    ...
```

> TODO：说明参数映射、返回值约定以及与 LLM Tool Schema 的关系。

---

## 4. L3 内部与实验性 API（不承诺稳定）

> 下列模块在 v1.0 中视为 **internal / experimental**，可能在小版本内发生破坏性变更，不建议外部直接依赖。

* `gecko.core.telemetry`

  * `GeckoTelemetry`, `TelemetryConfig`, 各种 exporter / 集成
* `gecko.core.tracing`

  * trace_id/span_id 上下文、Span 封装等
* `gecko.core.events`

  * `EventBus`, `BaseEvent`, `AgentRunEvent` 等
* `gecko.plugins.guardrails.*`

  * PII 过滤、内容安全策略等
* `gecko.plugins.knowledge.*`

  * RAG / 知识库相关插件框架
* 其它未在 L1 / L2 清单中出现的模块和符号

示例注释建议：

```python
# NOTE: Internal API. Behavior and signature may change without notice.
```

---

## 5. 版本管理与兼容性策略

### 5.1 版本号规范

* 采用语义化版本号：`MAJOR.MINOR.PATCH`

  * `1.0.0`：核心 API v1.0 首次发布
  * `1.x.y`：在保持 L1 兼容前提下的功能增强与修复
  * `2.0.0`：允许对 L1 做破坏性调整

### 5.2 兼容性承诺

* 对 **L1 API**：

  * v1.x 内不做破坏性改动；
  * 如必须调整，先标记 `@deprecated` 或在文档中标明弃用周期，并提供迁移方案。

* 对 **L2 API**：

  * 尽量保持兼容；
  * 发生变更时，在 CHANGELOG 中明确说明，并给出升级指引。

* 对 **L3 API**：

  * 不提供兼容性承诺；
  * 可以在任何小版本中调整或移除。

### 5.3 文档与测试要求

* 每个 L1 API 对应：

  * 文档示例（本文件 + API Reference）
  * 至少 1 个回归测试用例（import + 基础行为）
* CI 中至少包含：

  * `test_imports`：覆盖所有 `gecko.*` 模块可导入；
  * 核心工作流/多智能体 end-to-end 测试；
  * 结构化输出、Memory 裁剪、输出封装等关键路径测试。

---

## 6. 附录：示例工程与最佳实践（预留）

> TODO：本节可放置一个「最小可运行工程」示例结构，帮助新用户快速上手。

示例目录结构（草案）：

```text
examples/
  ├─ quickstart/
  │   ├─ simple_agent.py
  │   ├─ workflow_demo.py
  │   └─ team_demo.py
  └─ advanced/
      ├─ zhipu_agent.py
      ├─ structured_output_demo.py
      └─ memory_summary_demo.py
```
