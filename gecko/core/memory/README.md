# `gecko.core.memory` 模块说明

`gecko.core.memory` 是 Gecko 多智能体框架中的**上下文记忆管理模块**，负责在有限的模型上下文窗口内，尽可能高效、稳定地管理对话历史与 Token 预算。

本模块主要提供两类能力：

* **Token 级精确记忆管理**：`TokenMemory`
* **带摘要压缩的记忆管理**：`SummaryTokenMemory`

---

## 一、总体设计

### 功能目标

1. **精确计数**

   * 基于 `tiktoken` 或 `ModelProtocol.count_tokens` 进行模型特定 Token 计数；
   * 在 `tiktoken` 不可用时自动退化为“字符估算”。

2. **智能裁剪**

   * 支持在 `max_tokens` 限制内，以“反向回填”的方式保留最近的有效对话；
   * 支持对系统消息进行防御性截断与强制 Token 级截断。

3. **性能优化**

   * 内置 **LRU 缓存**，缓存消息 → Token 数的映射；
   * 通过 **全局线程池** 将 Token 计数的 CPU 密集型计算从事件循环中卸载出去；
   * 支持同步 / 异步两套批量计数 API。

4. **多模态支持**

   * 对文本块、图片块 Token 占用进行估算；
   * 兼容 **对象形式** 和 **dict 形式** 的多模态 content。

5. **摘要记忆（高级特性）**

   * 当历史超出限制时，`SummaryTokenMemory` 自动调用 LLM 将旧对话合并为摘要；
   * 通过摘要系统消息继续携带“语义历史”，而不是简单丢弃旧消息。

---

## 二、目录结构

原始代码为单文件 `memory.py`。现拆分为一个包：

```text
gecko/core/memory/
├─ __init__.py        # 对外统一入口（保持向后兼容）
├─ _executor.py       # 内部：全局线程池管理
├─ base.py            # TokenMemory 基础实现
└─ summary.py         # SummaryTokenMemory 摘要记忆扩展
```

### 向后兼容说明

拆分后，对外 API **保持不变**：

```python
from gecko.core.memory import TokenMemory, SummaryTokenMemory, shutdown_token_executor
```

* 原来 `from gecko.core.memory import ...` 的代码 **不需要做任何修改**。
* `__init__.py` 会将内部模块中的核心类/函数重新导出。

---

## 三、核心组件

### 1. TokenMemory（基础记忆管理器）

定义位置：`gecko/core/memory/base.py`

#### 主要职责

* 为每条 `Message` 计算 Token 数；
* 维护 Token 计数的 LRU 缓存；
* 提供同步/异步的批量计数接口；
* 基于 `max_tokens` 对历史消息进行“反向回填式”裁剪；
* 对多模态 content（文本+图片）进行 Token 估算。

#### 构造参数

```python
TokenMemory(
    session_id: str,
    storage: Optional[SessionInterface] = None,
    max_tokens: int = 4000,
    model_name: str = "gpt-3.5-turbo",
    cache_size: int = 2000,
    max_message_length: int = 20000,
    enable_cache_for_batch: bool = True,
    model_driver: Optional[ModelProtocol] = None,
    enable_async_counting: bool = True,
)
```

关键参数说明：

* `max_tokens`：上下文最大 Token 限制（系统 + 历史 + 当前输入）；
* `model_name`：用于 `tiktoken.encoding_for_model` 选择编码器；
* `model_driver`：可选的模型驱动，若实现了 `count_tokens`，则单条计数时会优先调用；
* `enable_async_counting`：是否通过线程池加速异步批量 Token 计算。

#### 常用方法

* `count_message_tokens(message: Message) -> int`
  计算单条消息 Token 数（带缓存）。

* `count_messages_batch(messages: List[Message], use_cache: Optional[bool] = None) -> List[int]`
  批量同步计数；`use_cache=None` 时跟随实例级配置 `enable_cache_for_batch`。

* `count_messages_batch_async(messages: List[Message], use_cache: Optional[bool] = None) -> List[int]`
  异步批量计数；内部使用线程池卸载 CPU 计算。

* `async get_history(raw_messages: List[Dict[str, Any]], preserve_system: bool = True) -> List[Message]`
  从原始消息字典列表中解析 `Message`，并在 `max_tokens` 限制内保留尽可能多的最近对话历史。

* `count_total_tokens(messages: List[Message]) -> int`
  计算消息列表总 Token 数。

* `estimate_tokens(text: str) -> int`
  快速估算一段文本的 Token 数。

* `clear_cache()` / `get_cache_stats()` / `print_cache_stats()`
  缓存管理与监控工具。

---

### 2. SummaryTokenMemory（摘要记忆管理器）

定义位置：`gecko/core/memory/summary.py`

#### 主要职责

在 `TokenMemory` 的基础上增加“摘要记忆”能力：

* 根据 `max_tokens` 和 `summary_reserve_tokens` 对历史进行分段：

  * 最近消息部分：直接保留（recent）；
  * 较旧消息部分：通过模型生成摘要（to_summarize）。

* 将旧摘要 + 新旧对话合并为新的 `current_summary`；

* 在 `get_history` 返回结果中，将摘要以 `system` 消息注入上下文。

#### 构造参数

```python
SummaryTokenMemory(
    session_id: str,
    model: ModelProtocol,
    summary_prompt: Optional[str] = None,
    summary_reserve_tokens: int = 500,
    **kwargs,  # 透传给 TokenMemory
)
```

说明：

* `model`：需要实现 `acompletion` 和 `count_tokens` 的模型驱动；
* `summary_reserve_tokens`：为“摘要 + recent”预留的 Token 预算（软预算）；
* 其余参数与 `TokenMemory` 一致，通过 `kwargs` 透传。

#### 定制 Prompt

可在构造时传入自定义 `summary_prompt`，内部会使用 `PromptTemplate`：

```python
from gecko.core.memory import SummaryTokenMemory

memory = SummaryTokenMemory(
    session_id="user_123",
    model=model_driver,
    summary_prompt=(
        "请将以下对话内容浓缩成简洁摘要，保留关键信息，尽量使用中文：\n\n"
        "{{ history }}\n\n摘要："
    ),
    max_tokens=4000,
)
```

---

### 3. 线程池工具：shutdown_token_executor

定义位置：`gecko/core/memory/_executor.py`
对外导出：`gecko.core.memory.shutdown_token_executor`

用于在测试或进程退出前显式关闭 Token 计算线程池：

```python
from gecko.core.memory import shutdown_token_executor

# 单元测试 teardown 时调用
shutdown_token_executor()
```

> 正常服务场景下通常不需要手动调用，依赖进程退出时自动回收即可。

---

## 四、使用示例

### 1. 基础用法：TokenMemory

```python
from gecko.core.memory import TokenMemory
from gecko.core.message import Message

memory = TokenMemory(
    session_id="user_123",
    max_tokens=4000,
    model_name="gpt-4",
)

# 构造消息
msgs = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="你好，请帮我写一个 BMS 诊断策略示例。"),
    Message(role="assistant", content="好的，下面是一个简单示例……"),
]

# 同步批量计数
token_counts = memory.count_messages_batch(msgs)
total_tokens = memory.count_total_tokens(msgs)

# 按 max_tokens 自动裁剪历史
raw_msgs = [m.model_dump() for m in msgs]
history = asyncio.run(memory.get_history(raw_msgs))
```

### 2. 带摘要的记忆：SummaryTokenMemory

```python
from gecko.core.memory import SummaryTokenMemory
from gecko.core.message import Message

# 假设 model_driver 实现了 ModelProtocol，支持 acompletion/count_tokens
memory = SummaryTokenMemory(
    session_id="user_123",
    model=model_driver,
    max_tokens=4000,
    summary_reserve_tokens=500,
)

raw_msgs = [...]  # 来自存储的历史记录（list[dict]）

# 获取已裁剪 + 摘要后的历史
history = asyncio.run(memory.get_history(raw_msgs))

# 清空摘要（例如用户点击“重新开始”时）
memory.clear_summary()
```

---

## 五、注意事项 & 设计细节

1. **多模态 content 支持**

   * 支持 `Message.content` 为：

     * `str`
     * `list[对象]`，对象拥有 `type` / `text` / `image_url` 属性
     * `list[dict]`，形如 `{"type": "text", "text": "..."}`
   * 对图片块采用粗略估算，`detail == "low"` 时 Token 较少。

2. **模型驱动计数与线程池**

   * `_count_tokens_impl` 在 **未显式传入 encode** 时，会优先调用 `model_driver.count_tokens`；
   * 在线程池路径中会显式传入 `encode`，从而避免在子线程中调用远程模型接口；
   * 若需使用远程 Token 计数（例如云端 API），建议只在单条或少量同步计数场景中使用。

3. **Token 限制与摘要预算**

   * `SummaryTokenMemory` 内部使用：

     * `reserved = summary_reserve_tokens + sys_tokens`
     * `available = max_tokens - reserved`
   * 先用 `available` 为 recent 消息分配 Token；
   * 再用剩余预算为摘要分配 Token，必要时会对摘要内容进行二分截断。

4. **向后兼容**

   * 拆包前后，对外的 import 与 API 均保持一致；
   * 若你在项目中使用的是 `from gecko.core.memory import TokenMemory` 等写法，**无需修改**。

---

## 六、后续扩展方向（建议）

* 支持针对 **文件/表格/音频** 等其他模态的 Token 估算接口；
* 将 `storage: SessionInterface` 与记忆模块打通，实现“自动从存储加载 + 写回”的高级记忆管理；
* 针对不同模型（如 OpenAI / 自研 LLM）封装多套 Token 计数策略，并通过配置切换。

---

如需在此基础上继续扩展（比如增加「层级记忆」「长期记忆」等），推荐以 `TokenMemory` 为底座，在 `summary.py` 类似的方式上派生新的记忆策略类即可。
