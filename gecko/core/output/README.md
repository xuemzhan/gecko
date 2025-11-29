# gecko.core.output 模块说明

`gecko.core.output` 是 Gecko 框架中用于**统一描述 Agent 输出结果**的核心模块。  
在 v0.2.x 之后，它从单文件重构为一个**模块化包**，同时保持对外 API 完全兼容。

---

## 1. 功能概览

该模块解决几个核心问题：

1. **统一的输出结构**  
   - 使用 `AgentOutput` 作为标准输出模型：
     - 文本内容 `content`
     - 工具调用 `tool_calls`
     - Token 使用统计 `usage`
     - 原始响应 `raw`
     - 附加元数据 `metadata`

2. **统一的 Token 统计模型**  
   - 使用 `TokenUsage` 对齐各家 LLM Provider 的 usage 字段。

3. **扩展输出类型**  
   - `JsonOutput`：用于承载结构化 JSON 结果。
   - `StreamingOutput`：用于管理流式输出（Streaming Chunk）并最终汇总为 `AgentOutput`。

4. **便捷工厂 & 合并工具**  
   - 快速创建输出：`create_text_output`、`create_tool_output`、`create_json_output`
   - 合并多个输出：`merge_outputs`

---

## 2. 模块结构

```text
gecko/core/output/
├─ __init__.py          # 对外统一出口（保持兼容）
├─ token_usage.py       # TokenUsage 模型
├─ agent_output.py      # AgentOutput 模型
├─ factories.py         # create_* 系列工厂方法
├─ merge.py             # merge_outputs 合并逻辑
├─ json_output.py       # JsonOutput 结构化输出
└─ streaming_output.py  # StreamingOutput / StreamingChunk 流式输出
````

对外可用的对象：

```python
from gecko.core.output import (
    TokenUsage,
    AgentOutput,
    create_text_output,
    create_tool_output,
    create_json_output,
    merge_outputs,
    JsonOutput,
    StreamingOutput,
    StreamingChunk,
)
```

> ✅ 对原有代码完全兼容：
> 老代码中 `from gecko.core.output import AgentOutput, TokenUsage, ...` **无需修改**。

---

## 3. TokenUsage：Token 使用统计

### 3.1 基本用法

```python
from gecko.core.output import TokenUsage

usage = TokenUsage(
    prompt_tokens=100,
    completion_tokens=50,
    # total_tokens 可省略，构造时会自动补全
)

print(usage.total_tokens)   # 150
print(str(usage))           # TokenUsage(prompt=100, completion=50, total=150)
```

### 3.2 自动校验与补全逻辑

* 若 `total_tokens == 0` 且 `prompt + completion > 0`：自动补全 `total_tokens`。
* 若 `total_tokens != 0` 且与计算值不一致：

  * **不覆盖 provider 的值**，只记录一条 warning，方便排查。
* 三者都为 0 时：保持为 0。

### 3.3 成本估算

```python
cost = usage.get_cost_estimate(
    prompt_price_per_1k=0.03,      # 输入 1K tokens 价格
    completion_price_per_1k=0.06,  # 输出 1K tokens 价格
)
print(f"Estimated cost: ${cost:.4f}")
```

---

## 4. AgentOutput：标准 Agent 输出模型

### 4.1 基本结构

```python
from gecko.core.output import AgentOutput, TokenUsage

output = AgentOutput(
    content="Hello, how can I help?",
    tool_calls=[],
    usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    raw=None,
    metadata={"model": "glm-4-flash"},
)
```

关键字段说明：

* `content: str`
  最终文本回复，自动转换为字符串（传入非 str 时会 `str(value)`）。
* `tool_calls: List[dict]`
  工具调用列表，兼容 OpenAI `tool_calls` 格式。
* `usage: TokenUsage | None`
  Token 使用统计。
* `raw: Any`
  原始模型响应，用于调试（类型不限制）。
* `metadata: dict`
  任意附加信息，例如模型名、请求耗时、是否包含 raw 等。

### 4.2 安全 & 容错行为

* `content`：

  * `None` → `""`
  * 非 str → `str(value)`
* `tool_calls`：

  * `None` → `[]`
  * `list` → 原样返回
  * `tuple` / `set` → 转为 `list`，并记录 warning
  * `dict`（单个调用）→ 包装成 `[dict]`，并记录 warning
  * 其他类型 → `[]`，记录 warning

### 4.3 常用方法

```python
if output.has_content():
    print("回复内容：", output.content)

if output.has_tool_calls():
    print("调用工具数量：", output.tool_call_count())
    print("工具列表：", output.get_tool_names())

if output.has_usage():
    print("总 Tokens:", output.usage.total_tokens)

print("是否空输出：", output.is_empty())
print("简要摘要：", output.summary())
print("详细格式化：")
print(output.format(include_metadata=True))

# 转为 OpenAI 消息格式（用于下一轮对话）
msg = output.to_message_dict()
# => {"role": "assistant", "content": "...", "tool_calls": [...]}
```

### 4.4 统计信息

```python
stats = output.get_stats()
# {
#   "content_length": 123,
#   "has_content": True,
#   "tool_call_count": 1,
#   "tool_names": ["search"],
#   "is_empty": False,
#   "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
# }
```

---

## 5. 工厂函数：create_text_output / create_tool_output / create_json_output

### 5.1 纯文本输出

```python
from gecko.core.output import create_text_output, TokenUsage

output = create_text_output(
    "Hello, world!",
    usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    model="glm-4-flash",            # 会写入 metadata
    request_id="req-123",
)

print(output.metadata["model"])
```

### 5.2 工具调用输出

```python
from gecko.core.output import create_tool_output

tool_calls = [
    {
        "id": "call_1",
        "function": {
            "name": "search",
            "arguments": '{"query": "Gecko 框架"}',
        },
    }
]

output = create_tool_output(
    tool_calls=tool_calls,
    content="我将为你搜索 Gecko 框架相关信息。",
    model="glm-4-flash",
)

print(output.has_tool_calls())   # True
```

### 5.3 结构化 JSON 输出

见下方 `JsonOutput` 小节。

---

## 6. merge_outputs：合并多个 AgentOutput

在多 Agent / 多阶段流水线场景中，可以将多个 `AgentOutput` 合并为一个：

```python
from gecko.core.output import AgentOutput, merge_outputs

step1 = AgentOutput(content="Part 1")
step2 = AgentOutput(content="Part 2")
step3 = AgentOutput(tool_calls=[{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}])

merged = merge_outputs([step1, step2, step3])

print(merged.content)          # "Part 1\nPart 2"
print(merged.tool_call_count())  # 1
```

合并策略：

* 内容 `content`：将所有有内容的输出按顺序用换行符拼接。
* 工具调用 `tool_calls`：简单拼接列表。
* 使用统计 `usage`：对 `prompt_tokens` 和 `completion_tokens` 做累加，`total_tokens = 二者之和`。
* 元数据 `metadata`：后面的覆盖前面的（`dict.update` 语义）。

---

## 7. JsonOutput：结构化 JSON 输出

在某些场景下，Agent 返回的不是一段文本，而是一段**结构化的数据**（dict / list），此时使用 `JsonOutput` 更合适。

### 7.1 创建 JsonOutput

```python
from gecko.core.output import create_json_output

data = {
    "status": "ok",
    "items": [
        {"id": 1, "name": "foo"},
        {"id": 2, "name": "bar"},
    ],
}

json_output = create_json_output(
    data=data,
    model="glm-4-flash",
    schema_version="v1",
)

print(json_output.data["status"])         # ok
print(json_output.metadata["schema_version"])  # v1
```

### 7.2 转为字典 / 摘要

```python
print(json_output.to_dict())
print(json_output.summary())   # 如: JSON: {'status': 'ok', 'items': [...]} | Tokens: 123
```

### 7.3 转换为 AgentOutput（文本形式）

```python
agent_output = json_output.to_agent_output(pretty=True)
print(agent_output.content)    # 会是格式化的 JSON 字符串
```

---

## 8. StreamingOutput：流式输出

用于管理**流式生成的多个片段**，例如：

* provider 的 `stream=True` 模式；
* 自定义 Agent pipeline 按 chunk 产出内容。

### 8.1 基本用法

```python
from gecko.core.output import StreamingOutput, StreamingChunk

stream = StreamingOutput(metadata={"model": "glm-4-flash"})

stream.append_chunk(StreamingChunk(
    index=0,
    content_delta="Hello",
))
stream.append_chunk(StreamingChunk(
    index=1,
    content_delta=", world!",
))

# 边消费边打印
for delta in stream.iter_contents():
    print(delta, end="")
# 输出：Hello, world!

# 流式结束后汇总为 AgentOutput
final_output = stream.finalize()
print(final_output.content)  # "Hello, world!"
```

### 8.2 含使用统计的流式输出

```python
from gecko.core.output import TokenUsage

stream.append_chunk(StreamingChunk(
    index=0,
    content_delta="Hel",
    usage_delta=TokenUsage(prompt_tokens=5, completion_tokens=3),
))
stream.append_chunk(StreamingChunk(
    index=1,
    content_delta="lo",
    usage_delta=TokenUsage(prompt_tokens=0, completion_tokens=2),
))

final_output = stream.finalize()
print(final_output.usage.total_tokens)   # 自动汇总为 5 + 3 + 0 + 2
```

> 注意：
>
> * 若 `StreamingOutput.usage` 非空，则 `finalize()` 直接使用整体 usage。
> * 否则，会尝试对所有 `chunk.usage_delta` 累加。

### 8.3 统计信息

```python
stats = stream.get_stats()
# {
#   "chunk_count": 2,
#   "total_content_length": 5,
#   "has_usage": True/False,
# }
```

---

## 9. 扩展建议

如果你需要扩展更多输出类型，例如：

* `HtmlOutput`：专门承载 HTML 内容；
* `MarkdownOutput`：带有目录/标题信息的 markdown 文档；
* `RichMediaOutput`：文本 + 图片/表格的混合结构；

推荐方式：

1. 在 `gecko/core/output/` 下新增相应的 `xxx_output.py` 文件；
2. 定义新模型（可参考 `JsonOutput` / `StreamingOutput`）；
3. 在 `__init__.py` 中导出该模型；
4. 如有必要，在 `factories.py` 中增加对应的 `create_xxx_output` 工厂方法。

这样可以保持：

* 对外 API 一致；
* 内部结构清晰易扩展；
* 与现有 `AgentOutput` / `TokenUsage` 生态兼容。

---

## 10. 快速对比：何时用哪种 Output？

| 类型                | 场景                    | 典型入口                   |
| ----------------- | --------------------- | ---------------------- |
| `AgentOutput`     | 普通对话、带工具调用的回复         | LLM 调用结果 / Agent 中间输出  |
| `JsonOutput`      | 结构化数据结果（表格、配置、解析结果等）  | 工具调用、结构化推理结果           |
| `StreamingOutput` | 流式生成结果，需要边输出边展示，并最终汇总 | OpenAI/GLM 等 stream 模式 |

---

如需在具体调用链路中落地（例如接入 ZhipuChat / OpenAI），可以在上层封装一个统一的适配器，将模型原始 response 统一转换为上述 Output 类型，方便日志、监控和后续处理。

