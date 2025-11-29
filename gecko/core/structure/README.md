# Gecko 结构化输出模块（`gecko.core.structure`）

> 将 LLM 的“杂乱文本输出 / Tool Call 输出”安全、稳定地解析为 **Pydantic 模型**。

本模块是 Gecko 多智能体框架中用于“**结构化输出解析**”的核心组件，支持多种 JSON 提取策略、Schema 校验、错误聚合，并提供可扩展的 **Strategy 插件机制**（已内置 YAML 策略）。

---

## 目录结构与职责划分

```text
gecko/core/structure/
├─ __init__.py          # 对外统一入口，暴露公共 API
├─ errors.py            # 自定义异常（StructureParseError）
├─ schema.py            # Schema 工具（OpenAI tools schema、schema diff）
├─ json_extractor.py    # 文本 → JSON → Pydantic 模型（多策略 + 插件机制）
├─ engine.py            # StructureEngine 引擎（协调 tool call + 文本解析）
├─ sync.py              # 同步封装 & 轻量工具函数
````

### 各文件职责简述

* **`errors.py`**

  * `StructureParseError`：统一的结构化解析失败异常，包含策略尝试列表和原始内容预览。

* **`schema.py`**

  * `to_openai_tool(model)`：Pydantic 模型 → OpenAI Function Calling `tools` schema。
  * `get_schema_diff(data, model_class)`：对比数据与模型 schema 的差异（缺字段、多字段、简易类型不匹配）。

* **`json_extractor.py`**

  * `extract_structured_data(text, model_class, ...)`：

    * 多策略从文本中提取 JSON 并校验为 Pydantic 模型。
  * `ExtractionStrategy` + `register_extraction_strategy`：

    * Strategy 插件机制入口，支持扩展自定义解析策略。
  * 内置 YAML 策略：

    * 自动检测 `PyYAML`，如存在则注册 `yaml_fulltext` 解析策略。

* **`engine.py`**

  * `StructureEngine.parse(...)`：

    * 核心异步解析入口，优先从 `raw_tool_calls` 解析，否则回退到纯文本 JSON。
  * `StructureEngine.to_openai_tool(...)` / `get_schema_diff(...)`：

    * 对 schema 工具的统一封装。

* **`sync.py`**

  * `parse_structured_output(...)`：

    * 同步封装（内部自动处理 `asyncio` 事件循环，禁止在已有事件循环中调用）。
  * `extract_json_from_text(text)`：

    * 轻量级 JSON 提取工具（只返回 `dict`）。

* **`__init__.py`**

  * 对外暴露统一 API：

    * `StructureEngine`
    * `StructureParseError`
    * `parse_structured_output`
    * `extract_json_from_text`
    * `ExtractionStrategy`
    * `register_extraction_strategy`

---

## 对外 API 一览

从外部使用时，你只需要引用 `gecko.core.structure`：

```python
from gecko.core.structure import (
    StructureEngine,
    StructureParseError,
    parse_structured_output,
    extract_json_from_text,
    ExtractionStrategy,
    register_extraction_strategy,
)
```

---

## 快速上手

### 1. 基本用法：解析纯文本 JSON

```python
from pydantic import BaseModel
from gecko.core.structure import StructureEngine

class User(BaseModel):
    name: str
    age: int

content = '{"name": "Alice", "age": 25}'

user = await StructureEngine.parse(
    content=content,
    model_class=User,
)
print(user.name, user.age)
```

### 2. Tool Call 输出解析（例如 OpenAI / Zhipu）

```python
from pydantic import BaseModel
from gecko.core.structure import StructureEngine

class User(BaseModel):
    name: str
    age: int

# 典型 tool call 返回结构
tool_calls = [{
    "function": {
        "name": "get_user",
        "arguments": '{"name": "Bob", "age": 30}'
    }
}]

user = await StructureEngine.parse(
    content="ignored when tool_calls provided",
    model_class=User,
    raw_tool_calls=tool_calls,
)
```

> 说明：如果 `raw_tool_calls` 不为空，Engine 会优先尝试从其中解析结构化数据，只有全部失败时才回退到 `content` 文本解析。

### 3. 同步版本（脚本环境）

```python
from pydantic import BaseModel
from gecko.core.structure import parse_structured_output

class User(BaseModel):
    name: str
    age: int

content = '{"name": "Alice", "age": 25}'

user = parse_structured_output(content, User)
```

> ⚠️ 注意：
>
> * `parse_structured_output` **不能**在已有事件循环中调用（例如 FastAPI、Jupyter notebook），否则会抛 `RuntimeError`。
> * 在异步环境中请直接使用 `await StructureEngine.parse(...)`。

### 4. 轻量 JSON 提取

只需要个大概的 JSON dict，而不需要 Pydantic 校验时：

````python
from gecko.core.structure import extract_json_from_text

text = """
LLM 输出前面一些说明文字...

```json
{"name": "Alice", "age": 25}
````

后面还有废话...
"""

data = extract_json_from_text(text)

# data -> {"name": "Alice", "age": 25}

````

---

## 多策略解析逻辑

`extract_structured_data` 的内置策略顺序：

1. **`direct_json`**  
   直接对整个 `text` 做 `json.loads(text)` + Pydantic 校验。

2. **`markdown_X`**  
   从 ``` ... ``` 代码块中提取内容，尝试 `json.loads` + 校验。

3. **`braced_X`**  
   使用栈匹配提取所有 `{...}` 片段（按长度从大到小），逐个尝试 `json.loads` + 校验。

4. **`bracket_X`**  
   使用栈匹配提取所有 `[...]` 数组片段，逐个尝试 `json.loads` + 校验。

5. **`cleaned_json`**  
   对全文执行：
   - 去掉 `//` 单行注释
   - 去掉 `/* ... */` 块注释
   - 去掉末尾多余逗号
   - 去掉控制字符  
   然后再次整体 `json.loads` + 校验。

6. **插件策略（`plugin_<name>`）**  
   所有通过 `register_extraction_strategy` 注册的策略，按照注册顺序逐个尝试。

> 所有失败的策略及其错误信息会被汇总到 `StructureParseError.attempts` 中，便于排查问题。

---

## Strategy 插件扩展机制

### 1. 基本概念

```python
from dataclasses import dataclass
from typing import Callable, TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

@dataclass
class ExtractionStrategy:
    name: str
    func: Callable[[str, Type[T]], T]
````

* `name`：策略名称，用于日志和错误报告。
* `func(text, model_class)`：

  * 成功：返回 `model_class` 实例；
  * 失败：抛异常（类型不限，外层会记入 attempts）。

### 2. 注册自定义策略

```python
from gecko.core.structure import ExtractionStrategy, register_extraction_strategy

def my_strategy(text, model_class):
    # 这里写你自己的解析逻辑
    data = {"name": "FromStrategy", "age": 18}
    return model_class(**data)

register_extraction_strategy(
    ExtractionStrategy(
        name="my_strategy",
        func=my_strategy,
    )
)
```

> 插件策略会在**所有内置策略失败之后**被依次尝试，不会影响原来的行为。

---

## YAML 解析策略（内置插件）

### 自动启用条件

模块内部会尝试导入 `PyYAML`：

```python
try:
    import yaml

    def _yaml_fulltext_strategy(text, model_class):
        data = yaml.safe_load(text)
        return _validate_model(data, model_class)

    register_extraction_strategy(
        ExtractionStrategy(name="yaml_fulltext", func=_yaml_fulltext_strategy)
    )

except ImportError:
    # 没装 PyYAML 就静默忽略，不影响 JSON 能力
```

* 如果 `PyYAML` 已安装（例如 `pip install pyyaml`），则会自动注册 `yaml_fulltext` 策略。
* 如果未安装，则不会启用 YAML 策略，一切行为与纯 JSON 解析保持一致。

### 使用场景示例

```python
from pydantic import BaseModel
from gecko.core.structure import StructureEngine

class Config(BaseModel):
    host: str
    port: int

content = """
host: "127.0.0.1"
port: 8080
"""

config = await StructureEngine.parse(content, Config)
# 当 JSON 策略全部失败时，YAML 策略会接管并尝试解析全文
```

---

## Schema 工具

### 1. Pydantic → OpenAI tools schema

```python
from pydantic import BaseModel, Field
from gecko.core.structure import StructureEngine

class SearchQuery(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")

tool = StructureEngine.to_openai_tool(SearchQuery)
# tool 可直接传给 OpenAI / Zhipu 等的 tools / functions 参数
```

### 2. 数据与 Schema 差异分析

```python
from pydantic import BaseModel
from gecko.core.structure import StructureEngine

class User(BaseModel):
    name: str
    age: int

data = {"name": "Alice", "age": "twenty"}

diff = StructureEngine.get_schema_diff(data, User)
# diff = {
#   "missing_required": [],
#   "extra_fields": [],
#   "type_mismatches": [
#       {"field": "age", "expected": "integer", "actual": "str"}
#   ],
# }
```

---

## 错误处理：`StructureParseError`

当所有策略都失败时，会抛出 `StructureParseError`：

```python
from gecko.core.structure import StructureEngine, StructureParseError

try:
    user = await StructureEngine.parse(content, User)
except StructureParseError as e:
    print(e.get_detailed_error())
    # 输出包括：
    # - 主错误描述
    # - 每个策略的名称与错误截断
    # - 原始内容的前 200 字符预览
```

常用字段：

* `e.attempts`: `List[{"strategy": str, "error": str}]`
* `e.raw_content`: 原始文本（可能已截断）

---

## 与旧版 `structure.py` 的兼容

> **目标：保持对外行为尽可能一致，同时内部结构更清晰、可扩展。**

* 旧版的：

  * `StructureEngine`
  * `StructureParseError`
  * `parse_structured_output`
  * `extract_json_from_text`
* 在新结构中仍然通过 `gecko.core.structure` 暴露，并保持相同语义。

区别 / 增强点：

* 原单文件拆分为多个模块，便于维护与扩展。
* 新增：

  * **Strategy 插件接口**：`ExtractionStrategy + register_extraction_strategy`
  * **YAML 全文解析策略**（可选，依赖 PyYAML）
* 同步封装的事件循环处理更加安全：

  * 在已有事件循环环境中调用会抛 `RuntimeError`，避免隐式嵌套 loop。

---

## 典型扩展路线建议

1. **只用默认能力**
   直接使用 `StructureEngine.parse` + Pydantic 模型，即可满足绝大多数 LLM 结构化输出场景。

2. **需要兼容“非严格 JSON”输出**
   安装 `PyYAML`，内置 `yaml_fulltext` 策略即可自动工作。

3. **项目有特殊输出格式**（例如自定义标记、表格转 JSON 等）
   使用 `ExtractionStrategy + register_extraction_strategy` 注册自己的解析策略。

4. **后续更复杂需求**
   可以在 `json_extractor` 中进一步丰富策略管线，或通过 `strategies` 参数（未来扩展）实现项目级策略组合。
