# gecko.core.prompt 模块说明

> 灵活、可扩展的 Prompt 模板系统：模板管理 + 组合器 + 验证 Lint + 版本注册中心。

---

## 1. 模块整体设计

`gecko.core.prompt` 从单文件重构为一个 **包**，在保持 **向后兼容** 的前提下，新增了更强的扩展能力与结构化管理能力。

### 1.1 目录结构

```text
gecko/core/prompt/
├─ __init__.py        ← 对外兼容层（统一出口）
├─ jinja_env.py       ← Jinja2 环境封装（懒加载 + 全局单例）
├─ template.py        ← PromptTemplate 核心实现
├─ library.py         ← 预定义 Prompt 模板库（PromptLibrary）
├─ composer.py        ← Prompt 组合器（PromptComposer）
├─ validators.py      ← Prompt 验证 / Lint 工具
└─ registry.py        ← Prompt 注册中心 / 版本管理
````

### 1.2 向后兼容说明

旧代码中常见写法：

```python
from gecko.core.prompt import PromptTemplate, PromptLibrary, DEFAULT_REACT_PROMPT
```

仍然 **完全可用**。
`__init__.py` 中通过 re-export + 修正 `__module__`，保证：

* 现有导入路径不需要修改；
* repr / 日志 / pickle 等场景中仍显示为 `gecko.core.prompt.PromptTemplate`。

---

## 2. 依赖 & 基础约定

### 2.1 Jinja2 依赖

* 默认模板格式使用 **Jinja2** 语法；
* 通过 `jinja_env.get_jinja2_env()` 懒加载：

  * 未安装时，在首次使用时抛出 `ImportError`；
  * 避免在不使用 Prompt 功能时引入多余依赖。

安装方式示例：

```bash
pip install jinja2
# 或
rye add jinja2
```

### 2.2 模板格式

`PromptTemplate.template_format` 支持两种值：

* `"jinja2"`（默认）：
  使用 `{{ var }}`, `{% if %}`, `{% for %}` 等 Jinja2 语法。
* `"f-string"`：
  **实际上是 `str.format()` 风格**，例如：

  ```python
  template = "Hello {name}"
  template.format(name="Alice")
  ```

> 注意：这里的 `"f-string"` 为历史命名保留，不是 Python 原生 `f"..."` 语法。

---

## 3. 核心类：`PromptTemplate`

定义位置：`gecko/core/prompt/template.py`
导出路径：`from gecko.core.prompt import PromptTemplate`

### 3.1 基本用法（Jinja2 模板）

```python
from gecko.core.prompt import PromptTemplate

tpl = PromptTemplate(
    template="Hello, {{ name }}! You are {{ age }} years old.",
    input_variables=["name", "age"],
)

result = tpl.format(name="Alice", age=25)
print(result)
# Hello, Alice! You are 25 years old.
```

### 3.2 partial() 部分应用

**修复后的语义**：partial 不再直接渲染模板，而是 **预绑定参数**。

```python
tpl = PromptTemplate(
    template="Hello, {{ name }}! You are {{ age }}.",
    input_variables=["name", "age"],
)

p = tpl.partial(name="Alice")  # 预绑定 name
print(p.input_variables)       # ["age"]

print(p.format(age=25))
# Hello, Alice! You are 25.
```

### 3.3 format_safe() 宽松模式

自动为缺失变量填充默认值：

```python
tpl = PromptTemplate(
    template="User: {{ user_input }}\nHistory: {{ history }}",
    input_variables=["user_input"],
)

s = tpl.format_safe(user_input="hello")
print(s)
# History 会被填成 [] 或 "<MISSING: history>"，不会抛异常
```

### 3.4 从文件加载模板

```python
tpl = PromptTemplate.from_file("./prompts/system.txt")
print(tpl.input_variables)  # 自动从模板中提取变量并排序
```

### 3.5 从 examples 构建 few-shot 模板

```python
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+5", "output": "8"},
]

tpl = PromptTemplate.from_examples(examples)
print(tpl.template)
# 生成一个“示例 + 分隔符”的大 Prompt
```

---

## 4. 预定义模板库：`PromptLibrary`

定义位置：`gecko/core/prompt/library.py`
导出路径：`from gecko.core.prompt import PromptLibrary, DEFAULT_REACT_PROMPT`

### 4.1 ReAct 推理模板

```python
from gecko.core.prompt import PromptLibrary

tpl = PromptLibrary.get_react_prompt()
prompt = tpl.format(
    tools=[{"name": "search", "description": "Web search tool"}],
    question="What is AI?",
)
print(prompt)
```

### 4.2 通用对话模板 `get_chat_prompt()`

```python
from gecko.core.prompt import PromptLibrary

chat_tpl = PromptLibrary.get_chat_prompt()

prompt = chat_tpl.format(user_input="你好，今天天气如何？")
print(prompt)
```

**内部行为**：

* 模板中使用 `system` 和 `history`；
* 在 StrictUndefined 模式下，本来未提供会报错；
* 重构后通过 `partial(system=None, history=[])` 为其提供默认值；
* 调用方只需提供 `user_input` 即可正常使用。

### 4.3 其它预置模板

* `get_summarization_prompt()`：摘要模板；
* `get_extraction_prompt()`：信息提取模板（JSON 输出）；
* `get_translation_prompt()`：翻译模板；

以及兼容旧接口的：

* `DEFAULT_REACT_PROMPT`：原有默认 ReAct 风格模板。

---

## 5. Prompt 组合器：`PromptComposer`

定义位置：`gecko/core/prompt/composer.py`
导出路径：`from gecko.core.prompt import PromptComposer, PromptSection`

### 5.1 背景

在实际项目中，一个完整 Prompt 往往由多个部分组成，例如：

* system 角色设定；
* few-shot 示例；
* 当前任务描述；
* 约束与输出要求。

`PromptComposer` 用来 **声明式地管理这些 Section**，并能：

* 直接渲染出最终字符串；
* 或组合成新的 `PromptTemplate`。

### 5.2 基本用法

```python
from gecko.core.prompt import PromptComposer, PromptTemplate

composer = PromptComposer()

# 1）System 部分
composer.add_text_section(
    name="system",
    text="You are a senior BMS and AI expert.",
)

# 2）任务模板部分
task_tpl = PromptTemplate(
    template="User question: {{ question }}",
    input_variables=["question"],
)
composer.add_template_section(
    name="task",
    template=task_tpl,
)

# 3）渲染直接字符串
prompt_str = composer.render(question="如何设计电池预测性维护方案？")
print(prompt_str)

# 4）或转换为一个新的 PromptTemplate
final_tpl = composer.to_template()
print(final_tpl.template)
print(final_tpl.input_variables)  # ["question"]
```

### 5.3 Section 开关与前后缀

```python
composer.disable_section("system")      # 动态禁用 system 部分
composer.enable_section("system")       # 再开启

composer.add_text_section(
    name="footer",
    text="请用中文回答。",
    prefix="\n\n---\n",                # Section 前缀
)
```

---

## 6. Prompt 验证 / Lint：`PromptValidator` & `lint_prompt`

定义位置：`gecko/core/prompt/validators.py`
导出路径：

```python
from gecko.core.prompt import (
    IssueSeverity,
    PromptIssue,
    PromptValidator,
    lint_prompt,
)
```

### 6.1 主要功能

`PromptValidator` 可以对 `PromptTemplate` 做一些静态检查，例如：

* 模板中使用了但未在 `input_variables` 声明的变量（UNDECLARED_VAR）；
* `input_variables` 中声明但实际模板未使用的变量（UNUSED_INPUT_VAR）；
* 使用了不在 `allowed_variables` 白名单中的变量（UNKNOWN_VAR）；
* 模板长度超过指定上限（PROMPT_TOO_LONG）；
* 包含某些不推荐的短语（BANNED_PHRASE）。

### 6.2 快速 Lint 用法

```python
from gecko.core.prompt import PromptTemplate, lint_prompt

tpl = PromptTemplate(
    template="Hello {{ name }}, {{ role }}",
    input_variables=["name"],
)

issues = lint_prompt(tpl)

for issue in issues:
    print(issue.code, issue.severity, issue.message)
```

### 6.3 自定义 Validator

```python
from gecko.core.prompt import PromptTemplate, PromptValidator, IssueSeverity

validator = PromptValidator(
    max_length=2000,
    length_severity=IssueSeverity.WARNING,
    banned_phrases=["As an AI language model"],
)

tpl = PromptTemplate(
    template="As an AI language model, I cannot ...",
    input_variables=[],
)

issues = validator.validate(tpl)

for issue in issues:
    print(issue.code, issue.message, issue.hint)
```

---

## 7. Prompt 注册中心 / 版本管理：`PromptRegistry`

定义位置：`gecko/core/prompt/registry.py`
导出路径：

```python
from gecko.core.prompt import (
    PromptRecord,
    PromptRegistry,
    default_registry,
    register_prompt,
    get_prompt,
    list_prompts,
)
```

### 7.1 设计目标

为 Prompt 提供一种“配置中心”式的访问方式：

* 按 `name + version` 注册和获取模板；
* 支持描述、标签、元数据；
* 支持简单的版本解析策略（latest / default / 最大版本号）。

### 7.2 使用全局 `default_registry`

```python
from gecko.core.prompt import PromptTemplate, register_prompt, get_prompt

tpl_v1 = PromptTemplate(
    template="Hello {{ name }} (v1)",
    input_variables=["name"],
)
tpl_v2 = PromptTemplate(
    template="Hello {{ name }} (v2)",
    input_variables=["name"],
)

# 注册两个版本
register_prompt(
    name="greeting.simple",
    version="v1",
    template=tpl_v1,
    description="打招呼模板 v1",
    tags={"greeting"},
)

register_prompt(
    name="greeting.simple",
    version="latest",
    template=tpl_v2,
    description="打招呼模板最新版本",
    tags={"greeting", "latest"},
)

# 未指定版本时，会按 resolve_version 策略选择 "latest"
tpl = get_prompt("greeting.simple")
print(tpl.format(name="Alice"))  # 使用 v2 模板
```

### 7.3 列表与筛选

```python
from gecko.core.prompt import list_prompts

records = list_prompts(tags={"greeting"})
for rec in records:
    print(rec.name, rec.version, rec.tags)
```

> 将来如果需要接入 DB / 配置中心，可以在现有 `PromptRegistry` 上做子类扩展替换。

---

## 8. 扩展建议 & 后续演进

基于当前架构，可以很方便地进一步演化：

* **模板管理平台**：

  * 前端 UI + 后端 API 调用 `PromptRegistry` 管理 Prompt；
  * 支持审批、灰度、环境隔离（dev/stage/prod）。

* **Prompt 质量控制**：

  * CI 中自动调用 `lint_prompt()` 对新提交的 Prompt 做 Lint；
  * 与代码 Review 集成，避免上线“问题 Prompt”。

* **Prompt A/B Test / 多版本切换**：

  * 在 `PromptRegistry.resolve_version()` 中引入更多策略能力（semver 解析、流量切分等）；
  * 配合你现有的实验平台使用。

---

## 9. 快速接口一览（从 `gecko.core.prompt` 导入）

```python
from gecko.core.prompt import (
    # 核心模板
    PromptTemplate,

    # 预定义模板库
    PromptLibrary,
    DEFAULT_REACT_PROMPT,

    # Jinja2 环境（一般仅框架/内部使用）
    get_jinja2_env,
    check_jinja2,

    # 组合器
    PromptSection,
    PromptComposer,

    # 验证 / Lint
    IssueSeverity,
    PromptIssue,
    PromptValidator,
    lint_prompt,

    # 注册中心 / 版本管理
    PromptRecord,
    PromptRegistry,
    default_registry,
    register_prompt,
    get_prompt,
    list_prompts,
)
```

> 建议业务层优先使用：
> `PromptTemplate` + `PromptLibrary` + `PromptComposer` + `register_prompt` / `get_prompt`，
> 再在 CI / 工具链中接入 `PromptValidator` / `lint_prompt` 做质量控制。