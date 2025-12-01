# Prompt 管理与工程

Gecko v0.3.1 引入了全新的 Prompt 引擎，帮助开发者以工程化的方式管理提示词。

## 1. Prompt 模板

使用 `PromptTemplate` 替代硬编码的字符串。支持 Jinja2 语法。

```python
from gecko.core.prompt import PromptTemplate

# 定义模板
tpl = PromptTemplate(
    template="你是一个{{ role }}专家。请解释：{{ topic }}",
    input_variables=["role", "topic"]
)

# 渲染
prompt = tpl.format(role="Python", topic="AsyncIO")
# 输出: "你是一个Python专家。请解释：AsyncIO"
```

## 2. 组合器 (Composer)

使用 `PromptComposer` 将多个片段（System, Few-Shot, User Input）动态组合，便于复用和维护。

```python
from gecko.core.prompt import PromptComposer, PromptSection

composer = PromptComposer()

# 添加 System Prompt
composer.add_text_section("system", "你是一个有用的助手。")

# 添加 Few-Shot 示例
examples_tpl = PromptTemplate("Q: {{q}}\nA: {{a}}", ["q", "a"])
composer.add_template_section("examples", examples_tpl)

# 渲染
# 可以选择性禁用某些部分
composer.disable_section("examples") 
final_prompt = composer.render(q="...", a="...")
```

## 3. 注册表 (Registry)

使用 `PromptRegistry` 进行版本管理。方便在代码中引用 prompt 名称，而在配置中切换 prompt 内容。

```python
from gecko.core.prompt import register_prompt, get_prompt

# 注册 v1 版本
register_prompt(
    name="chat.system",
    version="v1",
    template=PromptTemplate("You are a bot.", []),
    tags={"production"}
)

# 在业务代码中获取
# 默认获取 "latest" 或 "default"
tpl = get_prompt("chat.system")
```

## 4. 静态检查 (Validator)

在运行前检查 Prompt 质量，防止低级错误或 Prompt 注入风险。

```python
from gecko.core.prompt import PromptValidator, lint_prompt

tpl = PromptTemplate("Hello {{ name }}", ["name"])

# 检查是否有未定义的变量、长度是否超限、是否包含禁用词
issues = lint_prompt(tpl)
if issues:
    print("Prompt 存在问题:", issues)
```