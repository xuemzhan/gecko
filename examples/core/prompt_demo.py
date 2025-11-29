# examples/prompt_demo.py
"""
Gecko Prompt 模板系统 Demo

涵盖内容：
1. PromptTemplate 基础用法（变量、条件、循环、from_examples 等）
2. partial() 部分填充（预绑定变量）
3. format_safe() 宽松格式化
4. PromptLibrary 预定义模板（ReAct / Chat / 摘要 / 提取 / 翻译）
5. PromptComposer 组合多个 Prompt 片段
6. PromptValidator 对 Prompt 做 Lint / 质量检查
7. PromptRegistry 对 Prompt 做版本管理
8. 使用 Zhipu LLM + PromptTemplate 进行一次真实调用（需要 ZHIPU_API_KEY）
"""

from __future__ import annotations

import asyncio
import os

from gecko.core.prompt import (
    PromptTemplate,
    PromptLibrary,
    PromptComposer,
    PromptValidator,
    lint_prompt,
    register_prompt,
    get_prompt,
)
from gecko.core.builder import AgentBuilder
from gecko.plugins.models import ZhipuChat


def basic_prompt_demos() -> None:
    print("=== Gecko Prompt 模板示例 ===\n")

    # 1. 基础模板
    print("1. 基础模板")
    template = PromptTemplate(
        template="Hello, {{ name }}! You are {{ age }} years old.",
        input_variables=["name", "age"],
    )
    result = template.format(name="Alice", age=25)
    print(f"   结果: {result}\n")

    # 2. 带条件的模板
    print("2. 带条件的模板")
    conditional = PromptTemplate(
        template="""
{% if premium %}
Welcome, Premium User {{ name }}!
You have access to advanced features.
{% else %}
Welcome, {{ name }}!
Consider upgrading to Premium.
{% endif %}
        """,
        input_variables=["name", "premium"],
    )
    print("   Premium 用户：")
    print(conditional.format(name="Bob", premium=True))
    print("   普通用户：")
    print(conditional.format(name="Charlie", premium=False))

    # 3. 带循环的模板
    print("\n3. 带循环的模板")
    loop_template = PromptTemplate(
        template="""
Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
        """,
        input_variables=["tools"],
    )
    tools = [
        {"name": "search", "description": "Search the web"},
        {"name": "calculator", "description": "Perform calculations"},
    ]
    print(loop_template.format(tools=tools))

    # 4. 自动提取变量
    print("4. 自动提取变量")
    auto_template = PromptTemplate(
        template="User {{ user }} asked: {{ question }}",
    )
    detected_vars = auto_template.get_variables_from_template()
    print(f"   检测到的变量: {detected_vars}")
    auto_template.input_variables = list(detected_vars)
    print(
        "   格式化:",
        auto_template.format(user="Alice", question="What is AI?"),
        "\n",
    )

    # 5. 部分填充（partial 预绑定变量）
    print("5. 部分填充（partial 预绑定变量）")
    partial_template = PromptTemplate(
        template="Translate {{ text }} from {{ source }} to {{ target }}",
        input_variables=["text", "source", "target"],
    )
    # 固定源语言和目标语言（partial 不会立即渲染，只是绑定默认值）
    partial = partial_template.partial(source="English", target="Chinese")
    print(f"   剩余变量: {partial.input_variables}")
    print(f"   结果: {partial.format(text='Hello')}")
    # 覆盖 partial 预设值
    print(
        "   覆盖预设值:",
        partial.format(text="Hi", source="French", target="German"),
        "\n",
    )

    # 6. 安全格式化（缺少变量时不抛异常）
    print("6. 安全格式化 format_safe（缺少变量）")
    result = template.format_safe(name="David")  # 缺少 age
    print(f"   结果: {result}\n")

    # 8. Few-shot 模板（from_examples）
    print("7. Few-shot 学习模板（from_examples）")
    examples = [
        {"input": "2 + 2", "output": "4"},
        {"input": "5 + 3", "output": "8"},
        {"input": "10 - 7", "output": "3"},
    ]
    few_shot = PromptTemplate.from_examples(
        examples,
        template="Q: {{ input }}\nA: {{ output }}",
    )
    print("   生成的 Few-shot 模板内容：")
    print(few_shot.template)
    print()


def library_demos() -> None:
    """演示 PromptLibrary 提供的预定义模板。"""
    print("8. 使用预定义模板库 PromptLibrary")

    # 8.1 ReAct 模板
    print("8.1 ReAct 模板")
    react_template = PromptLibrary.get_react_prompt()
    react_prompt = react_template.format(
        tools=[
            {"name": "search", "description": "Search the web"},
        ],
        question="What is the capital of France?",
    )
    print(react_prompt)
    print()

    # 8.2 Chat 对话模板
    print("8.2 Chat 对话模板")
    chat_template = PromptLibrary.get_chat_prompt()
    chat_prompt = chat_template.format(
        user_input="你好，给我推荐一本关于深度学习的中文书。",
        # system/history 已在 get_chat_prompt 内通过 partial 提供了默认值
    )
    print(chat_prompt)
    print()

    # 8.3 摘要模板
    print("8.3 摘要模板")
    summary_template = PromptLibrary.get_summarization_prompt()
    summary_prompt = summary_template.format(
        text="This is a very long text that needs to be summarized...",
        max_words=20,
    )
    print(summary_prompt)
    print()

    # 8.4 信息提取模板
    print("8.4 信息提取模板")
    extraction_template = PromptLibrary.get_extraction_prompt()
    extraction_prompt = extraction_template.format(
        fields=["name", "age"],
        text="Tom is 18 years old.",
    )
    print(extraction_prompt)
    print()

    # 8.5 翻译模板
    print("8.5 翻译模板")
    translation_template = PromptLibrary.get_translation_prompt()
    translation_prompt = translation_template.format(
        source_lang="Chinese",
        target_lang="English",
        text="你好，世界",
    )
    print(translation_prompt)
    print()


def composer_demo() -> None:
    """演示 PromptComposer 组合多个 Prompt 片段。"""
    print("9. 使用 PromptComposer 组合复杂 Prompt")

    composer = PromptComposer()

    # system 片段（说明角色）
    composer.add_text_section(
        name="system",
        text="You are a senior BMS and AI expert.",
    )

    # few-shot 示例片段
    examples_tpl = PromptTemplate.from_examples(
        examples=[
            {"input": "什么是 BMS？", "output": "BMS 是电池管理系统。"},
            {"input": "什么是 SOC？", "output": "SOC 是电池荷电状态。"},
        ],
        template="Q: {{ input }}\nA: {{ output }}",
    )
    composer.add_template_section(
        name="examples",
        template=examples_tpl,
        prefix="\n\nExamples:\n",
    )

    # 当前任务片段
    task_tpl = PromptTemplate(
        template="\n\nUser question: {{ question }}",
        input_variables=["question"],
    )
    composer.add_template_section(
        name="task",
        template=task_tpl,
    )

    # 直接渲染
    rendered = composer.render(
        question="请用面向产品经理的语言解释一下 Gecko 多智能体框架的价值。",
    )
    print("   直接渲染结果：")
    print(rendered)

    # 合成新的 PromptTemplate
    combined = composer.to_template()
    print("\n   合成后的 PromptTemplate：")
    print("   input_variables:", combined.input_variables)
    print("   template 片段预览：")
    print(combined.template[:200], "...\n")


def validator_demo() -> None:
    """演示 PromptValidator / lint_prompt 的使用。"""
    print("10. 使用 PromptValidator 对 Prompt 进行 Lint")

    tpl = PromptTemplate(
        template=(
            "As an AI language model, I cannot do everything.\n"
            "Hello {{ used }} and {{ unknown }}."
        ),
        input_variables=["used", "unused"],
    )

    validator = PromptValidator(
        max_length=200,
        banned_phrases=["As an AI language model"],
    )

    issues = validator.validate(tpl, allowed_variables={"used"})
    if not issues:
        print("   未发现问题。")
        return

    for i, issue in enumerate(issues, start=1):
        print(f"   问题 {i}: [{issue.severity}] {issue.code}")
        print(f"     描述: {issue.message}")
        if issue.hint:
            print(f"     建议: {issue.hint}")
    print()


def registry_demo() -> None:
    """演示 PromptRegistry / register_prompt / get_prompt 的使用。"""
    print("11. 使用 PromptRegistry 做 Prompt 版本管理")

    name = "demo.greeting"
    # 注册一个简单的打招呼模板
    tpl_v1 = PromptTemplate(
        template="Hi {{ name }}, this is v1.",
        input_variables=["name"],
    )
    register_prompt(
        name=name,
        version="v1",
        template=tpl_v1,
        description="Greeting template v1",
        tags={"demo", "greeting"},
    )

    # 再注册一个 latest 版本
    tpl_latest = PromptTemplate(
        template="Hi {{ name }}, this is the LATEST version.",
        input_variables=["name"],
    )
    register_prompt(
        name=name,
        version="latest",
        template=tpl_latest,
        description="Greeting template latest",
        tags={"demo", "greeting", "latest"},
    )

    # 不带 version 获取 -> 会根据策略拿到 latest
    tpl = get_prompt(name)
    result = tpl.format(name="Katherine") # type: ignore
    print(f"   使用 latest 版本：{result}")

    # 指定版本获取
    tpl_old = get_prompt(name, version="v1")
    result_old = tpl_old.format(name="Katherine") # type: ignore
    print(f"   使用 v1 版本：{result_old}\n")

async def zhipu_llm_demo() -> None:
    """
    使用 Zhipu LLM + PromptTemplate 进行一次真实调用。

    要求：
        - 环境变量 ZHIPU_API_KEY 必须存在；
        - 示例模型使用 glm-4-flash（可以根据需要调整）。
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("12. Zhipu LLM Demo（跳过）: 未检测到环境变量 ZHIPU_API_KEY，跳过在线调用。\n")
        return

    print("12. 使用 Zhipu LLM 调用组合后的 Prompt\n")

    # 1）构造一个稍复杂的 Prompt（这里复用 PromptLibrary 的 Chat 模板）
    chat_tpl = PromptLibrary.get_chat_prompt()
    prompt_text = chat_tpl.format(
        user_input="用通俗、专业又接地气的方式，介绍一下 Gecko 多智能体框架适合哪些业务场景。",
        # system/history 已由 partial 提供默认值，也可以在这里覆盖：
        system="你是一名资深的 AI 架构师，擅长给企业决策层做技术方案讲解。",
        history=[],
    )

    # 2）构建 Agent（使用 ZhipuChat 模型）
    agent = (
        AgentBuilder()
        .with_model(ZhipuChat(api_key=api_key, model="glm-4-flash"))
        .build()
    )

    # 3）将我们构造好的 prompt 直接作为“用户输入”传给 Agent
    print(">>> 发送给 LLM 的最终 Prompt：")
    print(prompt_text)
    print("\n>>> Zhipu LLM 返回：\n")

    output = await agent.run(prompt_text)
    # gecko.core.message.Message 的 content 字段为最终文本
    print(output.content)  # type: ignore
    print()


def main() -> None:
    basic_prompt_demos()
    library_demos()
    composer_demo()
    validator_demo()
    registry_demo()

    # 在线调用部分使用 asyncio.run 包一层
    asyncio.run(zhipu_llm_demo())


if __name__ == "__main__":
    main()