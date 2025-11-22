# examples/prompt_demo.py
from gecko.core.prompt import PromptTemplate, PromptLibrary


def main():
    print("=== Gecko Prompt 模板示例 ===\n")
    
    # 1. 基础模板
    print("1. 基础模板")
    template = PromptTemplate(
        template="Hello, {{ name }}! You are {{ age }} years old.",
        input_variables=["name", "age"]
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
        input_variables=["name", "premium"]
    )
    print(conditional.format(name="Bob", premium=True))
    print(conditional.format(name="Charlie", premium=False))
    
    # 3. 带循环的模板
    print("3. 带循环的模板")
    loop_template = PromptTemplate(
        template="""
Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
        """,
        input_variables=["tools"]
    )
    tools = [
        {"name": "search", "description": "Search the web"},
        {"name": "calculator", "description": "Perform calculations"},
    ]
    print(loop_template.format(tools=tools))
    
    # 4. 自动提取变量
    print("4. 自动提取变量")
    auto_template = PromptTemplate(
        template="User {{ user }} asked: {{ question }}"
    )
    detected_vars = auto_template.get_variables_from_template()
    print(f"   检测到的变量: {detected_vars}")
    auto_template.input_variables = list(detected_vars)
    print(f"   格式化: {auto_template.format(user='Alice', question='What is AI?')}\n")
    
    # 5. 部分填充
    print("5. 部分填充")
    partial_template = PromptTemplate(
        template="Translate {{ text }} from {{ source }} to {{ target }}",
        input_variables=["text", "source", "target"]
    )
    # 固定源语言和目标语言
    partial = partial_template.partial(source="English", target="Chinese")
    print(f"   剩余变量: {partial.input_variables}")
    print(f"   结果: {partial.format(text='Hello')}\n")
    
    # 6. 安全格式化（缺少变量）
    print("6. 安全格式化")
    result = template.format_safe(name="David")  # 缺少 age
    print(f"   结果: {result}\n")
    
    # 7. 从文件加载（示例）
    # print("7. 从文件加载")
    # template = PromptTemplate.from_file("./prompts/system.txt")
    
    # 8. Few-shot 模板
    print("8. Few-shot 学习模板")
    examples = [
        {"input": "2 + 2", "output": "4"},
        {"input": "5 + 3", "output": "8"},
        {"input": "10 - 7", "output": "3"},
    ]
    few_shot = PromptTemplate.from_examples(
        examples,
        template="Q: {{ input }}\nA: {{ output }}"
    )
    print(few_shot.template)
    print()
    
    # 9. 使用预定义模板
    print("9. 使用预定义模板库")
    react_template = PromptLibrary.get_react_prompt()
    react_prompt = react_template.format(
        tools=[
            {"name": "search", "description": "Search the web"},
        ],
        question="What is the capital of France?"
    )
    print(react_prompt)
    print()
    
    # 10. 摘要模板
    print("10. 摘要模板")
    summary_template = PromptLibrary.get_summarization_prompt()
    summary_prompt = summary_template.format(
        text="This is a very long text that needs to be summarized...",
        max_words=20
    )
    print(summary_prompt)


if __name__ == "__main__":
    main()