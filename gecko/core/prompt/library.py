# gecko/core/prompt/library.py
"""
预定义 Prompt 模板库模块

职责：
- 提供一组常用的 Prompt 模板工厂方法，供业务层直接使用；
- 集中管理 ReAct / Chat / 摘要 / 提取 / 翻译等常见模板；
- 保持与旧版 prompt.py 中 PromptLibrary 行为兼容。

可扩展点：
- 若后续需要更多标准模板（例如 Code Review 模板、SQL 生成模板等），
  可以在该模块中继续扩展静态方法；
- 若数量较多，也可以按业务拆分多个子库，再在 __init__.py 中统一 re-export。
"""

from __future__ import annotations

from gecko.core.prompt.template import PromptTemplate


class PromptLibrary:
    """
    常用 Prompt 模板库。

    用法示例：
        ```python
        from gecko.core.prompt import PromptLibrary

        tpl = PromptLibrary.get_react_prompt()
        prompt_str = tpl.format(tools=[...], question="What is AI?")
        ```
    """

    @staticmethod
    def get_react_prompt() -> PromptTemplate:
        """
        获取 ReAct 风格推理模板。

        特点：
            - 支持工具列表展示；
            - 约定工具调用输出格式；
            - 引导模型 "Let's think step by step"。
        """
        template = """You are a helpful AI assistant with access to tools.

{% if tools %}
Available Tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}

To use a tool, respond with a tool call in the following format:
Action: tool_name
Action Input: {"param": "value"}

Then wait for the observation before continuing.

Question: {{ question }}

Let's think step by step."""
        return PromptTemplate(
            template=template,
            input_variables=["tools", "question"],
        )

    @staticmethod
    def get_chat_prompt() -> PromptTemplate:
        """
        获取通用对话模板。

        模板结构：
            - 可选 system 提示（如系统角色设定）；
            - history 列表，用于多轮对话上下文；
              每个元素建议为：{"role": "...", "content": "..."}；
            - user_input 表示当前用户最新输入；
            - 模板以 "Assistant:" 结尾，方便模型补全。

        实现细节：
            - 在 Jinja2 StrictUndefined 模式下，未提供 system/history 会抛错；
            - 为保证“system / history 可选”的设计语义，这里采用：
                1) 在 input_variables 中声明 ["user_input", "system", "history"]；
                2) 使用 partial(system=None, history=[]) 绑定默认值；
            - 对调用方来说，只需要提供 user_input 即可完成渲染。
        """
        template = """{% if system %}{{ system }}

{% endif %}{% for message in history %}{{ message.role }}: {{ message.content }}
{% endfor %}User: {{ user_input }}
Assistant:"""

        base = PromptTemplate(
            template=template,
            input_variables=["user_input", "system", "history"],
        )
        return base.partial(system=None, history=[])

    @staticmethod
    def get_summarization_prompt() -> PromptTemplate:
        """
        获取摘要模板。

        约定：
            - max_words 控制摘要的词数上限；
            - text 为待摘要文本。
        """
        template = """Please summarize the following text in {{ max_words }} words or less:

{{ text }}

Summary:"""
        return PromptTemplate(
            template=template,
            input_variables=["text", "max_words"],
        )

    @staticmethod
    def get_extraction_prompt() -> PromptTemplate:
        """
        获取信息提取模板。

        约定：
            - fields: 需要提取的字段名称列表（字符串）；
            - text: 原始文本；
            - 要求模型以 JSON 格式返回提取结果。
        """
        template = """Extract the following information from the text:

{% for field in fields %}
- {{ field }}
{% endfor %}

Text: {{ text }}

Respond in JSON format."""
        return PromptTemplate(
            template=template,
            input_variables=["fields", "text"],
        )

    @staticmethod
    def get_translation_prompt() -> PromptTemplate:
        """
        获取翻译模板。

        约定：
            - source_lang: 源语言名称（如 "Chinese"）；
            - target_lang: 目标语言名称（如 "English"）；
            - text: 待翻译文本。
        """
        template = """Translate the following text from {{ source_lang }} to {{ target_lang }}:

{{ text }}

Translation:"""
        return PromptTemplate(
            template=template,
            input_variables=["source_lang", "target_lang", "text"],
        )


# ===== 原有默认模板（兼容旧接口）=====

DEFAULT_REACT_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant.
Current time: {{ current_time }}

{% if tools %}
You have access to the following tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}

Answer the user's question using the tools if necessary.
""",
    input_variables=["current_time", "tools"],
)


__all__ = ["PromptLibrary", "DEFAULT_REACT_PROMPT"]
