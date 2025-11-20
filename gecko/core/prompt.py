# gecko/core/prompt.py
from typing import List, Any, Optional
from pydantic import BaseModel
from jinja2 import Template

class PromptTemplate(BaseModel):
    """
    Prompt 模板类
    支持 Jinja2 语法，用于动态生成 System Prompt
    """
    template: str
    input_variables: List[str] = []

    def format(self, **kwargs: Any) -> str:
        """渲染模板"""
        # 简单的校验，确保并未遗漏关键变量（可选）
        # for var in self.input_variables:
        #     if var not in kwargs:
        #         raise ValueError(f"Missing variable: {var}")
        
        try:
            tmpl = Template(self.template)
            return tmpl.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Prompt rendering failed: {e}")

# 预置一些常用模板
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
    input_variables=["current_time", "tools"]
)