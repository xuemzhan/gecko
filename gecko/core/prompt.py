# gecko/core/prompt.py  
"""  
PromptTemplate（增强版）  
  
- 支持 input_variables 校验，缺失变量时抛出明确异常  
- 在渲染失败时提供模板片段和错误类型，便于定位  
- 延迟导入 Jinja2 以减轻依赖  
"""  
  
from __future__ import annotations  
  
from typing import Any, Dict, List, Optional  
  
from pydantic import BaseModel  
  
  
class PromptTemplate(BaseModel):  
    template: str  
    input_variables: List[str] = []  
  
    def format(self, **kwargs: Any) -> str:  
        missing = [var for var in self.input_variables if var not in kwargs]  
        if missing:  
            raise ValueError(f"缺少模板变量: {', '.join(missing)}")  
  
        try:  
            from jinja2 import Template  
        except ImportError as e:  
            raise ImportError("PromptTemplate 依赖 jinja2，请先安装：pip install jinja2") from e  
  
        try:  
            tmpl = Template(self.template)  
            return tmpl.render(**kwargs)  
        except Exception as e:  
            snippet = self.template[:80].replace("\n", "\\n")  
            raise ValueError(f"Prompt 渲染失败: {e} | 模板片段: {snippet}") from e  
  
  
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
