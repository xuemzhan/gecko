# gecko/core/prompt.py
"""
Prompt 模板系统

提供灵活的提示词模板管理，基于 Jinja2 实现。

核心功能：
1. 动态变量替换
2. 模板验证
3. 模板缓存
4. 常用模板库
5. 模板组合

优化点：
1. 更好的错误处理
2. 模板缓存提升性能
3. 预定义模板库
4. 模板组合和继承
5. 安全的沙箱环境
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from gecko.core.logging import get_logger

logger = get_logger(__name__)

# ===== Jinja2 相关 =====

# 延迟导入 Jinja2（避免强依赖）
_jinja2_available = None
_jinja2_env = None


def _check_jinja2():
    """检查 Jinja2 是否可用"""
    global _jinja2_available
    if _jinja2_available is None:
        try:
            import jinja2
            _jinja2_available = True
        except ImportError:
            _jinja2_available = False
    return _jinja2_available


def _get_jinja2_env():
    """获取 Jinja2 环境（带缓存）"""
    global _jinja2_env
    
    if _jinja2_env is None:
        if not _check_jinja2():
            raise ImportError(
                "PromptTemplate 依赖 jinja2。\n"
                "请安装：pip install jinja2\n"
                "或：rye add jinja2"
            )
        
        from jinja2 import Environment, StrictUndefined
        
        # 创建安全的 Jinja2 环境
        _jinja2_env = Environment(
            # 严格模式：未定义变量会报错
            undefined=StrictUndefined,
            # ✅ 修复1：禁用自动转义（直接设置为 False）
            autoescape=False,
            # 保留换行符
            keep_trailing_newline=True,
            # 启用扩展
            extensions=[]
        )
        
        logger.debug("Jinja2 environment initialized")
    
    return _jinja2_env


# ===== Prompt 模板 =====

class PromptTemplate(BaseModel):
    """
    Prompt 模板
    
    使用 Jinja2 语法，支持动态变量替换、条件判断、循环等。
    
    示例:
        ```python
        # 基础模板
        template = PromptTemplate(
            template="Hello, {{ name }}! You are {{ age }} years old.",
            input_variables=["name", "age"]
        )
        result = template.format(name="Alice", age=25)
        
        # 带条件的模板
        template = PromptTemplate(
            template='''
            {% if tools %}
            You have access to these tools:
            {% for tool in tools %}
            - {{ tool.name }}: {{ tool.description }}
            {% endfor %}
            {% endif %}
            
            User: {{ question }}
            ''',
            input_variables=["tools", "question"]
        )
        
        # 从文件加载
        template = PromptTemplate.from_file("./prompts/system.txt")
        ```
    
    属性:
        template: 模板字符串（Jinja2 语法）
        input_variables: 必需的变量列表
        template_format: 模板格式（默认 'jinja2'）
        validate_template: 是否验证模板语法（默认 True）
    """
    template: str = Field(..., description="模板字符串")
    input_variables: List[str] = Field(
        default_factory=list,
        description="必需的输入变量列表"
    )
    template_format: str = Field(
        default="jinja2",
        description="模板格式（jinja2/f-string）"
    )
    validate_template: bool = Field(
        default=True,
        description="是否验证模板语法"
    )
    
    # 私有字段：缓存编译后的模板
    _compiled_template: Any = None

    @field_validator("template_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """验证模板格式"""
        valid_formats = {"jinja2", "f-string"}
        if v not in valid_formats:
            raise ValueError(
                f"不支持的模板格式: {v}。支持的格式: {valid_formats}"
            )
        return v

    def model_post_init(self, __context):
        """初始化后验证"""
        if self.validate_template:
            self._validate_template_syntax()

    def _validate_template_syntax(self):
        """验证模板语法"""
        if self.template_format == "jinja2":
            try:
                env = _get_jinja2_env()
                # 尝试编译模板
                self._compiled_template = env.from_string(self.template)
                logger.debug("Template syntax validated")
            except Exception as e:
                error_msg = self._format_jinja2_error(str(e))
                raise ValueError(f"模板语法错误:\n{error_msg}") from e
        elif self.template_format == "f-string":
            # f-string 格式验证（基础检查）
            self._validate_fstring_syntax()

    def _validate_fstring_syntax(self):
        """验证 f-string 语法（基础）"""
        # 检查是否有未闭合的大括号
        open_count = self.template.count("{")
        close_count = self.template.count("}")
        
        if open_count != close_count:
            raise ValueError(
                f"f-string 语法错误: 大括号不匹配 "
                f"({{ {open_count} 个, }} {close_count} 个)"
            )

    def _format_jinja2_error(self, error: str) -> str:
        """格式化 Jinja2 错误信息"""
        # 提取关键信息
        lines = error.split("\n")
        formatted_lines = []
        
        for line in lines[:5]:  # 只取前 5 行
            if line.strip():
                formatted_lines.append(f"  {line}")
        
        # 添加模板片段
        template_preview = self.template[:100].replace("\n", "\\n")
        formatted_lines.append(f"\n模板片段: {template_preview}...")
        
        return "\n".join(formatted_lines)

    # ===== 格式化方法 =====

    def format(self, **kwargs: Any) -> str:
        """
        格式化模板（填充变量）
        
        参数:
            **kwargs: 模板变量
        
        返回:
            格式化后的字符串
        
        异常:
            ValueError: 缺少必需变量或渲染失败
        
        示例:
            ```python
            template = PromptTemplate(
                template="Hello, {{ name }}!",
                input_variables=["name"]
            )
            result = template.format(name="Alice")
            ```
        """
        # 检查必需变量
        missing = self._check_missing_variables(kwargs)
        if missing:
            raise ValueError(
                f"缺少必需的模板变量: {', '.join(missing)}\n"
                f"需要: {self.input_variables}\n"
                f"提供: {list(kwargs.keys())}"
            )
        
        # 根据格式渲染
        if self.template_format == "jinja2":
            return self._format_jinja2(**kwargs)
        elif self.template_format == "f-string":
            return self._format_fstring(**kwargs)
        else:
            raise ValueError(f"不支持的模板格式: {self.template_format}")

    def _check_missing_variables(self, kwargs: Dict[str, Any]) -> List[str]:
        """检查缺失的变量"""
        provided = set(kwargs.keys())
        required = set(self.input_variables)
        missing = required - provided
        return sorted(missing)

    def _format_jinja2(self, **kwargs: Any) -> str:
        """使用 Jinja2 渲染"""
        try:
            # 使用缓存的编译模板（如果有）
            if self._compiled_template is None:
                env = _get_jinja2_env()
                self._compiled_template = env.from_string(self.template)
            
            result = self._compiled_template.render(**kwargs)
            return result
            
        except Exception as e:
            # 提供更友好的错误信息
            error_msg = str(e)
            
            # 尝试识别具体错误
            if "is undefined" in error_msg:
                # 提取未定义的变量名
                match = re.search(r"'(\w+)' is undefined", error_msg)
                if match:
                    var_name = match.group(1)
                    raise ValueError(
                        f"模板变量 '{var_name}' 未定义。\n"
                        f"可用变量: {list(kwargs.keys())}"
                    ) from e
            
            # 通用错误
            raise ValueError(
                f"模板渲染失败: {error_msg}\n"
                f"模板: {self.template[:100]}..."
            ) from e

    def _format_fstring(self, **kwargs: Any) -> str:
        """使用 f-string 格式化"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"缺少变量: {e}\n"
                f"可用变量: {list(kwargs.keys())}"
            ) from e
        except Exception as e:
            raise ValueError(f"f-string 格式化失败: {e}") from e

    def format_safe(self, **kwargs: Any) -> str:
        """
        安全格式化（缺少变量时使用默认值）
        
        缺少的变量会被替换为 "<MISSING: var_name>"
        
        返回:
            格式化后的字符串
        
        注意:
            此方法会自动提取模板中的所有变量，
            不仅仅是 input_variables 中声明的变量。
        """
        try:
            all_vars = self.get_variables_from_template()
        except Exception as e:
            logger.warning("Failed to extract variables", error=str(e))
            all_vars = set(self.input_variables)
        
        safe_kwargs = dict(kwargs)
        for var in all_vars:
            if var not in safe_kwargs:
                # 智能推测默认值
                if var in ('history', 'messages', 'items', 'tools', 'examples'):
                    safe_kwargs[var] = []
                elif var in ('system', 'context', 'prefix', 'suffix'):
                    safe_kwargs[var] = None
                else:
                    safe_kwargs[var] = f"<MISSING: {var}>"
        
        try:
            if self.template_format == "jinja2":
                return self._format_jinja2(**safe_kwargs)
            elif self.template_format == "f-string":
                return self._format_fstring(**safe_kwargs)
            else:
                return f"<TEMPLATE ERROR: 不支持的格式>"
        except Exception as e:
            logger.error("Safe format failed", error=str(e))
            return f"<TEMPLATE ERROR: {e}>"

    # ===== 变量提取 =====

    def get_variables_from_template(self) -> Set[str]:
        """
        从模板中提取所有变量
        
        返回:
            变量名集合
        
        示例:
            ```python
            template = PromptTemplate(template="Hello {{ name }}, you are {{ age }}")
            vars = template.get_variables_from_template()
            # {'name', 'age'}
            ```
        """
        if self.template_format == "jinja2":
            return self._extract_jinja2_variables()
        elif self.template_format == "f-string":
            return self._extract_fstring_variables()
        return set()

    def _extract_jinja2_variables(self) -> Set[str]:
        """从 Jinja2 模板中提取变量"""
        try:
            env = _get_jinja2_env()
            from jinja2 import meta
            
            ast = env.parse(self.template)
            variables = meta.find_undeclared_variables(ast)
            return variables
        except Exception as e:
            logger.warning("Failed to extract Jinja2 variables", error=str(e))
            return set()

    def _extract_fstring_variables(self) -> Set[str]:
        """从 f-string 模板中提取变量"""
        # 简单正则匹配 {var_name}
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, self.template)
        return set(matches)

    # ===== 模板操作 =====

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """
        部分填充变量（返回新模板）
        
        参数:
            **kwargs: 要填充的变量
        
        返回:
            新的 PromptTemplate，已填充部分变量
        
        示例:
            ```python
            template = PromptTemplate(
                template="Hello {{ name }}, you are {{ age }}",
                input_variables=["name", "age"]
            )
            partial = template.partial(name="Alice")
            result = partial.format(age=25)
            ```
        """
        # 填充变量
        partial_result = self.format_safe(**kwargs)
        
        # 计算剩余变量
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        
        return PromptTemplate(
            template=partial_result,
            input_variables=remaining_vars,
            template_format=self.template_format,
            validate_template=False  # 已经验证过了
        )

    def clone(self) -> "PromptTemplate":
        """
        克隆模板
        
        返回:
            新的 PromptTemplate 实例
        """
        return PromptTemplate(
            template=self.template,
            input_variables=self.input_variables.copy(),
            template_format=self.template_format,
            validate_template=False
        )

    # ===== 工厂方法 =====

    @classmethod
    def from_file(
        cls,
        path: str,
        input_variables: Optional[List[str]] = None,
        encoding: str = "utf-8"
    ) -> "PromptTemplate":
        """
        从文件加载模板
        
        参数:
            path: 文件路径
            input_variables: 变量列表（None 则自动提取）
            encoding: 文件编码
        
        返回:
            PromptTemplate 实例
        
        示例:
            ```python
            template = PromptTemplate.from_file("./prompts/system.txt")
            ```
        """
        from pathlib import Path
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {path}")
        
        try:
            with open(file_path, "r", encoding=encoding) as f:
                template_str = f.read()
        except Exception as e:
            raise IOError(f"读取模板文件失败: {e}") from e
        
        # 创建模板
        prompt = cls(
            template=template_str,
            input_variables=input_variables or []
        )
        
        # 如果未提供变量，自动提取
        if not input_variables:
            detected_vars = prompt.get_variables_from_template()
            prompt.input_variables = sorted(detected_vars)
            logger.info(
                "Auto-detected template variables",
                path=path,
                variables=prompt.input_variables
            )
        
        return prompt

    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, str]],
        template: str = "{{ input }}\n{{ output }}\n",
        separator: str = "\n---\n"
    ) -> "PromptTemplate":
        """
        从示例列表创建 few-shot 模板
        
        参数:
            examples: 示例列表 [{"input": "...", "output": "..."}, ...]
            template: 单个示例的模板
            separator: 示例之间的分隔符
        
        返回:
            PromptTemplate 实例
        
        示例:
            ```python
            examples = [
                {"input": "2+2", "output": "4"},
                {"input": "3+5", "output": "8"},
            ]
            template = PromptTemplate.from_examples(examples)
            ```
        """
        # 渲染所有示例
        example_template = cls(template=template, input_variables=[])
        
        rendered_examples = []
        for ex in examples:
            rendered = example_template.format_safe(**ex)
            rendered_examples.append(rendered)
        
        # 合并为完整模板
        full_template = separator.join(rendered_examples)
        
        return cls(
            template=full_template,
            input_variables=[]
        )

    # ===== 字符串表示 =====

    def __str__(self) -> str:
        """简洁表示"""
        preview = self.template[:50].replace("\n", "\\n")
        if len(self.template) > 50:
            preview += "..."
        return f"PromptTemplate('{preview}', vars={self.input_variables})"

    def __repr__(self) -> str:
        """详细表示"""
        return (
            f"PromptTemplate("
            f"template_length={len(self.template)}, "
            f"input_variables={self.input_variables}, "
            f"format={self.template_format}"
            f")"
        )


# ===== 预定义模板库 =====

class PromptLibrary:
    """
    常用 Prompt 模板库
    
    提供预定义的常用模板。
    
    示例:
        ```python
        # 使用预定义模板
        template = PromptLibrary.get_react_prompt()
        prompt = template.format(
            tools=[...],
            question="What is AI?"
        )
        ```
    """
    
    @staticmethod
    def get_react_prompt() -> PromptTemplate:
        """
        获取 ReAct 推理模板
        
        返回:
            ReAct 格式的 PromptTemplate
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
            input_variables=["tools", "question"]
        )
    
    @staticmethod
    def get_chat_prompt() -> PromptTemplate:
        """
        获取对话模板
        
        返回:
            对话格式的 PromptTemplate
        """
        template = """{% if system %}{{ system }}

{% endif %}{% for message in history %}{{ message.role }}: {{ message.content }}
{% endfor %}User: {{ user_input }}
Assistant:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["user_input"],
            # system 和 history 是可选的
        )
    
    @staticmethod
    def get_summarization_prompt() -> PromptTemplate:
        """
        获取摘要模板
        
        返回:
            摘要格式的 PromptTemplate
        """
        template = """Please summarize the following text in {{ max_words }} words or less:

{{ text }}

Summary:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["text", "max_words"]
        )
    
    @staticmethod
    def get_extraction_prompt() -> PromptTemplate:
        """
        获取信息提取模板
        
        返回:
            信息提取格式的 PromptTemplate
        """
        template = """Extract the following information from the text:

{% for field in fields %}
- {{ field }}
{% endfor %}

Text: {{ text }}

Respond in JSON format."""
        
        return PromptTemplate(
            template=template,
            input_variables=["fields", "text"]
        )
    
    @staticmethod
    def get_translation_prompt() -> PromptTemplate:
        """
        获取翻译模板
        
        返回:
            翻译格式的 PromptTemplate
        """
        template = """Translate the following text from {{ source_lang }} to {{ target_lang }}:

{{ text }}

Translation:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["source_lang", "target_lang", "text"]
        )


# ===== 原有的默认模板（保持兼容性）=====

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