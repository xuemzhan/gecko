# gecko/core/prompt/template.py
"""
PromptTemplate 核心实现模块

职责：
- 定义 PromptTemplate 类，负责：
    - 模板字段与校验逻辑；
    - Jinja2 / str.format 两种渲染模式；
    - 模板语法验证；
    - 模板变量提取；
    - partial() 部分应用；
    - from_file / from_examples 等工厂方法。

设计要点：
- 使用 Pydantic BaseModel 方便与其他配置/模型统一管理；
- 使用 PrivateAttr 存放运行时缓存（编译模板、partial 参数、变量缓存）；
- 与 Jinja2 环境解耦，通过 jinja_env.get_jinja2_env() 访问；
- partial() 实现为“记录预绑定参数”，而不是立即渲染模板，避免破坏占位符。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from gecko.core.logging import get_logger
from gecko.core.prompt.jinja_env import get_jinja2_env

logger = get_logger(__name__)


class PromptTemplate(BaseModel):
    """
    Prompt 模板类

    支持两种模板格式：
        1. Jinja2（默认）：
           - 使用 {{ var }} / {% if %} / {% for %} 等语法；
           - 通过 get_jinja2_env() 获得严格模式（StrictUndefined）的 Environment；
        2. "f-string"（内部实现基于 str.format）：
           - 实际上是 Python 的 "Hello {name}".format(name="Alice") 形式；
           - 为保持历史兼容，字段名仍使用 "f-string"。

    典型用法（Jinja2）：
        ```python
        tpl = PromptTemplate(
            template="Hello, {{ name }}! You are {{ age }} years old.",
            input_variables=["name", "age"],
        )
        result = tpl.format(name="Alice", age=25)
        ```

    典型用法（partial 部分应用）：
        ```python
        tpl = PromptTemplate(
            template="Hello, {{ name }}! You are {{ age }} years old.",
            input_variables=["name", "age"],
        )
        p = tpl.partial(name="Alice")  # 预设 name
        result = p.format(age=25)      # 仅需传剩余变量
        ```

    属性说明：
        template:
            模板字符串。
        input_variables:
            渲染时要求调用方必须提供的变量列表；
            用于 format() 前的缺失变量检查。
        template_format:
            模板格式：
              - "jinja2"   : 使用 Jinja2 模板渲染；
              - "f-string" : 使用 str.format 渲染（名称为历史兼容保留）。
        validate_template:
            是否在模型初始化时立即验证模板语法。
    """

    # ------------ 公有字段（参与序列化） ------------

    template: str = Field(..., description="模板字符串")
    input_variables: List[str] = Field(
        default_factory=list,
        description="必需的输入变量列表",
    )
    template_format: str = Field(
        default="jinja2",
        description="模板格式（jinja2 / f-string，后者实际使用 str.format 语法）",
    )
    validate_template: bool = Field(
        default=True,
        description="是否在初始化时验证模板语法",
    )

    # ------------ 私有字段（运行时缓存，不参与序列化） ------------

    # 缓存已编译的 Jinja2 模板对象（避免重复 from_string）
    _compiled_template: Any = PrivateAttr(default=None)
    # partial() 预绑定的变量（部分应用参数）
    _partial_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)
    # 模板中变量名的缓存集合（减少重复 AST 解析）
    _variables_cache: Optional[Set[str]] = PrivateAttr(default=None)

    # ===== Pydantic 钩子 & 字段校验 =====

    @field_validator("template_format", mode="before")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """
        校验 template_format 字段是否合法。

        说明：
            - 为保持向后兼容，继续使用 "f-string" 命名；
            - 实现上并非 Python 原生 f-string，而是 str.format。
        """
        valid_formats = {"jinja2", "f-string"}
        if v not in valid_formats:
            raise ValueError(
                f"不支持的模板格式: {v}。支持的格式: {valid_formats}"
            )
        return v

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        """
        Pydantic v2 风格的初始化后钩子。

        在模型实例创建完成后：
            - 若 validate_template=True，则立即进行语法验证；
            - 语法错误在此阶段就会被抛出，避免运行期才发现问题。
        """
        if self.validate_template:
            self._validate_template_syntax()

    # ===== 模板语法验证相关方法 =====

    def _validate_template_syntax(self) -> None:
        """
        验证模板语法是否正确。

        - Jinja2 模式：尝试编译模板字符串为 Jinja2 Template 对象；
        - f-string 模式：做大括号数量匹配检查（基础检查）。
        """
        if self.template_format == "jinja2":
            try:
                env = get_jinja2_env()
                # from_string 本身会做语法解析，失败时抛出异常
                self._compiled_template = env.from_string(self.template)
                logger.debug("Template syntax validated")
            except Exception as e:
                error_msg = self._format_jinja2_error(str(e))
                raise ValueError(f"模板语法错误:\n{error_msg}") from e
        elif self.template_format == "f-string":
            self._validate_fstring_syntax()

    def _validate_fstring_syntax(self) -> None:
        """
        验证 "f-string"（实际为 str.format）模板语法的基础一致性。

        当前仅做：
            - 检查 '{' 和 '}' 的数量是否一致；
        若不一致，基本可以判定模板存在语法错误。
        """
        open_count = self.template.count("{")
        close_count = self.template.count("}")

        if open_count != close_count:
            raise ValueError(
                f"f-string 语法错误: 大括号不匹配 "
                f"({{ {open_count} 个, }} {close_count} 个)"
            )

    def _format_jinja2_error(self, error: str) -> str:
        """
        对 Jinja2 抛出的错误信息进行更加友好、便于排查的格式化。

        处理方式：
            - 仅保留前几行（最多 5 行）错误信息，避免过长；
            - 附带模板前 100 个字符的预览，快速定位问题位置。
        """
        lines = error.split("\n")
        formatted_lines: List[str] = []

        for line in lines[:5]:
            if line.strip():
                formatted_lines.append(f"  {line}")

        template_preview = self.template[:100].replace("\n", "\\n")
        formatted_lines.append(f"\n模板片段: {template_preview}...")

        return "\n".join(formatted_lines)

    # ===== 核心渲染接口 =====

    def format(self, **kwargs: Any) -> str:
        """
        严格模式下渲染模板。

        行为：
            1. 首先合并 partial() 预绑定参数与本次调用参数；
               - 调用参数优先级更高，可覆盖 partial 预设值；
            2. 基于 input_variables 检查是否缺少必需变量；
            3. 根据 template_format 调用对应的渲染实现。

        参数:
            **kwargs: 渲染时提供的变量。

        返回:
            渲染后的字符串。

        异常:
            - ValueError: 缺少必需变量或渲染失败。
        """
        merged_kwargs: Dict[str, Any] = {**self._partial_kwargs, **kwargs}

        # 检查必需变量
        missing = self._check_missing_variables(merged_kwargs)
        if missing:
            raise ValueError(
                f"缺少必需的模板变量: {', '.join(missing)}\n"
                f"需要: {self.input_variables}\n"
                f"提供: {list(merged_kwargs.keys())}"
            )

        # 根据格式渲染
        if self.template_format == "jinja2":
            return self._format_jinja2(**merged_kwargs)
        elif self.template_format == "f-string":
            return self._format_fstring(**merged_kwargs)
        else:
            raise ValueError(f"不支持的模板格式: {self.template_format}")

    def _check_missing_variables(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        根据 input_variables 检查缺失的变量。

        注意：
            - 只检查 input_variables 中声明的必需变量；
            - 模板中实际引用的变量集合可能比 input_variables 多，二者不强制一致。
        """
        provided = set(kwargs.keys())
        required = set(self.input_variables)
        missing = required - provided
        return sorted(missing)

    def _format_jinja2(self, **kwargs: Any) -> str:
        """
        使用 Jinja2 渲染模板。

        缓存策略：
            - 若 _compiled_template 已存在，直接使用；
            - 否则调用 get_jinja2_env().from_string() 编译并缓存。
        """
        try:
            if self._compiled_template is None:
                env = get_jinja2_env()
                self._compiled_template = env.from_string(self.template)

            result = self._compiled_template.render(**kwargs)
            return result

        except Exception as e:
            error_msg = str(e)

            # 尝试识别「变量未定义」这种常见错误并给出更明确提示
            if "is undefined" in error_msg:
                match = re.search(r"'(\w+)' is undefined", error_msg)
                if match:
                    var_name = match.group(1)
                    raise ValueError(
                        f"模板变量 '{var_name}' 未定义。\n"
                        f"可用变量: {list(kwargs.keys())}"
                    ) from e

            # 兜底错误信息，附带模板片段
            raise ValueError(
                f"模板渲染失败: {error_msg}\n"
                f"模板: {self.template[:100]}..."
            ) from e

    def _format_fstring(self, **kwargs: Any) -> str:
        """
        使用 str.format 渲染模板（对应 template_format='f-string'）。

        说明：
            - 这里并不是 Python 的 f"..." 语法，而是
              `"Hello {name}".format(name="Alice")` 的形式。
        """
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
        宽松模式渲染模板（“尽量渲染成功”，不会因为缺变量直接抛异常）。

        行为：
            1. 合并 partial 预绑定参数与本次调用参数；
            2. 尝试解析模板中实际出现的变量名；
            3. 对未提供的变量填充合理的默认值：
                - 列表相关变量：history/messages/items/tools/examples -> []
                - 上下文/字符串类变量：system/context/prefix/suffix -> None
                - 其它变量："<MISSING: var_name>" 字样；
            4. 若最终渲染仍然失败，记录日志并返回
               "<TEMPLATE ERROR: ...>" 字符串，避免异常向外冒泡。

        适用场景：
            - 调试；
            - 容忍部分信息缺失的场景（例如日志/监控生成 Prompt）。
        """
        safe_kwargs: Dict[str, Any] = {**self._partial_kwargs, **kwargs}

        try:
            all_vars = self.get_variables_from_template()
        except Exception as e:
            logger.warning("Failed to extract variables", error=str(e))
            all_vars = set(self.input_variables)

        # 填充默认值
        for var in all_vars:
            if var not in safe_kwargs:
                if var in ("history", "messages", "items", "tools", "examples"):
                    safe_kwargs[var] = []
                elif var in ("system", "context", "prefix", "suffix"):
                    safe_kwargs[var] = None
                else:
                    safe_kwargs[var] = f"<MISSING: {var}>"

        try:
            if self.template_format == "jinja2":
                return self._format_jinja2(**safe_kwargs)
            elif self.template_format == "f-string":
                return self._format_fstring(**safe_kwargs)
            else:
                return "<TEMPLATE ERROR: 不支持的格式>"
        except Exception as e:
            logger.error("Safe format failed", error=str(e))
            return f"<TEMPLATE ERROR: {e}>"

    # ===== 变量提取相关方法 =====

    def get_variables_from_template(self) -> Set[str]:
        """
        从模板中提取所有「在模板语法中被引用」的变量名。

        - 对于 Jinja2 模板：使用 jinja2.meta.find_undeclared_variables；
        - 对于 f-string（str.format）模板：使用正则匹配 {var_name} 形式。

        注意：
            - 返回结果是模板层面实际用到的变量集合，
              不一定与 input_variables 完全一致（可以超集）。
        """
        if self._variables_cache is not None:
            return set(self._variables_cache)

        if self.template_format == "jinja2":
            variables = self._extract_jinja2_variables()
        elif self.template_format == "f-string":
            variables = self._extract_fstring_variables()
        else:
            variables = set()

        self._variables_cache = set(variables)
        return variables

    def _extract_jinja2_variables(self) -> Set[str]:
        """
        从 Jinja2 模板中提取变量名集合。

        通过：
            - env.parse(template) 得到 AST；
            - meta.find_undeclared_variables(ast) 获取「未在局部声明的标识符」集合。
        """
        try:
            env = get_jinja2_env()
            from jinja2 import meta

            ast = env.parse(self.template)
            variables = meta.find_undeclared_variables(ast)
            return set(variables)
        except Exception as e:
            logger.warning("Failed to extract Jinja2 variables", error=str(e))
            return set()

    def _extract_fstring_variables(self) -> Set[str]:
        """
        从 "f-string"（str.format）模板中提取变量名。

        当前实现较为简单：
            - 使用正则 \\{(\\w+)\\} 匹配 {var_name}；
            - 不支持复杂表达式（如 {user.name}、{value:.2f} 等）。

        如有更复杂需求，可替换为 string.Formatter().parse()。
        """
        pattern = r"\{(\w+)\}"
        matches = re.findall(pattern, self.template)
        return set(matches)

    # ===== 模板操作：partial / clone =====

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """
        部分填充变量（部分应用），返回一个新的 PromptTemplate 实例。

        实现方式：
            - 不立即渲染模板字符串；
            - 将传入的变量保存到 _partial_kwargs 中；
            - 在新的实例上更新 input_variables（去掉已绑定的变量）；
            - 后续调用 new_tpl.format(...) 时，会自动合并该 partial 参数。

        示例：
            ```python
            tpl = PromptTemplate(
                template="Hello {{ name }}, you are {{ age }}",
                input_variables=["name", "age"],
            )
            p = tpl.partial(name="Alice")
            p.format(age=25)  # -> "Hello Alice, you are 25"
            ```
        """
        remaining_vars = [v for v in self.input_variables if v not in kwargs]

        new_tpl = self.clone()
        new_tpl._partial_kwargs = {**self._partial_kwargs, **kwargs}
        new_tpl.input_variables = remaining_vars
        return new_tpl

    def clone(self) -> "PromptTemplate":
        """
        克隆当前 PromptTemplate 实例。

        - template 原样复制；
        - input_variables 复制一份列表；
        - _partial_kwargs 复制一份字典；
        - _compiled_template 直接复用（懒加载缓存）；
        - _variables_cache 复制或置为 None。

        用于：
            - 需要在不影响原实例的前提下做局部修改；
            - 例如不同场景对同一模板做 slight customization。
        """
        new_tpl = PromptTemplate(
            template=self.template,
            input_variables=self.input_variables.copy(),
            template_format=self.template_format,
            validate_template=False,  # 克隆时无需再验证语法
        )
        new_tpl._partial_kwargs = dict(self._partial_kwargs)
        new_tpl._compiled_template = self._compiled_template
        new_tpl._variables_cache = (
            set(self._variables_cache) if self._variables_cache is not None else None
        )
        return new_tpl

    # ===== 工厂方法：from_file / from_examples =====

    @classmethod
    def from_file(
        cls,
        path: str,
        input_variables: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> "PromptTemplate":
        """
        从文件加载模板。

        参数:
            path:
                模板文件路径。
            input_variables:
                若提供，则直接使用；若为 None，则在创建后自动从模板中提取变量；
            encoding:
                文件编码，默认 "utf-8"。

        返回:
            PromptTemplate 实例。

        说明：
            - 自动提取变量依赖 get_variables_from_template()；
            - 会将变量名排序后写回 input_variables。
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

        prompt = cls(
            template=template_str,
            input_variables=input_variables or [],
        )

        if not input_variables:
            detected_vars = prompt.get_variables_from_template()
            prompt.input_variables = sorted(detected_vars)
            logger.info(
                "Auto-detected template variables",
                path=path,
                variables=prompt.input_variables,
            )

        return prompt

    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, str]],
        template: str = "{{ input }}\n{{ output }}\n",
        separator: str = "\n---\n",
    ) -> "PromptTemplate":
        """
        从示例列表构造 few-shot 模板。

        参数:
            examples:
                示例列表，形如 [{"input": "...", "output": "..."}, ...]；
            template:
                单个示例的模板字符串，默认 "{{ input }}\\n{{ output }}\\n"；
            separator:
                示例之间的分隔符，默认 "\\n---\\n"。

        返回:
            PromptTemplate 实例，其 template 为所有示例拼接后的字符串。
        """
        example_template = cls(template=template, input_variables=[])

        rendered_examples: List[str] = []
        for ex in examples:
            rendered = example_template.format_safe(**ex)
            rendered_examples.append(rendered)

        full_template = separator.join(rendered_examples)

        return cls(
            template=full_template,
            input_variables=[],
        )

    # ===== 字符串表示 =====

    def __str__(self) -> str:
        """
        简洁字符串表示：显示模板前 50 字符 + 变量列表。

        便于在日志中快速查看。
        """
        preview = self.template[:50].replace("\n", "\\n")
        if len(self.template) > 50:
            preview += "..."
        return f"PromptTemplate('{preview}', vars={self.input_variables})"

    def __repr__(self) -> str:
        """
        详细字符串表示：适合调试与日志。

        包含：
            - 模板长度；
            - input_variables；
            - template_format。
        """
        return (
            "PromptTemplate("
            f"template_length={len(self.template)}, "
            f"input_variables={self.input_variables}, "
            f"format={self.template_format}"
            ")"
        )


__all__ = ["PromptTemplate"]
