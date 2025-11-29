# gecko/core/prompt/composer.py
"""
Prompt 组合器模块（Composer）

职责：
- 将多个 PromptTemplate 片段（Section）按顺序组合成一个完整 Prompt；
- 支持为每个 Section 配置前后缀、是否启用等信息；
- 支持一键生成新的 PromptTemplate，或直接渲染为字符串。

设计目标：
- 解耦“模板片段组织方式”和“模板本身的渲染逻辑”；
- 提供可扩展的 Section 描述结构，将来可以挂更多配置（权重、条件、元数据等）；
- 在不破坏 PromptTemplate 现有接口的前提下，提升 Prompt 结构化管理能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable, Set

from gecko.core.logging import get_logger
from .template import PromptTemplate

logger = get_logger(__name__)


@dataclass
class PromptSection:
    """
    单个 Prompt 片段（Section）的描述。

    字段说明：
        name:
            Section 名称，用于标识用途，例如 "system" / "examples" / "task" 等。
        template:
            实际负责渲染的 PromptTemplate。
        enabled:
            是否启用该 Section，为后续做 A/B 配置或动态开关留扩展空间。
        prefix:
            可选的前缀字符串，会拼接在该 Section 渲染结果之前。
        suffix:
            可选的后缀字符串，会拼接在该 Section 渲染结果之后。

    扩展空间（未来可以考虑加入）：
        - condition: Callable[[Dict[str, Any]], bool]，按上下文决定是否启用；
        - weight / priority: 在自动排序时使用；
        - metadata: 任意结构化信息用于分析或可视化。
    """

    name: str
    template: PromptTemplate
    enabled: bool = True
    prefix: str = ""
    suffix: str = ""

    def render(self, context: Dict[str, Any]) -> str:
        """
        渲染当前 Section。

        当前策略：
            - 默认使用严格模式的 format()；
            - 如有需要，可以在上层 Composer 中选择使用 format_safe()。
        """
        logger.debug("Rendering section", extra={"name": self.name})
        body = self.template.format(**context)
        return f"{self.prefix}{body}{self.suffix}"


class PromptComposer:
    """
    Prompt 组合器（Composer）

    功能：
        - 维护一组按顺序排列的 PromptSection；
        - 支持按顺序渲染，或组合成一个新的 PromptTemplate；
        - 支持全局分隔符（global_separator）统一控制 Section 之间的连接方式。

    常见用法：
        ```python
        composer = PromptComposer()

        composer.add_text_section(
            name="system",
            text="You are a helpful assistant.",
        )

        composer.add_template_section(
            name="task",
            template=PromptTemplate(
                template="User question: {{ question }}",
                input_variables=["question"],
            )
        )

        prompt_str = composer.render(question="What is AI?")
        # 或者：
        tpl = composer.to_template()
        prompt_str = tpl.format(question="What is AI?")
        ```
    """

    def __init__(
        self,
        sections: Optional[Iterable[PromptSection]] = None,
        global_separator: str = "\n\n",
    ) -> None:
        """
        初始化组合器。

        参数：
            sections:
                初始的 Section 列表，保持顺序；
            global_separator:
                在渲染多个 Section 时用于连接各个部分的全局分隔符。
        """
        self._sections: List[PromptSection] = list(sections) if sections else []
        self.global_separator: str = global_separator

    # ==================== Section 管理 ====================

    @property
    def sections(self) -> List[PromptSection]:
        """
        返回所有 Section（按顺序）。

        注意：
            - 返回的是实际内部列表的引用，如果需要严格封装，可以改为返回副本。
        """
        return self._sections

    def add_section(self, section: PromptSection) -> "PromptComposer":
        """
        添加一个已构造好的 PromptSection。

        返回：
            self（支持链式调用）
        """
        self._sections.append(section)
        return self

    def add_template_section(
        self,
        name: str,
        template: PromptTemplate,
        enabled: bool = True,
        prefix: str = "",
        suffix: str = "",
    ) -> "PromptComposer":
        """
        使用已有 PromptTemplate 添加一个 Section。
        """
        section = PromptSection(
            name=name,
            template=template,
            enabled=enabled,
            prefix=prefix,
            suffix=suffix,
        )
        return self.add_section(section)

    def add_text_section(
        self,
        name: str,
        text: str,
        enabled: bool = True,
        prefix: str = "",
        suffix: str = "",
    ) -> "PromptComposer":
        """
        使用纯文本添加一个 Section。

        实现方式：
            - 内部会创建一个只包含固定文本的 PromptTemplate，
              不需要任何 input_variables。
        """
        tpl = PromptTemplate(
            template=text,
            input_variables=[],
        )
        section = PromptSection(
            name=name,
            template=tpl,
            enabled=enabled,
            prefix=prefix,
            suffix=suffix,
        )
        return self.add_section(section)

    def enable_section(self, name: str) -> None:
        """
        按名称启用某个 Section。
        """
        for s in self._sections:
            if s.name == name:
                s.enabled = True

    def disable_section(self, name: str) -> None:
        """
        按名称禁用某个 Section。
        """
        for s in self._sections:
            if s.name == name:
                s.enabled = False

    # ==================== 渲染 / 组合 ====================

    def render(self, **context: Any) -> str:
        """
        直接渲染所有启用的 Section，返回拼接后的字符串。

        行为：
            - 按 Section 顺序遍历；
            - 对 enabled=True 的 Section 调用 section.render(context)；
            - 使用 global_separator 连接各个片段；
            - 不自动在末尾追加额外换行。
        """
        logger.debug(
            "Rendering composed prompt",
            extra={"sections": [s.name for s in self._sections]},
        )

        parts: List[str] = []
        for section in self._sections:
            if not section.enabled:
                continue
            rendered = section.render(context)
            # 跳过完全空字符串的片段，避免产生多余空段
            if rendered.strip():
                parts.append(rendered)

        return self.global_separator.join(parts)

    def to_template(self) -> PromptTemplate:
        """
        将当前组合器转换为一个新的 PromptTemplate。

        实现方式：
            - 将每个启用的 Section 的 template.template 按顺序拼接；
            - 使用 global_separator 作为片段之间的分隔符；
            - input_variables 取所有 Section 模板的并集；
            - template_format 目前简单选择第一个 Section 的模板格式
              （假设所有 Section 一致，若不一致可未来扩展更复杂逻辑）。

        注意：
            - 该方法只组合「模板字符串」，不会直接渲染；
            - 如需要预绑定部分变量，可以对返回的 PromptTemplate 调用 partial()。
        """
        if not self._sections:
            # 空组合器返回一个空模板，方便链路调用。
            return PromptTemplate(template="", input_variables=[])

        # 只考虑 enabled=True 的 Section
        enabled_sections = [s for s in self._sections if s.enabled]
        if not enabled_sections:
            return PromptTemplate(template="", input_variables=[])

        # 1) 拼接模板字符串
        template_parts: List[str] = []
        for s in enabled_sections:
            tmpl_str = s.template.template
            # 如果 Section 自己有前后缀，也拼进去
            full_str = f"{s.prefix}{tmpl_str}{s.suffix}"
            template_parts.append(full_str)

        combined_template_str = self.global_separator.join(template_parts)

        # 2) 合并所有 input_variables（取并集）
        all_vars: Set[str] = set()
        for s in enabled_sections:
            all_vars.update(s.template.input_variables)

        # 3) 决定最终 template_format（这里简单采用第一个 Section 的格式）
        template_format = enabled_sections[0].template.template_format

        logger.debug(
            "Composed PromptTemplate",
            extra={
                "template_length": len(combined_template_str),
                "input_variables": sorted(all_vars),
                "template_format": template_format,
            },
        )

        return PromptTemplate(
            template=combined_template_str,
            input_variables=sorted(all_vars),
            template_format=template_format,
        )


__all__ = ["PromptSection", "PromptComposer"]
