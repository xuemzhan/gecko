# gecko/core/prompt/validators.py
"""
Prompt 验证 / Lint 模块

职责：
- 针对 PromptTemplate 提供一套基础的“静态检查”能力；
- 帮助发现潜在问题，如：
    - 模板中使用了但未声明的变量；
    - 声明了但实际上未使用的变量；
    - 使用了不在允许集合中的变量；
    - Prompt 过长；
    - 包含一些“反模式”词句（如 As an AI language model...）。

设计目标：
- 提供可扩展的 PromptValidator 类：
    - 可配置最大长度、禁用短语、开启/关闭不同规则；
    - validate() 返回结构化的 PromptIssue 列表；
- 提供顶层 lint_prompt() 便捷函数，方便快速使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

from gecko.core.logging import get_logger
from gecko.core.prompt.template import PromptTemplate

logger = get_logger(__name__)


class IssueSeverity(str, Enum):
    """
    诊断问题的严重程度枚举。

    分级：
        - INFO    : 信息提示，一般不需要立即处理；
        - WARNING : 警告，建议关注并优化；
        - ERROR   : 错误，可能导致运行时异常或明显的不符合预期。
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class PromptIssue:
    """
    单条 Prompt 验证 / Lint 结果。

    字段说明：
        code:
            规则编码，用于程序化处理，如 "UNDECLARED_VAR" / "PROMPT_TOO_LONG"。
        message:
            人类可读的错误/警告描述。
        severity:
            严重等级（INFO / WARNING / ERROR）。
        hint:
            可选的修复建议或提示说明。
        extra:
            额外的上下文信息，如变量名列表、长度等，方便日志与调试。
    """

    code: str
    message: str
    severity: IssueSeverity
    hint: Optional[str] = None
    extra: Dict[str, Any] = None # type: ignore


class PromptValidator:
    """
    Prompt 验证器（Validator）

    通过组合多条简单规则，对 PromptTemplate 进行静态检查。

    可配置项：
        max_length:
            PromptTemplate.template 允许的最大字符数，超过则给出 WARNING 或 ERROR；
        length_severity:
            长度超限的严重等级（默认 WARNING）；
        banned_phrases:
            禁用短语列表，例如 ["As an AI language model"]；
        enabled_rules:
            启用的规则集合（为空则全部启用）。

    当前内置规则：
        - UNDECLARED_VAR:
            模板中使用的变量未在 input_variables 声明；
        - UNUSED_INPUT_VAR:
            input_variables 中声明的变量在模板中未使用；
        - UNKNOWN_VAR:
            模板中使用的变量不在 allowed_variables 集合中（若提供）；
        - PROMPT_TOO_LONG:
            模板长度超过 max_length；
        - BANNED_PHRASE:
            模板中包含 banned_phrases 中的短语。
    """

    # 规则编码常量，便于外部统一引用
    RULE_UNDECLARED_VAR = "UNDECLARED_VAR"
    RULE_UNUSED_INPUT_VAR = "UNUSED_INPUT_VAR"
    RULE_UNKNOWN_VAR = "UNKNOWN_VAR"
    RULE_PROMPT_TOO_LONG = "PROMPT_TOO_LONG"
    RULE_BANNED_PHRASE = "BANNED_PHRASE"

    def __init__(
        self,
        max_length: int = 4000,
        length_severity: IssueSeverity = IssueSeverity.WARNING,
        banned_phrases: Optional[Iterable[str]] = None,
        enabled_rules: Optional[Set[str]] = None,
    ) -> None:
        self.max_length = max_length
        self.length_severity = length_severity
        self.banned_phrases: List[str] = list(banned_phrases or [])
        # enabled_rules = None 表示启用所有规则
        self.enabled_rules = enabled_rules

    # ========== 对外入口 ==========

    def validate(
        self,
        template: PromptTemplate,
        allowed_variables: Optional[Set[str]] = None,
    ) -> List[PromptIssue]:
        """
        对给定模板执行所有启用的规则检查，返回问题列表。

        参数：
            template:
                要检查的 PromptTemplate；
            allowed_variables:
                若提供，表示「允许出现的变量名」全集，
                多用于检查“模板是否越权访问了不该用的变量”。

        返回：
            PromptIssue 列表；若为空列表，表示未发现问题。
        """
        issues: List[PromptIssue] = []

        used_vars = template.get_variables_from_template()
        declared_vars = set(template.input_variables)
        allowed = allowed_variables or set()

        # 规则 1：使用了但未声明的变量
        if self._is_rule_enabled(self.RULE_UNDECLARED_VAR):
            issues.extend(
                self._check_undeclared_variables(
                    used_vars=used_vars, declared_vars=declared_vars
                )
            )

        # 规则 2：声明了但未使用的变量
        if self._is_rule_enabled(self.RULE_UNUSED_INPUT_VAR):
            issues.extend(
                self._check_unused_input_variables(
                    used_vars=used_vars, declared_vars=declared_vars
                )
            )

        # 规则 3：变量不在 allowed_variables 集合中
        if allowed and self._is_rule_enabled(self.RULE_UNKNOWN_VAR):
            issues.extend(
                self._check_unknown_variables(
                    used_vars=used_vars, allowed_vars=allowed
                )
            )

        # 规则 4：Prompt 过长
        if self._is_rule_enabled(self.RULE_PROMPT_TOO_LONG):
            issues.extend(self._check_prompt_length(template))

        # 规则 5：包含禁用短语
        if self.banned_phrases and self._is_rule_enabled(self.RULE_BANNED_PHRASE):
            issues.extend(self._check_banned_phrases(template))

        return issues

    # ========== 单条规则实现 ==========

    def _is_rule_enabled(self, rule_code: str) -> bool:
        """
        判断某条规则是否启用。

        enabled_rules 为 None 表示全部启用。
        """
        if self.enabled_rules is None:
            return True
        return rule_code in self.enabled_rules

    def _check_undeclared_variables(
        self,
        used_vars: Set[str],
        declared_vars: Set[str],
    ) -> List[PromptIssue]:
        """
        检查：模板中使用的变量未在 input_variables 中声明。
        """
        undeclared = used_vars - declared_vars
        if not undeclared:
            return []

        return [
            PromptIssue(
                code=self.RULE_UNDECLARED_VAR,
                message=(
                    f"模板中使用了未声明的变量: {', '.join(sorted(undeclared))}。"
                ),
                severity=IssueSeverity.WARNING,
                hint="考虑将这些变量加入 input_variables，或修改模板不再使用它们。",
                extra={
                    "used_vars": sorted(used_vars),
                    "declared_vars": sorted(declared_vars),
                    "undeclared_vars": sorted(undeclared),
                },
            )
        ]

    def _check_unused_input_variables(
        self,
        used_vars: Set[str],
        declared_vars: Set[str],
    ) -> List[PromptIssue]:
        """
        检查：input_variables 中声明的变量在模板中未使用。
        """
        unused = declared_vars - used_vars
        if not unused:
            return []

        return [
            PromptIssue(
                code=self.RULE_UNUSED_INPUT_VAR,
                message=(
                    f"以下 input_variables 没有在模板中使用: {', '.join(sorted(unused))}。"
                ),
                severity=IssueSeverity.INFO,
                hint="可以考虑移除这些未使用变量，以简化调用方接口。",
                extra={
                    "used_vars": sorted(used_vars),
                    "declared_vars": sorted(declared_vars),
                    "unused_vars": sorted(unused),
                },
            )
        ]

    def _check_unknown_variables(
        self,
        used_vars: Set[str],
        allowed_vars: Set[str],
    ) -> List[PromptIssue]:
        """
        检查：模板变量不在 allowed_variables 集合中。

        用于场景：
            - 限制模板只能使用某些“白名单变量”；
            - 防止模板越权访问敏感上下文数据。
        """
        unknown = used_vars - allowed_vars
        if not unknown:
            return []

        return [
            PromptIssue(
                code=self.RULE_UNKNOWN_VAR,
                message=(
                    f"模板中使用了不在允许列表中的变量: {', '.join(sorted(unknown))}。"
                ),
                severity=IssueSeverity.WARNING,
                hint="请检查这些变量是否应该出现在模板中，或更新 allowed_variables。",
                extra={
                    "used_vars": sorted(used_vars),
                    "allowed_vars": sorted(allowed_vars),
                    "unknown_vars": sorted(unknown),
                },
            )
        ]

    def _check_prompt_length(self, template: PromptTemplate) -> List[PromptIssue]:
        """
        检查：模板字符串长度是否超过 max_length。
        """
        length = len(template.template)
        if length <= self.max_length:
            return []

        return [
            PromptIssue(
                code=self.RULE_PROMPT_TOO_LONG,
                message=(
                    f"模板长度为 {length} 字符，超过了配置的上限 {self.max_length}。"
                ),
                severity=self.length_severity,
                hint="建议拆分为多个模块化模板，或进行内容精简，以减少 token 消耗。",
                extra={
                    "length": length,
                    "max_length": self.max_length,
                },
            )
        ]

    def _check_banned_phrases(self, template: PromptTemplate) -> List[PromptIssue]:
        """
        检查：模板中是否包含禁用短语。
        """
        content = template.template
        hits = [p for p in self.banned_phrases if p in content]
        if not hits:
            return []

        return [
            PromptIssue(
                code=self.RULE_BANNED_PHRASE,
                message=(
                    f"模板中包含以下不建议出现的短语: {', '.join(hits)}。"
                ),
                severity=IssueSeverity.WARNING,
                hint="考虑改写这些内容，以避免模型重复套话或暴露实现细节。",
                extra={
                    "banned_phrases": self.banned_phrases,
                    "hits": hits,
                },
            )
        ]


# ========== 便捷函数 ==========

def lint_prompt(
    template: PromptTemplate,
    allowed_variables: Optional[Set[str]] = None,
    validator: Optional[PromptValidator] = None,
) -> List[PromptIssue]:
    """
    便捷函数：对模板执行一次 Lint，返回问题列表。

    参数：
        template:
            待检查的 PromptTemplate；
        allowed_variables:
            可选的变量白名单；
        validator:
            若希望复用已有 PromptValidator 实例，则传入；
            若为 None，则使用默认配置创建一个临时实例。

    用法示例：
        ```python
        from gecko.core.prompt import PromptTemplate, lint_prompt

        tpl = PromptTemplate(
            template="Hello {{ name }}!",
            input_variables=[],
        )
        issues = lint_prompt(tpl)
        for issue in issues:
            print(issue.code, issue.severity, issue.message)
        ```
    """
    if validator is None:
        validator = PromptValidator(
            banned_phrases=["As an AI language model"],  # 示例默认配置
        )
    return validator.validate(template, allowed_variables=allowed_variables)


__all__ = ["IssueSeverity", "PromptIssue", "PromptValidator", "lint_prompt"]
