# gecko/core/prompt/__init__.py
"""
gecko.core.prompt 包初始化模块（对外兼容层）

重构背景：
- 原来 gecko.core.prompt 是一个单文件模块 prompt.py；
- 现在拆分为 prompt 包，下挂多个子模块（template, library, jinja_env 等）；
- 为了保持向后兼容，必须确保原有 import 写法继续生效。

兼容要求：
1. 以下写法仍然有效：
   - from gecko.core.prompt import PromptTemplate
   - from gecko.core.prompt import PromptLibrary, DEFAULT_REACT_PROMPT
   - import gecko.core.prompt as prompt; prompt.PromptTemplate(...)

2. 若老代码内部曾经使用过私有函数（不推荐但可能存在）：
   - from gecko.core.prompt import _get_jinja2_env, _check_jinja2
   可以通过轻量 alias 保留兼容性。

同时，为了给后续迭代留空间，这里还对外导出了：
- Prompt 组合器（PromptSection, PromptComposer）
- Prompt 验证 / Lint（PromptValidator, lint_prompt 等）
- Prompt 注册中心 / 版本管理（PromptRegistry, default_registry 等）
"""

from __future__ import annotations

# 核心模板 & 模板库
from gecko.core.prompt.template import PromptTemplate
from gecko.core.prompt.library import PromptLibrary, DEFAULT_REACT_PROMPT

# Jinja2 环境封装
from gecko.core.prompt.jinja_env import get_jinja2_env, check_jinja2

# Prompt 组合器
from gecko.core.prompt.composer import PromptSection, PromptComposer

# Prompt 验证 / Lint
from gecko.core.prompt.validators import (
    IssueSeverity,
    PromptIssue,
    PromptValidator,
    lint_prompt,
)

# Prompt 注册中心 / 版本管理
from gecko.core.prompt.registry import (
    PromptRecord,
    PromptRegistry,
    default_registry,
    register_prompt,
    get_prompt,
    list_prompts,
)


# ===== 向后兼容的私有函数别名 =====
# 老版本中的 `_get_jinja2_env()` / `_check_jinja2()` 若被其他模块引用，
# 这里提供别名以避免重构后出现 ImportError。
def _get_jinja2_env():
    """
    向后兼容的内部函数别名。

    新代码请直接使用：
        from gecko.core.prompt import get_jinja2_env
    """
    return get_jinja2_env()


def _check_jinja2():
    """
    向后兼容的内部函数别名。

    新代码请直接使用：
        from gecko.core.prompt import check_jinja2
    """
    return check_jinja2()


# ===== 修正 __module__ 提升兼容性 =====
# 这样在 repr / 日志 / 反射 / pickle 时，仍然表现为 "gecko.core.prompt.*"
PromptTemplate.__module__ = "gecko.core.prompt"
PromptLibrary.__module__ = "gecko.core.prompt"

# 视情况也可以把新类的 __module__ 统一成 gecko.core.prompt，方便调试/日志
PromptSection.__module__ = "gecko.core.prompt"
PromptComposer.__module__ = "gecko.core.prompt"
PromptValidator.__module__ = "gecko.core.prompt"
PromptRegistry.__module__ = "gecko.core.prompt"


__all__ = [
    # 核心模板
    "PromptTemplate",
    # 预定义模板库
    "PromptLibrary",
    "DEFAULT_REACT_PROMPT",
    # Jinja2 环境
    "get_jinja2_env",
    "check_jinja2",
    "_get_jinja2_env",
    "_check_jinja2",
    # 组合器
    "PromptSection",
    "PromptComposer",
    # 验证 / Lint
    "IssueSeverity",
    "PromptIssue",
    "PromptValidator",
    "lint_prompt",
    # 注册中心 / 版本管理
    "PromptRecord",
    "PromptRegistry",
    "default_registry",
    "register_prompt",
    "get_prompt",
    "list_prompts",
]
