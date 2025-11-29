# gecko/core/prompt/jinja_env.py
"""
Jinja2 环境封装模块

职责：
1. 懒加载检查当前环境是否已安装 jinja2；
2. 提供全局单例的 Jinja2 Environment 对象；
3. 作为 gecko.core.prompt 模块与 Jinja2 的解耦层，后续如果需要
   切换为 SandboxedEnvironment（安全沙箱）或增加过滤器，在本文件集中处理。

设计要点：
- 避免在 import 阶段就强依赖 jinja2，使用时再检测；
- 通过 module 内部的全局变量缓存 Environment，避免重复创建。
"""

from __future__ import annotations

from typing import Any, Optional

from gecko.core.logging import get_logger

logger = get_logger(__name__)

# 标记 jinja2 是否可用的缓存变量（None 表示尚未检测）
_jinja2_available: Optional[bool] = None
# 全局单例 Environment 实例缓存
_jinja2_env: Any = None


def check_jinja2() -> bool:
    """
    检查当前环境中是否可用 jinja2（懒加载）。

    返回:
        True  - 已安装 jinja2，后续可以正常使用 Jinja2 模板；
        False - 未安装 jinja2，使用相关功能会抛 ImportError。

    说明：
        - 只在第一次调用时真正 import jinja2，之后结果会缓存在 _jinja2_available 中；
        - 这样可以避免在不使用 Prompt 功能时引入多余依赖。
    """
    global _jinja2_available
    if _jinja2_available is None:
        try:
            import jinja2  # noqa: F401  # 仅用于验证模块存在
            _jinja2_available = True
        except ImportError:
            _jinja2_available = False
    return bool(_jinja2_available)


def get_jinja2_env():
    """
    获取全局 Jinja2 Environment 单例。

    行为：
        1. 若尚未创建 Environment，则调用 check_jinja2() 检查依赖是否存在；
        2. 若未安装 jinja2，抛出 ImportError，提示用户安装依赖；
        3. 若已安装，则创建 Environment 并缓存到模块级变量；
        4. 后续调用直接复用缓存对象。

    当前 Environment 配置：
        - undefined=StrictUndefined
          未定义变量会抛异常，避免静默错误；
        - autoescape=False
          Prompt 通常为纯文本，不需要 HTML 自动转义；
        - keep_trailing_newline=True
          保留模板末尾换行，避免拼接时格式错乱；
        - extensions=[]
          暂不启用额外扩展。

    可扩展点：
        - 如果未来需要严格的沙箱执行，可将 Environment 替换为 SandboxedEnvironment；
        - 如需增加自定义 filter / test，可在这里统一注册。
    """
    global _jinja2_env

    if _jinja2_env is None:
        if not check_jinja2():
            raise ImportError(
                "PromptTemplate 依赖 jinja2。\n"
                "请安装：pip install jinja2\n"
                "或：rye add jinja2"
            )

        from jinja2 import Environment, StrictUndefined

        _jinja2_env = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
            extensions=[],
        )
        logger.debug("Jinja2 environment initialized")

    return _jinja2_env


__all__ = ["check_jinja2", "get_jinja2_env"]
