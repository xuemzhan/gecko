# agno/utils/log.py

"""
日志记录模块

该模块为 agno 库提供了集中式、可配置的日志记录功能。
主要特性包括：
- 使用 rich 库实现美观、彩色的控制台输出。
- 为不同组件（Agent, Team, Workflow）提供不同颜色的日志，便于区分。
- 提供全局的 debug 模式开关，方便在运行时动态调整日志级别。
- 提供简单的函数式 API（`log_debug`, `log_info` 等）供整个库使用。
- 允许用户通过 `configure_agno_logging` 注入自定义的 logger 实例。
"""

import logging
from functools import lru_cache
from os import getenv
from typing import Any, Literal, Optional, Protocol, Union
from rich.logging import RichHandler
from rich.text import Text
import shutil


# --- 常量与类型定义 ---

LOGGER_NAME = "agno"
TEAM_LOGGER_NAME = f"{LOGGER_NAME}-team"
WORKFLOW_LOGGER_NAME = f"{LOGGER_NAME}-workflow"


class LoggerProtocol(Protocol):
    """定义 logger 必须支持的方法（用于自定义 logger 适配）"""
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


# 完整日志级别颜色映射（支持 debug/info/warning/error）
LOG_STYLES = {
    "agent": {
        "debug": "green",
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "exception": "red",
    },
    "team": {
        "debug": "magenta",
        "info": "blue",
        "warning": "orange1",
        "error": "red",
        "exception": "red",
    },
    "workflow": {
        "debug": "yellow",
        "info": "orange3",
        "warning": "red",
        "error": "red",
        "exception": "red",
    },
}


# --- 工具函数 ---

@lru_cache(maxsize=None)
def _get_terminal_width() -> int:
    """缓存终端宽度，避免重复系统调用"""
    try:
        return max(60, shutil.get_terminal_size().columns - 20)
    except (ImportError, OSError):
        return 80


def center_header(message: str, symbol: str = "*") -> str:
    """在终端宽度内居中显示一条消息，并用符号填充两侧。"""
    width = _get_terminal_width()
    return f" {message} ".center(width, symbol)


# --- 自定义日志处理器 ---

class ColoredRichHandler(RichHandler):
    """支持按 source_type 着色的日志处理器"""
    def __init__(self, source_type: str = "agent", **kwargs: Any):
        super().__init__(**kwargs)
        self.source_type = source_type

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level = record.levelname.lower()
        style_map = LOG_STYLES.get(self.source_type, LOG_STYLES["agent"])
        color = style_map.get(level, "default")
        return Text(record.levelname, style=color)


# --- 安全的 logger 构建器 ---

def _create_agno_logger(name: str, source_type: str) -> logging.Logger:
    """创建并配置一个独立的 AgnoLogger（不修改全局 Logger 类）"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # 直接创建 AgnoLogger 实例，避免 setLoggerClass 竞争
    logger.__class__ = type("AgnoLogger", (logging.Logger,), {
        "debug": _make_log_method(logging.DEBUG, "debug"),
        "info": _make_log_method(logging.INFO, "info"),
    })

    handler = ColoredRichHandler(
        show_time=False,
        rich_tracebacks=True,
        show_path=getenv("AGNO_DEBUG_SHOW_PATH", "False").lower() == "true",
        source_type=source_type,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _make_log_method(level: int, method_name: str):
    """动态生成支持 center/symbol 的日志方法"""
    def log_method(self, msg: str, center: bool = False, symbol: str = "*", *args, **kwargs):
        if self.isEnabledFor(level):
            if center:
                msg = center_header(str(msg), symbol)
            getattr(super(type(self), self), method_name)(msg, *args, **kwargs)
    return log_method


# --- 全局 logger 实例（延迟初始化）---

@lru_cache(maxsize=1)
def _get_agent_logger() -> logging.Logger:
    return _create_agno_logger(LOGGER_NAME, "agent")

@lru_cache(maxsize=1)
def _get_team_logger() -> logging.Logger:
    return _create_agno_logger(TEAM_LOGGER_NAME, "team")

@lru_cache(maxsize=1)
def _get_workflow_logger() -> logging.Logger:
    return _create_agno_logger(WORKFLOW_LOGGER_NAME, "workflow")


# 当前激活的 logger（默认为 agent）
_current_logger: LoggerProtocol = _get_agent_logger()

# 全局 debug 状态
_debug_on: bool = False
_debug_level: Literal[1, 2] = 1


# --- 日志级别控制 ---

def _set_all_loggers_level(level: int) -> None:
    """统一设置所有内置 logger 的级别"""
    _get_agent_logger().setLevel(level)
    _get_team_logger().setLevel(level)
    _get_workflow_logger().setLevel(level)


def set_log_level_to_debug(level: Literal[1, 2] = 1) -> None:
    global _debug_on, _debug_level
    _debug_on = True
    _debug_level = level
    _set_all_loggers_level(logging.DEBUG)


def set_log_level_to_info() -> None:
    global _debug_on
    _debug_on = False
    _set_all_loggers_level(logging.INFO)


def set_log_level_to_warning() -> None:
    global _debug_on
    _debug_on = False
    _set_all_loggers_level(logging.WARNING)


# --- Logger 上下文切换 ---

def use_agent_logger() -> None:
    global _current_logger
    _current_logger = _get_agent_logger()


def use_team_logger() -> None:
    global _current_logger
    _current_logger = _get_team_logger()


def use_workflow_logger() -> None:
    global _current_logger
    _current_logger = _get_workflow_logger()


# --- 公共日志 API ---

def log_debug(msg: Any, center: bool = False, symbol: str = "*", level: Literal[1, 2] = 1) -> None:
    if _debug_on and _debug_level >= level:
        if hasattr(_current_logger, 'debug'):
            _current_logger.debug(str(msg), center=center, symbol=symbol)
        else:
            _current_logger.debug(str(msg))


def log_info(msg: Any, center: bool = False, symbol: str = "*") -> None:
    if hasattr(_current_logger, 'info'):
        _current_logger.info(str(msg), center=center, symbol=symbol)
    else:
        _current_logger.info(str(msg))


def log_warning(msg: Any) -> None:
    _current_logger.warning(str(msg))


def log_error(msg: Any) -> None:
    _current_logger.error(str(msg))


def log_exception(msg: Any) -> None:
    _current_logger.exception(str(msg))


# --- 用户自定义配置 ---

def configure_agno_logging(
    custom_agent_logger: Optional[LoggerProtocol] = None,
    custom_team_logger: Optional[LoggerProtocol] = None,
    custom_workflow_logger: Optional[LoggerProtocol] = None,
) -> None:
    """
    注入自定义 logger。如果传入，则替换对应组件的 logger。
    为保证 `center` 等参数兼容，非 AgnoLogger 会被包装。
    """
    global _current_logger

    # 简单包装器，忽略 center/symbol（避免报错）
    class _LoggerAdapter:
        def __init__(self, logger: LoggerProtocol):
            self._logger = logger

        def debug(self, msg: str, **kwargs): self._logger.debug(msg)
        def info(self, msg: str, **kwargs): self._logger.info(msg)
        def warning(self, msg: str, **kwargs): self._logger.warning(msg)
        def error(self, msg: str, **kwargs): self._logger.error(msg)
        def exception(self, msg: str, **kwargs): self._logger.exception(msg)

    if custom_agent_logger:
        _current_logger = _LoggerAdapter(custom_agent_logger)
    if custom_team_logger:
        # 团队 logger 替换仅影响 use_team_logger()，此处暂不实现全局切换
        pass
    if custom_workflow_logger:
        pass
    # 注意：完整支持多 logger 注入需重构 _current_logger 机制，此处保持简单


# --- 兼容性别名（供外部使用）---
logger = _current_logger


# --- 测试代码 ---
if __name__ == "__main__":
    import sys

    # 重定向 rich 输出到 stdout（确保测试可见）
    logging.basicConfig(stream=sys.stdout, force=True)

    print("--- 正在运行 agno/utils/log.py 的测试代码 ---")

    # 1. 默认 INFO 级别
    print("\n[1] 默认 INFO 级别:")
    log_info("普通信息日志")
    log_debug("不应显示的 debug")
    log_info("居中标题", center=True, symbol="-")

    # 2. DEBUG 模式
    print("\n[2] DEBUG 模式:")
    set_log_level_to_debug(1)
    log_debug("level=1 debug 可见")
    log_debug("level=2 debug 不可见", level=2)
    set_log_level_to_debug(2)
    log_debug("level=2 debug 可见", level=2, center=True)
    set_log_level_to_info()

    # 3. 切换上下文
    print("\n[3] 切换 Logger 上下文（颜色应变化）:")
    set_log_level_to_debug(1)
    use_agent_logger()
    log_info("Agent: cyan")
    log_debug("Agent debug: green")
    use_team_logger()
    log_info("Team: blue")
    log_debug("Team debug: magenta")
    use_workflow_logger()
    log_info("Workflow: orange")
    log_debug("Workflow debug: yellow")
    use_agent_logger()
    set_log_level_to_info()

    # 4. 自定义 logger
    print("\n[4] 自定义 Logger:")
    class SimpleLogger:
        def info(self, msg): print(f"[CUSTOM] {msg}")
        def debug(self, msg): print(f"[CUSTOM-DBG] {msg}")
        def warning(self, msg): print(f"[CUSTOM-WARN] {msg}")
        def error(self, msg): print(f"[CUSTOM-ERR] {msg}")
        def exception(self, msg): print(f"[CUSTOM-EXC] {msg}")

    configure_agno_logging(custom_agent_logger=SimpleLogger())
    log_info("通过自定义 logger 输出")
    log_info("居中请求被忽略", center=True)

    print("\n--- 测试结束 ---")