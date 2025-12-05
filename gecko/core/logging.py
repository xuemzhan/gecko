# gecko/core/logging.py
"""
Gecko 结构化日志系统（改进版）

改进要点：
1. 优先使用 structlog（如未安装则自动降级到标准 logging）
2. 支持 trace_id / span_id / 额外上下文注入，便于分布式追踪与问题排查
3. 使用 ContextVar 保存追踪上下文，兼容多线程 / 异步场景
4. 修复：
   - ContextVar(default_factory=...) 的错误用法
   - structlog 不可用时 ContextLogger 与标准 logging 的兼容问题
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections import ChainMap
import logging
import sys
import uuid
import warnings
from typing import Any, Dict, Optional

# ========= 可选依赖：structlog =========

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    # 如果未安装 structlog，则降级为标准 logging，同时给出一次性警告
    STRUCTLOG_AVAILABLE = False
    warnings.warn(
        "structlog 未安装，将回退到标准 logging。\n"
        "如需结构化日志，请安装：pip install structlog",
        ImportWarning,
    )

from gecko.config import settings

# ========= 追踪上下文变量 =========
#
# 说明：
#   - 使用 ContextVar 而不是全局变量，可以在异步/多线程环境下保持各自独立的上下文
#   - trace_id / span_id 用于分布式追踪
#   - extra_context 用于附加业务相关的上下文字段（如 user_id, request_id 等）

# 追踪 ID（链路级别）
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# Span ID（单次调用级别）
span_id_var: ContextVar[str] = ContextVar("span_id", default="")

# 额外上下文字段
# 注意：default 使用 {} 虽然是可变对象，但：
#   - ContextVar 的 default 只在「从未 set」时作为初始返回值
#   - 我们在使用时会立即 .copy() 并 set 回去，不会修改这个默认对象本身
extra_context_var: ContextVar[Dict[str, Any]] = ContextVar(
    "extra_context",
    default={},  # 默认空字典；实际使用时会 copy
)


def generate_trace_id() -> str:
    """
    生成追踪 ID（16 字节十六进制字符串）

    建议：在一次完整请求 / 会话的生命周期内复用同一个 trace_id，
    便于跨服务聚合日志。
    """
    return uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    """
    生成 Span ID（8 字节十六进制字符串）

    通常每一个子操作 / 步骤使用一个新的 span_id，
    和 trace_id 一起用于构建调用树。
    """
    return uuid.uuid4().hex[:8]


# ========= 日志初始化 =========

_initialized = False  # 模块内全局标记，避免重复初始化


def setup_logging(
    level: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    初始化日志系统。

    参数：
        level:
            日志级别字符串，如 "DEBUG" / "INFO" / "WARNING"。
            默认为 settings.log_level。
        force:
            为 True 时强制重新初始化（但目前不会清理旧 handler，仅重新配置）。

    设计说明：
        - 优先使用 structlog 进行结构化日志输出
        - 如未安装 structlog，则降级为标准 logging.basicConfig
        - 并统一降低部分高噪声三方库的日志级别到 WARNING
    """
    global _initialized

    # 如果已经初始化过，且未要求强制重置，则直接返回
    if _initialized and not force:
        return

    # ========= 屏蔽特定三方库的烦人警告 =========
    # LiteLLM / Pydantic 的序列化警告
    warnings.filterwarnings(
        "ignore",
        message=".*Pydantic serializer warnings.*",
        category=UserWarning,
    )
    # DuckDuckGo backend='api' 废弃警告
    warnings.filterwarnings(
        "ignore",
        message=".*backend='api' is deprecated.*",
        category=UserWarning,
    )

    # 解析日志级别：优先使用参数，其次 settings.log_level，兜底为 INFO
    level = level or settings.log_level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # structlog 优先；否则回退到标准 logging
    if STRUCTLOG_AVAILABLE:
        _setup_structlog(log_level)
    else:
        _setup_standard_logging(log_level)

    # 降低易产生噪声的三方库日志级别
    for lib in ["httpx", "httpcore", "litellm", "openai"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    _initialized = True


def _setup_structlog(level: int) -> None:
    """
    配置 structlog 日志系统。

    说明：
        - 使用 PrintLoggerFactory 直接打印到 stdout
        - 使用 JSONRenderer 或 ConsoleRenderer 输出结构化日志
        - 不再使用 structlog.contextvars.merge_contextvars，
          上下文注入统一通过 ContextLogger 完成，避免混淆
    """
    assert STRUCTLOG_AVAILABLE  # 仅在 structlog 可用时调用

    structlog.configure(  # type: ignore
        processors=[
            # 注入日志级别字段
            structlog.processors.add_log_level,  # type: ignore
            # 渲染 stack_info（如设置了 stack_info=True）
            structlog.processors.StackInfoRenderer(),  # type: ignore
            # 将 exc_info 信息转换为可序列化结构（结合 logger.exception 使用）
            structlog.dev.set_exc_info,  # type: ignore
            # 增加时间戳字段，ISO 格式，UTC 时区
            structlog.processors.TimeStamper(fmt="iso", utc=True),  # type: ignore
            # 最终渲染器：JSON 或控制台友好格式
            (
                structlog.processors.JSONRenderer()  # type: ignore
                if settings.log_format == "json"  # type: ignore
                else structlog.dev.ConsoleRenderer()  # type: ignore
            ),
        ],
        # 根据日志级别过滤日志
        wrapper_class=structlog.make_filtering_bound_logger(level),  # type: ignore
        # 上下文字段的容器类型
        context_class=dict,
        # 输出到 stdout
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),  # type: ignore
        # 首次使用时缓存 logger，提高性能
        cache_logger_on_first_use=True,
    )


def _setup_standard_logging(level: int) -> None:
    """
    配置标准 logging（降级方案）。

    说明：
        - 使用 logging.basicConfig 输出到 stdout
        - 日志格式较为传统，但配合 ContextLogger 仍可注入 extra 字段
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ========= 获取基础 Logger =========

def get_logger(name: str) -> Any:
    """
    获取基础 Logger 实例（不带自动上下文注入）。

    返回：
        - structlog.BoundLogger（如果 structlog 可用）
        - logging.Logger（如果降级到标准 logging）

    使用示例：
        logger = get_logger(__name__)
        logger.info("event happened", user_id=123, action="login")

    注意：
        - 如果希望自动注入 trace_id / span_id / 额外上下文，
          推荐使用 get_context_logger 而不是直接使用 get_logger。
    """
    if not _initialized:
        setup_logging()

    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)  # type: ignore
    else:
        return logging.getLogger(name)


# ========= 追踪上下文管理器 =========

@contextmanager
def trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra: Any,
):
    """
    追踪上下文管理器。

    功能：
        - 自动生成或复用 trace_id
        - 自动生成 span_id
        - 注入额外上下文字段（extra），如 user_id, action 等
        - 与 ContextLogger 配合使用，所有日志自动带上上述字段

    示例：
        logger = get_context_logger(__name__)

        with trace_context(user_id="123", action="chat"):
            logger.info("Processing request")
            # 输出中会包含：trace_id, span_id, user_id, action

    嵌套行为：
        - 内层 trace_context 会基于外层 trace_id 或重新生成（如果显式传入）
        - 使用 ContextVar 的 token 机制，在退出时恢复外层上下文
    """
    # 1. 计算本次上下文的 trace_id 与 span_id
    #    - 优先使用传入参数
    #    - 其次复用当前上下文中的 trace_id
    #    - 最后才生成新的 trace_id
    tid = trace_id or trace_id_var.get() or generate_trace_id()
    sid = span_id or generate_span_id()

    # 2. 设置追踪 ID 到 ContextVar，并记录 token 以便退出时恢复
    trace_token = trace_id_var.set(tid)
    span_token = span_id_var.set(sid)

    # 3. 合并额外上下文（从当前 extra_context + 新传入的 extra）
    current_extra = extra_context_var.get().copy()  # copy 避免修改共享对象
    current_extra.update(extra)
    extra_token = extra_context_var.set(current_extra)

    try:
        # 在 with 体内可以拿到当前 trace/span，便于手工操作
        yield {"trace_id": tid, "span_id": sid}
    finally:
        # 4. 使用 token 恢复到进入 with 之前的上下文状态
        trace_id_var.reset(trace_token)
        span_id_var.reset(span_token)
        extra_context_var.reset(extra_token)


# ========= 带上下文注入的 Logger 包装器 =========

class ContextLogger:
    """
    带上下文的 Logger 包装器。

    功能：
        - 自动从 ContextVar 中取出 trace_id / span_id / extra_context
        - 对 structlog Logger：将上下文作为字段展开
        - 对标准 logging.Logger：通过 extra 参数注入上下文字段

    设计目标：
        - 对调用方提供统一的 API（debug/info/warning/error/exception）
        - 清晰区分「上下文字段」与「控制参数」（exc_info、stack_info 等）
    """

    def __init__(self, logger: Any):
        self._logger = logger
        # 简单的 duck-typing：
        # structlog 的 logger 一般有 bind 方法，标准 logging.Logger 没有
        self._is_structlog = hasattr(logger, "bind")

    # ---- 内部工具方法 ----

    def _enrich(self, **kwargs: Any) -> Dict[str, Any]:
        """
        构造完整的上下文字段字典。

        改进：使用 ChainMap 避免深 copy，降低 GC 压力。
        但对 structlog 和标准 logging，最终仍需转换为普通 dict。

        合并顺序：
            1. trace_id / span_id
            2. extra_context_var 中保存的额外上下文
            3. 调用方传入的业务字段（kwargs）
        """
        # 使用 ChainMap 避免深 copy（读操作时不分配新内存）
        maps_list: list[Dict[str, Any]] = []

        # 1. 追踪信息（创建新 dict 避免修改原 ContextVar）
        trace_info: Dict[str, Any] = {}
        trace_id = trace_id_var.get()
        if trace_id:
            trace_info["trace_id"] = trace_id

        span_id = span_id_var.get()
        if span_id:
            trace_info["span_id"] = span_id

        if trace_info:
            maps_list.append(trace_info)

        # 2. 额外上下文
        extra_ctx = extra_context_var.get()
        if extra_ctx:
            maps_list.append(extra_ctx)

        # 3. 调用方字段
        if kwargs:
            maps_list.append(kwargs)

        # 使用 ChainMap 组合所有层级（避免 copy），最后转换为 dict
        if not maps_list:
            return {}
        elif len(maps_list) == 1:
            return maps_list[0].copy() if maps_list[0] else {}
        else:
            # 多层时使用 ChainMap，最后再转为 dict
            chain = ChainMap(*maps_list)
            return dict(chain)

    def _log(self, method_name: str, message: str, **kwargs: Any) -> None:
        """
        通用日志调用入口。

        参数拆分：
            - 控制类参数（exc_info / stack_info / stacklevel）
            - 业务字段（将被合并到上下文字段中）

        对 structlog：
            logger.<level>(event=message, **enriched_fields, **control_kwargs)

        对标准 logging：
            logger.<level>(message, extra=enriched_fields, **control_kwargs)
        """
        # 从 kwargs 中拆分出控制参数（logging/structlog 认可的控制字段）
        control_keys = ("exc_info", "stack_info", "stacklevel")
        control_kwargs: Dict[str, Any] = {}
        business_kwargs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if k in control_keys:
                control_kwargs[k] = v
            else:
                business_kwargs[k] = v

        # 构造上下文字段
        enriched = self._enrich(**business_kwargs)
        log_method = getattr(self._logger, method_name)

        if self._is_structlog:
            # structlog：上下文字段直接展开成事件字段
            log_method(message, **enriched, **control_kwargs)
        else:
            # 标准 logging：
            #   - 上下文字段通过 extra 注入到 LogRecord.__dict__
            #   - 控制参数（exc_info 等）以正常方式传入
            log_method(message, extra=enriched, **control_kwargs)

    # ---- 对外日志方法 ----

    def debug(self, message: str, **kwargs: Any) -> None:
        """DEBUG 级别日志。"""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """INFO 级别日志。"""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """WARNING 级别日志。"""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """ERROR 级别日志。"""
        self._log("error", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """
        记录异常日志，自动附带当前异常堆栈。

        等价于：
            logger.error(..., exc_info=True)
        """
        # 强制携带 exc_info=True，兼容 structlog 和标准 logging 的习惯用法
        kwargs.setdefault("exc_info", True)
        self._log("exception", message, **kwargs)


def get_context_logger(name: str) -> ContextLogger:
    """
    获取带上下文自动注入能力的 Logger。

    使用示例：
        logger = get_context_logger(__name__)

        with trace_context(user_id="123"):
            logger.info("User action", action="click")

        # 结构化日志中会自动包含：
        #   - trace_id
        #   - span_id
        #   - user_id
        #   - action
    """
    base_logger = get_logger(name)
    return ContextLogger(base_logger)


# ========= 模块导入后自动初始化 =========
#
# 说明：
#   - 作为应用主工程使用时，自动初始化通常是方便的
#   - 如将 gecko 作为纯库使用，且需要自定义 logging 配置，
#     可以考虑未来通过配置开关控制是否自动初始化

setup_logging()
