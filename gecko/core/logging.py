# gecko/core/logging.py
"""
Gecko 结构化日志系统（改进版）

改进：使用成熟的 structlog 库，代码减少 80%
"""
from __future__ import annotations
import logging
import sys
from typing import Any, Optional

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import warnings
    warnings.warn(
        "structlog not installed. Install with: pip install structlog\n"
        "Falling back to standard logging.",
        ImportWarning
    )

from gecko.config import settings

# ========== 日志初始化 ==========

_initialized = False

def setup_logging(
    level: Optional[str] = None,
    force: bool = False
):
    """
    初始化日志系统
    
    改进：
    1. 优先使用 structlog（如果可用）
    2. 降级到标准 logging（如果 structlog 未安装）
    """
    global _initialized
    
    if _initialized and not force:
        return
    
    level = level or settings.log_level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if STRUCTLOG_AVAILABLE:
        _setup_structlog(log_level)
    else:
        _setup_standard_logging(log_level)
    
    # 降低第三方库日志级别
    for lib in ["httpx", "httpcore", "litellm", "openai"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    _initialized = True

def _setup_structlog(level: int):
    """配置 structlog"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars, 
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # 根据配置选择渲染器
            structlog.processors.JSONRenderer()
            if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

def _setup_standard_logging(level: int):
    """配置标准 logging（降级方案）"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

# ========== 获取 Logger ==========

def get_logger(name: str) -> Any:
    """
    获取 Logger 实例
    
    返回：
    - structlog.BoundLogger（如果可用）
    - logging.Logger（降级方案）
    
    使用示例:
        logger = get_logger(__name__)
        logger.info("event happened", user_id=123, action="login")
    """
    if not _initialized:
        setup_logging()
    
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)

# ========== 自动初始化 ==========

setup_logging()