# gecko/core/tracing.py
"""
请求追踪模块（基于 logging 统一实现的版本）

本模块是对 gecko.core.logging 中「追踪上下文能力」的一个
**语义化薄封装**，主要解决两个问题：

1. 对上层调用者提供更语义化的 API：
   - generate_trace_id / generate_span_id
   - get_trace_id / get_span_id / get_context
   - trace_context / set_trace_context / clear_trace_context

2. 所有追踪状态（trace_id / span_id / extra_context）只维护一份：
   - 统一复用 gecko.core.logging 中已经定义好的 ContextVar
   - 不再在 tracing.py 内单独维护一套重复的 ContextVar
   - 避免「两份状态 + 同步」带来的复杂度和潜在不一致问题

设计要点：
- tracing.py 不再保存独立的状态，只是对 logging 中的能力进行包装
- 任何通过本模块设置的上下文，日志中都会自动体现（因为使用同一组 ContextVar）
- 任何通过 gecko.core.logging.trace_context 设置的上下文，
  也可以通过本模块的 get_trace_id / get_context 等函数访问到
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

# 直接复用 logging 模块中的上下文变量与 ID 生成函数
from gecko.core.logging import (
    trace_id_var,        # ContextVar[str]
    span_id_var,         # ContextVar[str]
    extra_context_var,   # ContextVar[Dict[str, Any]]
    generate_trace_id as _generate_trace_id,
    generate_span_id as _generate_span_id,
    trace_context as _logging_trace_context,
)

# =========================================================
# 对外暴露的 ID 生成函数
# =========================================================

def generate_trace_id() -> str:
    """
    生成新的 trace_id。

    说明：
        - 直接复用 gecko.core.logging 中的实现
        - 返回值通常为 16 位十六进制字符串（uuid4 的截断）
        - 可用于与外部系统（如 APM、链路追踪系统）对接
    """
    return _generate_trace_id()


def generate_span_id() -> str:
    """
    生成新的 span_id。

    说明：
        - 直接复用 gecko.core.logging 中的实现
        - 返回值通常为 8 位十六进制字符串
        - 可用于标识单个子调用 / 子操作
    """
    return _generate_span_id()


# =========================================================
# 读取当前追踪上下文状态
# =========================================================

def get_trace_id() -> str:
    """
    获取当前上下文中的 trace_id。

    说明：
        - 如果当前没有设置 trace_id，则返回默认空字符串 ""
        - trace_id 的值来源：
            * 通过本模块的 trace_context / set_trace_context 设置
            * 或通过 gecko.core.logging.trace_context 直接设置
    """
    return trace_id_var.get()


def get_span_id() -> str:
    """
    获取当前上下文中的 span_id。

    说明：
        - 如果当前没有设置 span_id，则返回默认空字符串 ""
        - span_id 的值来源同 get_trace_id
    """
    return span_id_var.get()


def get_context() -> Dict[str, Any]:
    """
    获取当前追踪上下文的完整字典表示。

    返回字段（存在则返回，不存在则省略）：
        - "trace_id": 当前 trace_id
        - "span_id": 当前 span_id
        - 额外上下文字段（extra_context_var 中的内容），如：
            * "user_id"
            * "request_id"
            * "action"
            * ...

    示例：
        >>> get_context()
        {
            "trace_id": "abcd1234ef567890",
            "span_id": "1234abcd",
            "user_id": "u-001",
            "action": "login"
        }
    """
    ctx: Dict[str, Any] = {}

    trace_id = trace_id_var.get()
    if trace_id:
        ctx["trace_id"] = trace_id

    span_id = span_id_var.get()
    if span_id:
        ctx["span_id"] = span_id

    extra = extra_context_var.get()
    if extra:
        ctx.update(extra)

    return ctx


# =========================================================
# 追踪上下文管理器（语义化封装）
# =========================================================

@contextmanager
def trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra: Any,
) -> Iterator[Dict[str, Any]]:
    """
    追踪上下文管理器（统一封装，内部直接复用 logging.trace_context）。

    功能：
        - 自动生成或复用 trace_id / span_id
        - 设置额外的上下文字段（extra），如 user_id、request_id、action 等
        - 所有上下文状态都保存在 gecko.core.logging 中的 ContextVar 里
          -> 日志输出时会自动带上这些字段（使用 ContextLogger）

    参数：
        trace_id:
            可选。传入已有的 trace_id，用于链路复用；
            若不传，则优先使用当前上下文中的 trace_id，
            若仍为空，则调用 generate_trace_id() 生成新的。
        span_id:
            可选。传入已有的 span_id；
            若不传，则调用 generate_span_id() 生成新的。
        **extra:
            任意键值对，作为额外上下文字段注入，如：
                user_id="123",
                action="chat",
                request_id="req-001"

    返回（yield 的值）：
        一个 dict，包含：
            - "trace_id"
            - "span_id"
            - 以及传入的 extra 字段

    示例：
        >>> from gecko.core.tracing import trace_context, get_context
        >>> from gecko.core.logging import get_context_logger
        >>> logger = get_context_logger(__name__)

        with trace_context(user_id="123", action="chat") as ctx:
            # ctx = {"trace_id": "...", "span_id": "...", "user_id": "123", "action": "chat"}
            logger.info("Handling request")
            # 日志中会自动包含 trace_id / span_id / user_id / action

        # 上下文退出后，trace_id / span_id / extra 会自动恢复到进入 with 前的状态
    """
    # 这里不再重复实现 token/reset 逻辑，而是直接复用 logging.trace_context，
    # 确保两处行为完全一致，避免维护两份类似代码。
    #
    # 注意：_logging_trace_context 本身也是一个 @contextmanager，
    #       所以这里用嵌套 with 的方式做一层薄包装。
    with _logging_trace_context(trace_id=trace_id, span_id=span_id, **extra) as ctx:
        yield ctx


# =========================================================
# 非上下文管理器方式：直接设置 / 清除上下文
# =========================================================

def set_trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    直接设置当前追踪上下文（非 with 方式）。

    典型使用场景：
        - 框架型代码在请求进入入口处统一设置上下文：
            * 从 HTTP 头、gRPC metadata 中解析出 trace_id
            * 根据请求信息设置 user_id / request_id 等
        - 后面业务代码不需要关心设置，只需通过日志或 get_context 使用即可

    参数：
        trace_id:
            可选。不为空时，直接覆盖当前 trace_id。
        span_id:
            可选。不为空时，直接覆盖当前 span_id。
        **extra:
            可选。传入的键值对会与当前 extra_context 合并（后者覆盖前者）。
    """
    if trace_id:
        trace_id_var.set(trace_id)

    if span_id:
        span_id_var.set(span_id)

    if extra:
        # 取出当前的 extra_context，copy 后合并新字段再 set 回去，
        # 避免对原字典的就地修改带来意外共享。
        current = dict(extra_context_var.get())
        current.update(extra)
        extra_context_var.set(current)


def clear_trace_context() -> None:
    """
    清除当前追踪上下文。

    行为：
        - 将 trace_id 重置为空字符串 ""
        - 将 span_id 重置为空字符串 ""
        - 将 extra_context 重置为一个新的空字典 {}

    注意：
        - 如果你在 with trace_context(...) 内调用 clear_trace_context()，
          上下文退出时（__exit__ 阶段）logging.trace_context 会使用 token.reset()
          将 ContextVar 恢复到进入 with 前的值，因此：
              * clear 的效果只在当前「层」中有效
              * 一旦退出 with，外层上下文会被恢复

        - 一般而言，clear_trace_context 更适合作为「最外层」的清理动作，
          例如在请求处理完毕、线程结束、协程回收时调用。
    """
    trace_id_var.set("")
    span_id_var.set("")
    extra_context_var.set({})


__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "get_trace_id",
    "get_span_id",
    "get_context",
    "trace_context",
    "set_trace_context",
    "clear_trace_context",
]
