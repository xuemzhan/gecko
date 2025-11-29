# gecko/core/output/factories.py
"""
AgentOutput / JsonOutput 工厂函数模块

提供若干便捷方法，用于快速构造常见形式的输出：
- create_text_output : 仅包含文本内容的 AgentOutput
- create_tool_output : 包含工具调用的 AgentOutput
- create_json_output : 承载结构化 JSON 数据的 JsonOutput
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gecko.core.output.agent_output import AgentOutput
from gecko.core.output.token_usage import TokenUsage
from gecko.core.output.json_output import JsonOutput   # 新增导入


def create_text_output(
    content: str,
    usage: Optional[TokenUsage] = None,
    **metadata: Any,
) -> AgentOutput:
    """
    快速创建纯文本输出（AgentOutput）。

    参数:
        content: 文本内容
        usage: Token 使用统计（可选）
        **metadata: 附加元数据，会存入 AgentOutput.metadata
    """
    return AgentOutput(
        content=content,
        usage=usage,
        metadata=metadata,
    )


def create_tool_output(
    tool_calls: List[Dict[str, Any]],
    content: str = "",
    usage: Optional[TokenUsage] = None,
    **metadata: Any,
) -> AgentOutput:
    """
    快速创建「包含工具调用」的输出（AgentOutput）。

    参数:
        tool_calls: 工具调用列表（OpenAI 工具调用格式）
        content: 可选的文本内容（一般为「我去调用某工具」之类的说明）
        usage: Token 使用统计（可选）
        **metadata: 附加元数据
    """
    return AgentOutput(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        metadata=metadata,
    )


def create_json_output(
    data: Any,
    usage: Optional[TokenUsage] = None,
    **metadata: Any,
) -> JsonOutput:
    """
    快速创建结构化 JSON 输出（JsonOutput）。

    参数:
        data: 结构化数据，一般为 dict/list/标量等
        usage: Token 使用统计（可选）
        **metadata: 附加元数据，会存入 JsonOutput.metadata

    返回:
        JsonOutput 实例

    典型场景：
        - 工具调用返回了结构化结果，想在框架内部完整保留
        - 中间推理结果是一个复杂 dict，而不是单纯文本
    """
    return JsonOutput(
        data=data,
        usage=usage,
        metadata=metadata,
    )
