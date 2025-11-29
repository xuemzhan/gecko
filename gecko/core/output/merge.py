# gecko/core/output/merge.py
"""
AgentOutput 聚合/合并工具模块

当前提供：
- merge_outputs : 将多个 AgentOutput 合并为一个（多 Agent 场景、流水线场景）

后续如果有更多聚合策略（例如按角色合并、按对话轮次分组），
也可以在本模块中扩展。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gecko.core.output.agent_output import AgentOutput
from gecko.core.output.token_usage import TokenUsage


def merge_outputs(outputs: List[AgentOutput]) -> AgentOutput:
    """
    合并多个 AgentOutput（用于多 Agent 或多轮中间结果汇总）。

    合并策略（保持原有语义）：
        - 内容（content）：将所有「有非空内容」的输出按顺序用换行符拼接。
        - 工具调用（tool_calls）：简单拼接列表。
        - Usage：对 prompt_tokens / completion_tokens 做逐一累加，
                 total_tokens = 两者之和。
        - 元数据（metadata）：后者覆盖前者（dict 的 update 语义）。

    参数:
        outputs: AgentOutput 列表

    返回:
        AgentOutput: 合并后的输出对象

    示例:
        ```python
        output1 = AgentOutput(content="Part 1")
        output2 = AgentOutput(content="Part 2")
        merged = merge_outputs([output1, output2])
        print(merged.content)  # "Part 1\nPart 2"
        ```
    """
    if not outputs:
        # 空列表时返回一个空的 AgentOutput，避免上层还要做 None 判断
        return AgentOutput()

    if len(outputs) == 1:
        # 单元素直接返回原对象，避免不必要的拷贝
        return outputs[0]

    # —— 合并内容 —— #
    contents: List[str] = [o.content for o in outputs if o.has_content()]
    merged_content = "\n".join(contents)

    # —— 合并工具调用 —— #
    merged_tool_calls: List[Dict[str, Any]] = []
    for output in outputs:
        if output.tool_calls:
            merged_tool_calls.extend(output.tool_calls)

    # —— 合并 usage —— #
    merged_usage: Optional[TokenUsage] = None
    if any(o.has_usage() for o in outputs):
        total_prompt = sum(
            o.usage.prompt_tokens  # type: ignore[union-attr]
            for o in outputs
            if o.usage is not None
        )
        total_completion = sum(
            o.usage.completion_tokens  # type: ignore[union-attr]
            for o in outputs
            if o.usage is not None
        )
        merged_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    # —— 合并元数据 —— #
    merged_metadata: Dict[str, Any] = {}
    for output in outputs:
        # 后面的输出可以覆盖前面的同名 key，符合「后写优先」直觉
        merged_metadata.update(output.metadata)

    return AgentOutput(
        content=merged_content,
        tool_calls=merged_tool_calls,
        usage=merged_usage,
        metadata=merged_metadata,
    )
