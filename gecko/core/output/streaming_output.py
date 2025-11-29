# gecko/core/output/streaming_output.py
"""
StreamingOutput 流式输出类型

使用场景：
- LLM/Agent 支持流式输出（stream=True），每次返回一个小片段（chunk）。
- 希望在框架内部以统一结构记录所有 chunk，并在结束时汇总成一个 AgentOutput。

核心结构：
- StreamingChunk : 表示一次增量输出
- StreamingOutput: 维护所有 chunk，并提供：
    - append_chunk(...)
    - iter_contents()
    - finalize() -> AgentOutput
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

from gecko.core.output.token_usage import TokenUsage
from gecko.core.output.agent_output import AgentOutput


class StreamingChunk(BaseModel):
    """
    单个流式输出片段。

    属性:
        index:        片段序号（递增，便于调试/排序）
        content_delta: 本次新增的文本内容（增量）
        tool_calls_delta: 本次新增的工具调用（通常为空，部分模型支持流式工具调用）
        usage_delta:  本次片段对应的增量 token 使用统计（可选）
        raw:          原始 provider 返回的该片段响应（用于调试）
        metadata:     附加元数据（如时间戳、来源模型等）
    """

    index: int = Field(
        ...,
        ge=0,
        description="片段序号（从 0 开始递增）",
    )
    content_delta: str = Field(
        default="",
        description="本次新增的文本内容（增量）",
    )
    tool_calls_delta: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="本次新增的工具调用（少数模型支持流式工具调用）",
    )
    usage_delta: Optional[TokenUsage] = Field(
        default=None,
        description="本次片段对应的增量 token 使用统计（可选）",
    )
    raw: Any = Field(
        default=None,
        description="原始 provider 返回的该片段响应（用于调试）",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据（如时间戳、来源模型等）",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("content_delta", mode="before")
    @classmethod
    def ensure_str(cls, value: Any) -> str:
        """
        内容增量统一转成字符串，防止 provider 返回奇怪类型。
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)


class StreamingOutput(BaseModel):
    """
    流式输出聚合器。

    核心职责：
    - 持有多个 StreamingChunk
    - 提供追加接口（append_chunk）
    - 支持遍历所有文本增量（iter_contents）
    - 在流结束时汇总成一个完整的 AgentOutput（finalize）

    注意：
    - StreamingOutput 本身不直接提供 content 字段，
      而是通过 finalize() 生成最终 AgentOutput。
    """

    chunks: List[StreamingChunk] = Field(
        default_factory=list,
        description="按顺序记录的所有流式输出片段",
    )
    # 可选：整体级别的 usage（有的 provider 只在流末尾给总 usage）
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="整体 Token 使用统计（可选，不同于各个 chunk 的增量）",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="整体元数据（例如模型名称、会话 ID 等）",
    )

    model_config = {"arbitrary_types_allowed": True}

    # ===== 追加 & 遍历 =====

    def append_chunk(self, chunk: StreamingChunk) -> None:
        """
        追加一个流式片段。

        使用方式：
            streaming = StreamingOutput()
            streaming.append_chunk(StreamingChunk(index=0, content_delta="Hel"))
            streaming.append_chunk(StreamingChunk(index=1, content_delta="lo"))
        """
        self.chunks.append(chunk)

    def extend_chunks(self, chunks: Iterable[StreamingChunk]) -> None:
        """
        一次性追加多个片段。
        """
        self.chunks.extend(chunks)

    def iter_contents(self) -> Iterable[str]:
        """
        迭代每个片段的 content_delta，适合边消费边打印。

        示例：
            for delta in streaming.iter_contents():
                print(delta, end="", flush=True)
        """
        for chunk in self.chunks:
            if chunk.content_delta:
                yield chunk.content_delta

    # ===== 汇总逻辑 =====

    def _aggregate_content(self) -> str:
        """
        内部方法：将所有 chunk 的 content_delta 按 index 排序后拼接。
        """
        # 保险起见，按 index 排序一次（如果调用方保证按顺序 append，可忽略排序开销）
        sorted_chunks = sorted(self.chunks, key=lambda c: c.index)
        return "".join(c.content_delta for c in sorted_chunks if c.content_delta)

    def _aggregate_tool_calls(self) -> List[Dict[str, Any]]:
        """
        内部方法：合并所有 chunk 的 tool_calls_delta。
        """
        merged: List[Dict[str, Any]] = []
        for c in self.chunks:
            if c.tool_calls_delta:
                merged.extend(c.tool_calls_delta)
        return merged

    def _aggregate_usage(self) -> Optional[TokenUsage]:
        """
        内部方法：汇总 usage。

        策略：
        - 若整体 self.usage 已存在，直接返回（通常是 provider 在流末尾给的）。
        - 否则尝试对所有 chunk 的 usage_delta 做累加。
        """
        if self.usage is not None:
            return self.usage

        # 汇总 chunk 级别的 usage_delta
        deltas = [c.usage_delta for c in self.chunks if c.usage_delta is not None]
        if not deltas:
            return None

        total_prompt = sum(u.prompt_tokens for u in deltas)
        total_completion = sum(u.completion_tokens for u in deltas)

        return TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    def finalize(self) -> AgentOutput:
        """
        将所有流式片段汇总为一个完整的 AgentOutput。

        返回:
            AgentOutput 实例，其中：
                - content 为所有 content_delta 的拼接结果
                - tool_calls 为所有 tool_calls_delta 的合并结果
                - usage 为汇总后的 TokenUsage（若可用）
                - metadata 为 StreamingOutput.metadata
        """
        content = self._aggregate_content()
        tool_calls = self._aggregate_tool_calls()
        usage = self._aggregate_usage()

        return AgentOutput(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata={**self.metadata, "from": "StreamingOutput"},
        )

    # ===== 辅助方法 =====

    def get_stats(self) -> Dict[str, Any]:
        """
        获取流式输出的统计信息（结构化）。

        返回:
            dict，包含：
                - chunk_count: 片段数量
                - total_content_length: 所有增量内容拼接后的长度
                - has_usage: 是否包含 usage（整体或增量）
        """
        aggregated_content = self._aggregate_content()
        has_usage = self.usage is not None or any(
            c.usage_delta is not None for c in self.chunks
        )

        return {
            "chunk_count": len(self.chunks),
            "total_content_length": len(aggregated_content),
            "has_usage": has_usage,
        }

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"StreamingOutput(chunks={stats['chunk_count']}, "
            f"total_content_length={stats['total_content_length']}, "
            f"has_usage={stats['has_usage']})"
        )

    def __repr__(self) -> str:
        return str(self)
