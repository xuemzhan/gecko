# gecko/core/output/agent_output.py
"""
Agent 输出模型

本模块定义 Agent 执行后的标准输出结构：
- 最终回复内容（content）
- 工具调用信息（tool_calls）
- Token 使用统计（usage）
- 原始模型响应（raw）
- 附加元数据（metadata）

后续如果需要扩展不同形态的输出（如 StreamingOutput、JsonSchemaOutput 等），
可以在本模块旁新增新的模型，或从 AgentOutput 继承。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from gecko.core.logging import get_logger
from gecko.core.output.token_usage import TokenUsage

logger = get_logger(__name__)


class AgentOutput(BaseModel):
    """
    Agent 执行结果

    属性:
        content: 最终文本回复（可能为空，如果只有工具调用）
        tool_calls: 工具调用列表（OpenAI 工具调用格式）
        usage: Token 使用统计（可选）
        raw:  原始模型响应（用于调试，不保证可 JSON 序列化）
        metadata: 附加元数据（例如模型名、耗时、include_raw 标记等）
    """

    content: str = Field(
        default="",
        description="最终文本回复",
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="工具调用列表（OpenAI 工具调用格式）",
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token 使用统计",
    )
    raw: Any = Field(
        default=None,
        description="原始模型响应（用于调试），类型不做强约束",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据（例如模型名、请求耗时、调试开关等）",
    )

    # 允许 raw 等字段为任意类型
    model_config = {"arbitrary_types_allowed": True}

    # ------ 字段校验器（构造阶段自动执行） ------ #

    @field_validator("tool_calls", mode="before")
    @classmethod
    def ensure_tool_calls(cls, value: Any) -> List[Dict[str, Any]]:
        """
        确保 tool_calls 始终是列表。

        兼容性与容错策略：
        - None            -> []
        - list            -> 原样返回
        - tuple / set     -> 转为 list，并记录一条 warning（提示调用方修正）
        - dict（单个调用） -> 包装为 [dict]，并记录 warning
        - 其他非法类型      -> 返回 []，记录 warning

        这样可以最大程度避免「调用方类型传错导致工具调用被静默丢弃」的问题，
        同时通过日志提示引导调用方修复。
        """
        if value is None:
            return []

        # 正常情况：已经是 list
        if isinstance(value, list):
            return value

        # 常见误用 1：tuple / set
        if isinstance(value, (tuple, set)):
            converted = list(value)
            logger.warning(
                "tool_calls converted to list from %s",
                type(value).__name__,
            )
            return converted

        # 常见误用 2：单个 dict
        if isinstance(value, dict):
            logger.warning("tool_calls single dict converted to list")
            return [value]

        # 其他异常类型：强制为空，但保留 warning 方便定位
        logger.warning(
            "tool_calls should be a list or dict-like, got %s",
            type(value).__name__,
        )
        return []

    @field_validator("content", mode="before")
    @classmethod
    def ensure_content(cls, value: Any) -> str:
        """
        确保 content 最终是字符串。

        规则：
        - None       -> ""
        - str        -> 原样返回
        - 其他类型    -> 调用 str()，以避免因为类型不符导致序列化报错。
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # 容忍上层传入非字符串类型（如 Message 对象），尽量转换为 str
        return str(value)

    # ===== 检查方法 =====

    def has_content(self) -> bool:
        """
        检查是否有「非空白」文本内容。

        返回:
            bool: 是否有非空文本（会 strip 空白）
        """
        return bool(self.content and self.content.strip())

    def has_tool_calls(self) -> bool:
        """
        检查是否有工具调用。

        返回:
            bool: 是否包含工具调用
        """
        return len(self.tool_calls) > 0

    def tool_call_count(self) -> int:
        """
        获取工具调用数量。

        返回:
            int: 工具调用的数量
        """
        return len(self.tool_calls)

    def is_empty(self) -> bool:
        """
        检查输出是否完全为空。

        定义为：
        - 没有非空文本内容，且
        - 没有任何工具调用

        返回:
            bool: 是否既无内容也无工具调用
        """
        return not self.has_content() and not self.has_tool_calls()

    def has_usage(self) -> bool:
        """
        检查是否有 usage 信息。

        返回:
            bool: 是否包含 token 使用统计
        """
        return self.usage is not None

    # ===== 提取方法 =====

    def get_tool_names(self) -> List[str]:
        """
        提取所有被调用的工具名称。

        返回:
            List[str]: 工具名称列表
        """
        names: List[str] = []
        for call in self.tool_calls:
            func = call.get("function", {})
            name = func.get("name")
            if name:
                names.append(name)
        return names

    def get_tool_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取工具调用。

        参数:
            call_id: 工具调用 ID

        返回:
            dict | None: 工具调用字典，如果不存在返回 None
        """
        for call in self.tool_calls:
            if call.get("id") == call_id:
                return call
        return None

    def get_text_preview(self, max_length: int = 100) -> str:
        """
        获取内容预览（用于日志/显示）。

        参数:
            max_length: 最大预览长度（超出会截断并追加 '...'）

        返回:
            str: 截断后的文本预览
        """
        if not self.content:
            return ""

        if len(self.content) <= max_length:
            return self.content

        return self.content[:max_length] + "..."

    # ===== 转换方法 =====

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（便于序列化 / 日志记录）。

        注意：
        - usage 字段通过 Pydantic 的 model_dump 导出。
        - raw 字段默认不导出，只有在 metadata 中显式设置
          include_raw=True 且 raw is not None 时才导出。
        - raw 尽量保留原始结构（dict/list 等），由上层决定是否做 JSON 序列化。
        """
        data: Dict[str, Any] = {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }

        if self.usage is not None:
            # exclude_none=True 可以避免把 None 写进字典，减小日志体积
            data["usage"] = self.usage.model_dump(exclude_none=True)

        # raw 字段可能无法直接 JSON 序列化，仅在调试模式下包含
        if self.metadata.get("include_raw") and self.raw is not None:
            try:
                # 若 raw 已经是常见的 JSON 友好类型（dict/list/标量），直接保留结构
                if isinstance(self.raw, (dict, list, tuple, str, int, float, bool)):
                    data["raw"] = self.raw
                else:
                    # 非常规类型（如复杂对象），退而求其次转成字符串
                    data["raw"] = str(self.raw)
            except Exception:
                # 一旦转换异常，给出占位文本，避免序列化失败
                data["raw"] = "<non-serializable>"

        return data

    def to_message_dict(self) -> Dict[str, Any]:
        """
        转换为 OpenAI 消息格式（用于下一轮对话）。

        返回:
            dict: 符合 OpenAI API 的消息字典
        """
        msg: Dict[str, Any] = {
            "role": "assistant",
            # OpenAI 允许 content 为 null，此处用 None 表示「无内容，仅工具调用」
            "content": self.content or None,
        }

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        return msg

    # ===== 格式化输出 =====

    def format(self, include_metadata: bool = False) -> str:
        """
        格式化输出为可读文本（多段落），适合打印到控制台或日志。

        参数:
            include_metadata: 是否包含元数据段落

        返回:
            str: 格式化后的字符串
        """
        lines: List[str] = []

        # —— 内容 —— #
        if self.has_content():
            lines.append("=== 回复内容 ===")
            lines.append(self.content)
            lines.append("")

        # —— 工具调用 —— #
        if self.has_tool_calls():
            lines.append("=== 工具调用 ===")
            for i, call in enumerate(self.tool_calls, 1):
                func = call.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                lines.append(f"{i}. {name}")
                lines.append(f"   参数: {args}")
            lines.append("")

        # —— Token 使用 —— #
        if self.usage:
            lines.append("=== Token 使用 ===")
            lines.append(f"输入: {self.usage.prompt_tokens}")
            lines.append(f"输出: {self.usage.completion_tokens}")
            lines.append(f"总计: {self.usage.total_tokens}")
            lines.append("")

        # —— 元数据 —— #
        if include_metadata and self.metadata:
            lines.append("=== 元数据 ===")
            for key, value in self.metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """
        生成简短摘要（一行），适合在列表 / 简要日志中展示。

        返回:
            str: 一行摘要文本
        """
        parts: List[str] = []

        if self.has_content():
            preview = self.get_text_preview(30)
            parts.append(f"回复: {preview}")

        if self.has_tool_calls():
            parts.append(f"工具调用: {self.tool_call_count()}")

        if self.usage:
            parts.append(f"Tokens: {self.usage.total_tokens}")

        if not parts:
            return "空输出"

        return " | ".join(parts)

    # ===== 统计方法 =====

    def get_stats(self) -> Dict[str, Any]:
        """
        获取输出统计信息（结构化）。

        返回:
            dict: 包含各种统计数据的字典

        字段说明:
            - content_length: 文本内容长度（包括空白字符）
            - has_content: 是否存在非空白内容
            - tool_call_count: 工具调用数量
            - tool_names: 工具名称列表
            - is_empty: 是否既无内容也无工具调用
            - usage（可选）: token 使用统计快照
        """
        stats: Dict[str, Any] = {
            "content_length": len(self.content),  # 原始长度，用于粗略估算字数
            "has_content": self.has_content(),
            "tool_call_count": self.tool_call_count(),
            "tool_names": self.get_tool_names(),
            "is_empty": self.is_empty(),
        }

        if self.usage:
            stats["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }

        return stats

    # ===== 字符串 & 布尔表示 =====

    def __str__(self) -> str:
        """使用 summary 作为默认字符串表示，便于日志阅读。"""
        return self.summary()

    def __repr__(self) -> str:
        """提供稍详细的调试信息。"""
        return (
            f"AgentOutput("
            f"content_length={len(self.content)}, "
            f"tool_calls={self.tool_call_count()}, "
            f"has_usage={self.has_usage()}"
            f")"
        )

    def __bool__(self) -> bool:
        """
        布尔值转换（是否有有效输出）。

        用于 if output: ... 这种写法，语义为「是否不为空输出」。
        """
        return not self.is_empty()
