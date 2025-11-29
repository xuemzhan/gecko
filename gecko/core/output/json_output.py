# gecko/core/output/json_output.py
"""
JsonOutput 结构化输出类型

使用场景：
- 对话/工具调用的最终结果是结构化数据（dict / list），
  希望保留为 JSON 形式，而不是直接转成文本。
- 例如：表格解析结果、配置项、校验报告等。

与 AgentOutput 的关系：
- JsonOutput 更偏向「数据结果」，AgentOutput 更偏向「对话消息」。
- 可以通过 to_agent_output() 将 JsonOutput 转换为可读文本形式。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from gecko.core.output.token_usage import TokenUsage
from gecko.core.output.agent_output import AgentOutput


class JsonOutput(BaseModel):
    """
    结构化 JSON 输出模型。

    属性:
        data:     结构化数据，一般为 dict / list，也可以是任意 JSON 兼容类型
        usage:    Token 使用统计（可选）
        metadata: 元数据（例如来源模型、生成时间、schema 版本等）
    """

    data: Any = Field(
        ...,
        description="结构化数据，要求是 JSON 友好的类型（dict/list/标量等）",
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token 使用统计（可选）",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据（如 schema 版本、生成时间等）",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("data", mode="before")
    @classmethod
    def ensure_json_friendly(cls, value: Any) -> Any:
        """
        尽量保证 data 是 JSON 友好的类型。

        这里不做强制转换，只做简单的防御：
        - 若是 Pydantic BaseModel，则转为 dict
        - 其他类型交由上层决定是否可 JSON 序列化
        """
        try:
            from pydantic import BaseModel as _BaseModel  # 局部导入，避免循环引用
        except Exception:
            _BaseModel = None  # type: ignore

        if _BaseModel is not None and isinstance(value, _BaseModel):
            return value.model_dump()

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典形式（适合日志/序列化）。

        返回:
            dict: 包含 data, metadata, usage 的结构
        """
        result: Dict[str, Any] = {
            "data": self.data,
            "metadata": self.metadata,
        }
        if self.usage is not None:
            result["usage"] = self.usage.model_dump(exclude_none=True)
        return result

    def summary(self, max_preview_length: int = 80) -> str:
        """
        生成简短摘要文本，通常用于日志 / CLI 显示。

        参数:
            max_preview_length: 预览字符串的最大长度

        返回:
            str: 摘要文本
        """
        # 尝试用简短的 repr 展示 data
        preview = repr(self.data)
        if len(preview) > max_preview_length:
            preview = preview[:max_preview_length] + "..."

        parts = [f"JSON: {preview}"]

        if self.usage is not None:
            parts.append(f"Tokens: {self.usage.total_tokens}")

        return " | ".join(parts)

    def to_agent_output(self, pretty: bool = False) -> AgentOutput:
        """
        将 JsonOutput 转换为 AgentOutput 形式（方便统一处理）。

        参数:
            pretty: 是否以缩进 JSON 的形式输出 content（便于人类阅读）

        返回:
            AgentOutput 实例，content 为 JSON 字符串
        """
        import json

        if pretty:
            content = json.dumps(self.data, ensure_ascii=False, indent=2)
        else:
            content = json.dumps(self.data, ensure_ascii=False)

        return AgentOutput(
            content=content,
            usage=self.usage,
            metadata={**self.metadata, "from": "JsonOutput"},
        )

    def __str__(self) -> str:
        """默认字符串表示，使用 summary。"""
        return self.summary()

    def __repr__(self) -> str:
        return f"JsonOutput(data_type={type(self.data).__name__}, has_usage={self.usage is not None})"
