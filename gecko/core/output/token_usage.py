# gecko/core/output/token_usage.py
"""
Token 使用统计模型

本模块单独抽出 TokenUsage，便于：
- 与不同模型提供商的 usage 字段对齐/适配
- 后续扩展更多维度（如 cached_tokens、reasoning_tokens 等）
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator

from gecko.core.logging import get_logger

logger = get_logger(__name__)


class TokenUsage(BaseModel):
    """
    Token 使用统计

    设计目标：
    - 与 OpenAI API 的 usage 结构基本兼容
    - 在调用方缺省 total_tokens 时自动计算
    - 保留 provider 提供的原始 total_tokens（如有冲突，仅记录 warning）

    示例:
        ```python
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        ```
    """

    # —— 基本字段 —— #
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="提示词（输入）消耗的 tokens",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="生成（输出）消耗的 tokens",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="总消耗 tokens",
    )

    # —— 扩展字段（部分模型会提供更细粒度统计） —— #
    prompt_tokens_details: Optional[Dict[str, int]] = Field(
        default=None,
        description="提示词 tokens 详细信息，例如 {'cached_tokens': 10}",
    )
    completion_tokens_details: Optional[Dict[str, int]] = Field(
        default=None,
        description="生成 tokens 详细信息",
    )

    @model_validator(mode="after")
    def validate_total(self) -> "TokenUsage":
        """
        验证并补全 total_tokens 字段。

        逻辑说明：
        - calculated_total = prompt_tokens + completion_tokens
        - 如果 total_tokens 仍为 0 且 calculated_total > 0：
            认为调用方未显式提供 total_tokens，自动补全。
        - 如果 total_tokens 非 0 且与 calculated_total 不一致：
            不改写 total_tokens，只记录 warning 方便排查。
        - 如果三者都是 0：
            保持为 0，不做任何额外处理。
        """
        calculated_total = self.prompt_tokens + self.completion_tokens

        # 情况一：total_tokens 默认值 0，且显然有 token 消耗，则自动补全
        if self.total_tokens == 0 and calculated_total > 0:
            self.total_tokens = calculated_total

        # 情况二：provider 明确给了 total_tokens（非 0），但与计算值不一致，仅告警
        elif self.total_tokens != 0 and self.total_tokens != calculated_total:
            logger.warning(
                "Token usage total mismatch",
                total=self.total_tokens,
                calculated=calculated_total,
            )

        # 情况三：三者皆 0，保持 0，不做干预
        return self

    def get_cost_estimate(
        self,
        prompt_price_per_1k: float = 0.0,
        completion_price_per_1k: float = 0.0,
    ) -> float:
        """
        估算成本（美元）

        说明：
        - 使用 prompt_tokens 和 completion_tokens 分开计费，
          不使用 total_tokens，符合大多数厂商的计费模式。
        - 如果价格参数为 0，返回值自然为 0，不做额外处理。

        参数:
            prompt_price_per_1k: 输入 token 每 1000 个的价格
            completion_price_per_1k: 输出 token 每 1000 个的价格

        返回:
            估算成本（美元）
        """
        prompt_cost = (self.prompt_tokens / 1000) * prompt_price_per_1k
        completion_cost = (self.completion_tokens / 1000) * completion_price_per_1k
        return prompt_cost + completion_cost

    def __str__(self) -> str:
        """简洁的字符串表示，便于日志打印。"""
        return (
            f"TokenUsage("
            f"prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}"
            f")"
        )
