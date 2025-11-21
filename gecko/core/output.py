# gecko/core/output.py
"""
Agent 输出模型

定义 Agent 执行后的标准输出格式，包含：
- 最终回复内容
- 工具调用信息
- Token 使用统计
- 原始模型响应

优化点：
1. 结构化的 Usage 模型
2. 输出验证和后处理
3. 丰富的工具方法
4. 格式化输出
5. 统计信息提取
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from gecko.core.logging import get_logger

logger = get_logger(__name__)


# ===== Token 使用统计 =====

class TokenUsage(BaseModel):
    """
    Token 使用统计
    
    符合 OpenAI API 的 usage 格式
    
    示例:
        ```python
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        ```
    """
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="提示词（输入）消耗的 tokens"
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="生成（输出）消耗的 tokens"
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="总消耗 tokens"
    )
    
    # 扩展字段（某些模型提供）
    prompt_tokens_details: Optional[Dict[str, int]] = Field(
        default=None,
        description="提示词 tokens 详细信息"
    )
    completion_tokens_details: Optional[Dict[str, int]] = Field(
        default=None,
        description="生成 tokens 详细信息"
    )

    @model_validator(mode="after")
    def validate_total(self):
        """验证总 tokens 是否正确"""
        calculated_total = self.prompt_tokens + self.completion_tokens
        
        # 如果 total_tokens 为 0，自动计算
        if self.total_tokens == 0:
            self.total_tokens = calculated_total
        
        # 如果不一致，记录警告
        elif self.total_tokens != calculated_total:
            logger.warning(
                "Token usage total mismatch",
                total=self.total_tokens,
                calculated=calculated_total
            )
        
        return self

    def get_cost_estimate(
        self,
        prompt_price_per_1k: float = 0.0,
        completion_price_per_1k: float = 0.0
    ) -> float:
        """
        估算成本（美元）
        
        参数:
            prompt_price_per_1k: 输入 token 每 1000 个的价格
            completion_price_per_1k: 输出 token 每 1000 个的价格
        
        返回:
            估算成本（美元）
        
        示例:
            ```python
            # GPT-4 价格（示例）
            cost = usage.get_cost_estimate(
                prompt_price_per_1k=0.03,      # $0.03/1K tokens
                completion_price_per_1k=0.06   # $0.06/1K tokens
            )
            print(f"Estimated cost: ${cost:.4f}")
            ```
        """
        prompt_cost = (self.prompt_tokens / 1000) * prompt_price_per_1k
        completion_cost = (self.completion_tokens / 1000) * completion_price_per_1k
        return prompt_cost + completion_cost

    def __str__(self) -> str:
        """简洁的字符串表示"""
        return (
            f"TokenUsage("
            f"prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}"
            f")"
        )


# ===== Agent 输出 =====

class AgentOutput(BaseModel):
    """
    Agent 执行结果
    
    包含 Agent 执行后的完整输出信息。
    
    属性:
        content: 最终文本回复（可能为空，如果只有工具调用）
        tool_calls: 工具调用列表
        usage: Token 使用统计（可选）
        raw: 原始模型响应（用于调试）
        metadata: 附加元数据
    
    示例:
        ```python
        # 简单文本输出
        output = AgentOutput(content="Hello, how can I help?")
        
        # 带工具调用的输出
        output = AgentOutput(
            content="I'll search for that information.",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "AI"}'
                    }
                }
            ],
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50
            )
        )
        
        # 检查输出
        if output.has_tool_calls():
            print(f"需要执行 {output.tool_call_count()} 个工具")
        
        if output.has_content():
            print(f"回复: {output.content}")
        ```
    """
    content: str = Field(
        default="",
        description="最终文本回复"
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="工具调用列表（OpenAI 格式）"
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token 使用统计"
    )
    raw: Any = Field(
        default=None,
        description="原始模型响应（用于调试）"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据"
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("tool_calls", mode="before")
    @classmethod
    def ensure_tool_calls(cls, value):
        """确保 tool_calls 始终是列表"""
        if value is None:
            return []
        if not isinstance(value, list):
            logger.warning("tool_calls should be a list", type=type(value).__name__)
            return []
        return value

    @field_validator("content", mode="before")
    @classmethod
    def ensure_content(cls, value):
        """确保 content 是字符串"""
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    # ===== 检查方法 =====

    def has_content(self) -> bool:
        """
        检查是否有文本内容
        
        返回:
            是否有非空文本
        """
        return bool(self.content and self.content.strip())

    def has_tool_calls(self) -> bool:
        """
        检查是否有工具调用
        
        返回:
            是否包含工具调用
        """
        return len(self.tool_calls) > 0

    def tool_call_count(self) -> int:
        """
        获取工具调用数量
        
        返回:
            工具调用的数量
        """
        return len(self.tool_calls)

    def is_empty(self) -> bool:
        """
        检查输出是否完全为空
        
        返回:
            是否既无内容也无工具调用
        """
        return not self.has_content() and not self.has_tool_calls()

    def has_usage(self) -> bool:
        """
        检查是否有 usage 信息
        
        返回:
            是否包含 token 使用统计
        """
        return self.usage is not None

    # ===== 提取方法 =====

    def get_tool_names(self) -> List[str]:
        """
        提取所有被调用的工具名称
        
        返回:
            工具名称列表
        
        示例:
            ```python
            output = AgentOutput(tool_calls=[...])
            tools = output.get_tool_names()
            print(f"调用的工具: {', '.join(tools)}")
            ```
        """
        names = []
        for call in self.tool_calls:
            func = call.get("function", {})
            name = func.get("name")
            if name:
                names.append(name)
        return names

    def get_tool_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取工具调用
        
        参数:
            call_id: 工具调用 ID
        
        返回:
            工具调用字典，如果不存在返回 None
        """
        for call in self.tool_calls:
            if call.get("id") == call_id:
                return call
        return None

    def get_text_preview(self, max_length: int = 100) -> str:
        """
        获取内容预览（用于日志/显示）
        
        参数:
            max_length: 最大长度
        
        返回:
            截断后的文本预览
        """
        if not self.content:
            return ""
        
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length] + "..."

    # ===== 转换方法 =====

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（便于序列化）
        
        返回:
            包含所有字段的字典
        """
        data = {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }
        
        if self.usage:
            data["usage"] = self.usage.model_dump()
        
        # raw 字段可能无法序列化，仅在调试模式下包含
        if self.raw and self.metadata.get("include_raw"):
            try:
                data["raw"] = str(self.raw)
            except Exception:
                data["raw"] = "<non-serializable>"
        
        return data

    def to_message_dict(self) -> Dict[str, Any]:
        """
        转换为 OpenAI 消息格式（用于下一轮对话）
        
        返回:
            符合 OpenAI API 的消息字典
        
        示例:
            ```python
            output = AgentOutput(content="Hello", tool_calls=[...])
            msg_dict = output.to_message_dict()
            # 可以直接添加到对话历史
            ```
        """
        msg = {
            "role": "assistant",
            "content": self.content or None,  # OpenAI 允许 null
        }
        
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        
        return msg

    # ===== 格式化输出 =====

    def format(self, include_metadata: bool = False) -> str:
        """
        格式化输出为可读文本
        
        参数:
            include_metadata: 是否包含元数据
        
        返回:
            格式化后的字符串
        
        示例:
            ```python
            output = AgentOutput(...)
            print(output.format())
            ```
        """
        lines = []
        
        # 内容
        if self.has_content():
            lines.append("=== 回复内容 ===")
            lines.append(self.content)
            lines.append("")
        
        # 工具调用
        if self.has_tool_calls():
            lines.append("=== 工具调用 ===")
            for i, call in enumerate(self.tool_calls, 1):
                func = call.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                lines.append(f"{i}. {name}")
                lines.append(f"   参数: {args}")
            lines.append("")
        
        # Token 使用
        if self.usage:
            lines.append("=== Token 使用 ===")
            lines.append(f"输入: {self.usage.prompt_tokens}")
            lines.append(f"输出: {self.usage.completion_tokens}")
            lines.append(f"总计: {self.usage.total_tokens}")
            lines.append("")
        
        # 元数据
        if include_metadata and self.metadata:
            lines.append("=== 元数据 ===")
            for key, value in self.metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        return "\n".join(lines)

    def summary(self) -> str:
        """
        生成简短摘要
        
        返回:
            一行摘要文本
        
        示例:
            ```python
            output = AgentOutput(...)
            print(output.summary())
            # 输出: "回复: Hello... (50 chars) | 工具调用: 2 | Tokens: 150"
            ```
        """
        parts = []
        
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
        获取输出统计信息
        
        返回:
            包含各种统计数据的字典
        """
        stats = {
            "content_length": len(self.content),
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

    # ===== 字符串表示 =====

    def __str__(self) -> str:
        """简洁的字符串表示"""
        return self.summary()

    def __repr__(self) -> str:
        """详细的字符串表示"""
        return (
            f"AgentOutput("
            f"content_length={len(self.content)}, "
            f"tool_calls={self.tool_call_count()}, "
            f"has_usage={self.has_usage()}"
            f")"
        )

    def __bool__(self) -> bool:
        """
        布尔值转换（是否有有效输出）
        
        返回:
            是否不为空
        """
        return not self.is_empty()


# ===== 工具函数 =====

def create_text_output(
    content: str,
    usage: Optional[TokenUsage] = None,
    **metadata
) -> AgentOutput:
    """
    快速创建纯文本输出
    
    参数:
        content: 文本内容
        usage: Token 使用统计（可选）
        **metadata: 附加元数据
    
    返回:
        AgentOutput 实例
    
    示例:
        ```python
        output = create_text_output(
            "Hello, world!",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5)
        )
        ```
    """
    return AgentOutput(
        content=content,
        usage=usage,
        metadata=metadata
    )


def create_tool_output(
    tool_calls: List[Dict[str, Any]],
    content: str = "",
    usage: Optional[TokenUsage] = None,
    **metadata
) -> AgentOutput:
    """
    快速创建工具调用输出
    
    参数:
        tool_calls: 工具调用列表
        content: 可选的文本内容
        usage: Token 使用统计（可选）
        **metadata: 附加元数据
    
    返回:
        AgentOutput 实例
    """
    return AgentOutput(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        metadata=metadata
    )


def merge_outputs(outputs: List[AgentOutput]) -> AgentOutput:
    """
    合并多个输出（用于多 Agent 场景）
    
    参数:
        outputs: AgentOutput 列表
    
    返回:
        合并后的 AgentOutput
    
    策略:
        - 内容：用换行符连接
        - 工具调用：合并所有
        - Usage：累加 tokens
        - 元数据：合并（后者覆盖前者）
    
    示例:
        ```python
        output1 = AgentOutput(content="Part 1")
        output2 = AgentOutput(content="Part 2")
        merged = merge_outputs([output1, output2])
        print(merged.content)  # "Part 1\nPart 2"
        ```
    """
    if not outputs:
        return AgentOutput()
    
    if len(outputs) == 1:
        return outputs[0]
    
    # 合并内容
    contents = [o.content for o in outputs if o.has_content()]
    merged_content = "\n".join(contents)
    
    # 合并工具调用
    merged_tool_calls = []
    for output in outputs:
        merged_tool_calls.extend(output.tool_calls)
    
    # 合并 usage
    merged_usage = None
    if any(o.has_usage() for o in outputs):
        total_prompt = sum(
            o.usage.prompt_tokens for o in outputs if o.usage
        )
        total_completion = sum(
            o.usage.completion_tokens for o in outputs if o.usage
        )
        merged_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion
        )
    
    # 合并元数据
    merged_metadata = {}
    for output in outputs:
        merged_metadata.update(output.metadata)
    
    return AgentOutput(
        content=merged_content,
        tool_calls=merged_tool_calls,
        usage=merged_usage,
        metadata=merged_metadata
    )