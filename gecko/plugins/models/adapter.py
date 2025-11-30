# gecko/plugins/models/adapter.py
from __future__ import annotations

from typing import Any, List, Optional

from gecko.core.protocols import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
    StreamChunk,
)


def safe_access(obj: Any, key: str, default: Any = None) -> Any:
    """
    [工具] 通用属性/字典获取 (增强版)
    
    修复了 Pydantic getattr 抛错和 Mock 对象误判的问题。
    """
    if obj is None:
        return default

    # 1. 优先尝试字典访问 (最安全)
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError, AttributeError):
        pass

    # 2. 尝试属性访问 (需捕获 AttributeError)
    try:
        # 注意：不要使用 hasattr，因为它在某些动态代理对象(如 Mock, Pydantic Lazy)上可能误判
        val = getattr(obj, key)
        return val if val is not None else default
    except (AttributeError, TypeError):
        pass

    return default


class LiteLLMAdapter:
    """
    LiteLLM 响应适配器 (Anti-Corruption Layer)
    
    职责：
    将 LiteLLM 返回的异构对象手动映射为 Gecko 的标准协议对象。
    避免调用 model_dump() 从而消除 Pydantic 序列化警告。
    """

    @staticmethod
    def to_gecko_response(resp: Any) -> CompletionResponse:
        """将 LiteLLM 响应转换为 Gecko CompletionResponse"""
        
        # 1. 提取 Choices
        choices: List[CompletionChoice] = []
        raw_choices = safe_access(resp, "choices", [])
        
        if isinstance(raw_choices, list):
            for c in raw_choices:
                # 提取 Message
                raw_msg = safe_access(c, "message", {})
                message_dict = {
                    "role": safe_access(raw_msg, "role", "assistant"),
                    "content": safe_access(raw_msg, "content", None),
                }
                
                # 提取 Tool Calls
                raw_tool_calls = safe_access(raw_msg, "tool_calls", None)
                if raw_tool_calls:
                    sanitized_tool_calls = []
                    for tc in raw_tool_calls:
                        sanitized_tool_calls.append({
                            "id": safe_access(tc, "id", ""),
                            "type": safe_access(tc, "type", "function"),
                            "function": {
                                "name": safe_access(safe_access(tc, "function"), "name", ""),
                                "arguments": safe_access(safe_access(tc, "function"), "arguments", "")
                            }
                        })
                    message_dict["tool_calls"] = sanitized_tool_calls

                choices.append(CompletionChoice(
                    index=safe_access(c, "index", 0),
                    finish_reason=safe_access(c, "finish_reason", None),
                    message=message_dict,
                    logprobs=safe_access(c, "logprobs", None)
                ))

        # 2. 提取 Usage
        usage = None
        raw_usage = safe_access(resp, "usage", None)
        if raw_usage:
            usage = CompletionUsage(
                prompt_tokens=safe_access(raw_usage, "prompt_tokens", 0),
                completion_tokens=safe_access(raw_usage, "completion_tokens", 0),
                total_tokens=safe_access(raw_usage, "total_tokens", 0)
            )

        # 3. 构建最终响应
        return CompletionResponse(
            id=safe_access(resp, "id", ""),
            object=safe_access(resp, "object", "chat.completion"),
            created=safe_access(resp, "created", 0),
            model=safe_access(resp, "model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=safe_access(resp, "system_fingerprint", None),
            metadata=safe_access(resp, "_hidden_params", {})
        )

    @staticmethod
    def to_gecko_chunk(chunk: Any) -> Optional[StreamChunk]:
        """
        将 LiteLLM 流式块转换为 StreamChunk
        """
        raw_choices = safe_access(chunk, "choices", [])
        
        # 过滤 Keep-Alive 空包
        # 有些 provider 会发送仅含 id 的包作为心跳，或者完全空的包
        if not raw_choices and not safe_access(chunk, "id"):
            return None

        mapped_choices = []
        if isinstance(raw_choices, list):
            for c in raw_choices:
                delta = safe_access(c, "delta", {})
                
                # [修复] 确保 delta 是字典，防止 AttributeError
                if not isinstance(delta, dict):
                    # 某些情况下 delta 可能是 Pydantic 对象
                    if hasattr(delta, "model_dump"):
                        delta = delta.model_dump()
                    elif hasattr(delta, "dict"):
                        delta = delta.dict()
                    else:
                        delta = {}

                mapped_choices.append({
                    "index": safe_access(c, "index", 0),
                    "delta": {
                        "role": safe_access(delta, "role", None),
                        "content": safe_access(delta, "content", None),
                        "tool_calls": safe_access(delta, "tool_calls", None)
                    },
                    "finish_reason": safe_access(c, "finish_reason", None)
                })

        return StreamChunk(
            id=safe_access(chunk, "id", ""),
            created=safe_access(chunk, "created", 0),
            model=safe_access(chunk, "model", ""),
            choices=mapped_choices
        )