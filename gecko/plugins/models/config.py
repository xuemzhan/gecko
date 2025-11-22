# gecko/plugins/models/config.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """
    模型通用配置对象
    """
    model_name: str = Field(..., description="模型名称，如 'gpt-4o', 'ollama/llama3'")
    
    # 驱动类型 (不再是 Enum，支持字符串扩展)
    driver_type: str = Field(default="litellm", description="驱动类型: litellm, openai_native, etc.")
    
    # 连接配置
    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: Optional[str] = Field(default=None, description="API Base URL (本地模型必填)")
    timeout: float = Field(default=60.0, description="请求超时时间(秒)")
    max_retries: int = Field(default=2, description="最大重试次数")
    
    # 运行时参数透传 (Provider Specific)
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict, description="透传给底层驱动的额外参数")
    
    # 能力标识 (Capability Flags)
    supports_vision: bool = Field(default=False, description="是否支持视觉输入")
    supports_audio: bool = Field(default=False, description="是否支持音频输入")
    supports_function_calling: bool = Field(default=True, description="是否支持工具调用")