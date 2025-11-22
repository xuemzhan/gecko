"""事件基础类型"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    """事件基类"""
    type: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}