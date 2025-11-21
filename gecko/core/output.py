# gecko/core/output.py  
from __future__ import annotations  
  
from typing import Any, Dict, List, Optional  
  
from pydantic import BaseModel, Field, field_validator  
  
  
class AgentOutput(BaseModel):  
    """  
    Agent 执行结果对象  
    - content: 最终文本回复  
    - tool_calls: 工具调用列表，默认为空列表（兼容 LLM 返回 None 的场景）  
    - usage: 可选的 token 消耗等统计  
    - raw: 底层模型的原始返回对象  
    """  
    content: str = ""  
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)  
    usage: Optional[Dict[str, Any]] = None  
    raw: Any = None  
  
    model_config = {"arbitrary_types_allowed": True}  
  
    @field_validator("tool_calls", mode="before")  
    @classmethod  
    def ensure_tool_calls(cls, value):  
        return value or []  
