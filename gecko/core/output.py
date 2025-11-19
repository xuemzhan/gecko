from pydantic import BaseModel, Field
from typing import Any

class AgentOutput(BaseModel):
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw: Any = None  # 原始模型响应，便于调试