# gecko/core/message.py
from __future__ import annotations

from typing import Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: Role
    content: Union[str, List[Dict[str, Any]]] = Field(default="")
    name: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None
    tool_call_id: str | None = None