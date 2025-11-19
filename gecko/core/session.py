# gecko/core/session.py
from typing import Dict, Any

class Session:
    def __init__(self, session_id: str | None = None, state: Dict[str, Any] | None = None):
        self.session_id = session_id or "default"
        self.state = state or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any):
        self.state[key] = value