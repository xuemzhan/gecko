# gecko/core/builder.py
from typing import Any, Dict, List, Optional, Self

from gecko.core.agent import Agent
from gecko.core.session import Session
from gecko.core.events import EventBus
from gecko.core.runner import AsyncRunner
from gecko.core.protocols import ModelProtocol

class AgentBuilder:
    def __init__(self):
        self._model_instance: ModelProtocol | None = None
        self._tools: List[Any] = []
        self._session: Optional[Session] = None
        self._event_bus: Optional[EventBus] = None
        self._kwargs: Dict[str, Any] = {}

    def with_model(self, model_instance: ModelProtocol) -> Self:
        """
        注入模型实例。模型必须实现 ModelProtocol。
        """
        if not hasattr(model_instance, "acompletion"):
            raise ValueError("Model must implement async acompletion method")
        
        self._model_instance = model_instance
        return self

    def with_tools(self, tools: List[Any]) -> Self:
        self._tools.extend(tools)
        return self

    def with_session(self, session: Session) -> Self:
        self._session = session
        return self

    def with_event_bus(self, event_bus: EventBus) -> Self:
        self._event_bus = event_bus
        return self

    def with_kwargs(self, **kwargs: Any) -> Self:
        self._kwargs.update(kwargs)
        return self
    
    def with_session_storage_url(self, url: str) -> Self:
        from gecko.plugins.storage.factory import get_storage_by_url
        self._kwargs["session_storage"] = get_storage_by_url(url, required="session")
        return self

    def with_vector_storage_url(self, url: str) -> Self:
        from gecko.plugins.storage.factory import get_storage_by_url
        self._kwargs["vector_storage"] = get_storage_by_url(url, required="vector")
        return self

    def build(self) -> Agent:
        if not self._model_instance:
            # [修复] 移除括号 (...)，避免 pytest match 正则歧义
            raise ValueError("Model is required. Call with_model first.")

        return Agent(
            model=self._model_instance,
            tools=self._tools,
            session=self._session,
            event_bus=self._event_bus,
            **self._kwargs
        )