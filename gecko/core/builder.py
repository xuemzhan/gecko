# gecko/core/builder.py
from typing import Any, Dict, List, Optional, Self

from gecko.core.agent import Agent
from gecko.core.session import Session
from gecko.core.events import EventBus
from gecko.core.protocols import ModelProtocol

class AgentBuilder:
    def __init__(self):
        self._model_instance: ModelProtocol | None = None
        self._tools: List[Any] = []
        self._session: Optional[Session] = None
        self._event_bus: Optional[EventBus] = None
        self._storage: Optional[Any] = None
        self._vector_storage: Optional[Any] = None  # [新增] 向量存储字段
        self._kwargs: Dict[str, Any] = {}

    def with_model(self, model_instance: ModelProtocol) -> Self:
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
    
    def with_storage(self, url: str) -> Self:
        """配置 Agent 的 Session 持久化存储"""
        from gecko.plugins.storage.factory import get_storage_by_url
        self._storage = get_storage_by_url(url, required="session")
        return self
    
    def with_session_storage_url(self, url: str) -> Self:
        """with_storage 的别名，兼容旧 API"""
        return self.with_storage(url)

    # [新增] 修复 AttributeError
    def with_vector_storage_url(self, url: str) -> Self:
        """配置 Vector 存储（RAG 用）"""
        from gecko.plugins.storage.factory import get_storage_by_url
        self._vector_storage = get_storage_by_url(url, required="vector")
        return self

    def build(self) -> Agent:
        if not self._model_instance:
            raise ValueError("Model is required. Call with_model first.")

        return Agent(
            model=self._model_instance,
            tools=self._tools,
            session=self._session,
            event_bus=self._event_bus,
            storage=self._storage,
            vector_storage=self._vector_storage,  # [新增] 注入向量存储
            **self._kwargs
        )