# gecko/core/builder.py
from typing import Any, Dict, List, Optional, Self

from gecko.core.agent import Agent
from gecko.core.session import Session
from gecko.core.events import EventBus
from gecko.core.runner import AsyncRunner
from gecko.plugins.models.litellm import LiteLLMModel

class AgentBuilder:
    def __init__(self):
        self._model_instance: Optional[LiteLLMModel] = None
        self._tools: List[Any] = []
        self._session: Optional[Session] = None
        self._event_bus: Optional[EventBus] = None
        self._kwargs: Dict[str, Any] = {}

    def with_model(
        self,
        model: str | dict,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs
    ) -> Self:
        if isinstance(model, dict):
            config = model
        else:
            config = {"model": model, "base_url": base_url, "api_key": api_key, **kwargs}
        self._model_instance = LiteLLMModel(**config)
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

    def build(self) -> Agent:
        if not self._model_instance:
            raise ValueError("Model is required. Call .with_model(...) first.")

        return Agent(
            model=self._model_instance,
            tools=self._tools,
            session=self._session,
            event_bus=self._event_bus,
            # Runner 由 Agent 内部创建，Builder 不干预
            **self._kwargs
        )