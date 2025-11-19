# gecko/core/builder.py
from typing import Any, Dict, List, Optional, Self

from gecko.core.agent import Agent
from gecko.core.session import Session
from gecko.core.events import EventBus
from gecko.core.runner import AsyncRunner
from gecko.core.protocols import ModelProtocol
from gecko.plugins.models.litellm import LiteLLMModel

class AgentBuilder:
    def __init__(self):
        self._model_instance: ModelProtocol | None = None
        self._tools: List[Any] = []
        self._session: Optional[Session] = None
        self._event_bus: Optional[EventBus] = None
        self._kwargs: Dict[str, Any] = {}

    # def with_model(
    #     self,
    #     model: str | dict,
    #     base_url: str | None = None,
    #     api_key: str | None = None,
    #     **kwargs
    # ) -> Self:
    #     if isinstance(model, dict):
    #         config = model
    #     else:
    #         config = {"model": model, "base_url": base_url, "api_key": api_key, **kwargs}
    #     self._model_instance = LiteLLMModel(**config)
    #     return self
    def with_model(self, model_instance: ModelProtocol) -> Self:
        """
        统一入口：接收任何实现了 acompletion 方法的模型实例
        支持：
          - LiteLLMModel(**config)
          - glm_4_5_air()
          - 未来所有自定义模型
        """
        if not hasattr(model_instance, "acompletion"):
            raise ValueError("Model must implement async acompletion(messages, **kwargs) method")
        if not callable(getattr(model_instance, "acompletion")):
            raise ValueError("acompletion must be callable")
        
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