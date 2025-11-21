# gecko/core/builder.py  
from __future__ import annotations  
  
from typing import Any, Sequence, Type  
  
from gecko.core.agent import Agent  
from gecko.core.memory import TokenMemory  
from gecko.core.toolbox import ToolBox  
from gecko.core.engine.base import CognitiveEngine  
from gecko.core.engine.react import ReActEngine  
from gecko.plugins.storage.interfaces import SessionInterface  
from gecko.plugins.tools.base import BaseTool  
from gecko.core.exceptions import ConfigurationError  
  
  
class AgentBuilder:  
    """  
    Agent 构建器（改进版）  
    关键改进：  
    1. system_prompt 等引擎参数统一通过 engine_kwargs 传递，避免与 Agent.__init__ 不匹配  
    2. 工具列表自动去重并校验是否继承 BaseTool  
    3. storage 必须实现 SessionInterface，否则在 TokenMemory 中使用会报错  
    4. 支持自定义 Engine 类 & 额外参数  
    """  
  
    def __init__(self):  
        self._model: Any | None = None  
        self._tools: list[BaseTool] = []  
        self._storage: SessionInterface | None = None  
        self._session_id: str = "default"  
        self._max_tokens: int = 4000  
        self._engine_cls: Type[CognitiveEngine] = ReActEngine  
        self._engine_kwargs: dict[str, Any] = {}  # 统一放置系统 Prompt、Hook 等  
        self._toolbox_config: dict[str, Any] = {}  
  
    # ---------------- 基础配置 ----------------  
    def with_model(self, model: Any) -> "AgentBuilder":  
        # 检查模型是否实现必要方法  
        missing = [m for m in ("acompletion",) if not hasattr(model, m)]  
        if missing:  
            raise ConfigurationError(  
                f"Model 缺少必要方法: {', '.join(missing)}",  
                context={"model": repr(model)}  
            )  
        self._model = model  
        return self  
  
    def with_tools(self, tools: Sequence[BaseTool]) -> "AgentBuilder":  
        for tool in tools:  
            if not isinstance(tool, BaseTool):  
                raise TypeError(f"Tool 必须继承 BaseTool，收到 {type(tool)}")  
            self._tools.append(tool)  
        return self  
  
    def with_storage(self, storage: SessionInterface | None) -> "AgentBuilder":  
        if storage and not isinstance(storage, SessionInterface):  
            raise TypeError(  
                "storage 必须实现 SessionInterface，用于 TokenMemory 持久化"  
            )  
        self._storage = storage  
        return self  
  
    def with_session_id(self, session_id: str) -> "AgentBuilder":  
        self._session_id = session_id  
        return self  
  
    def with_max_tokens(self, max_tokens: int) -> "AgentBuilder":  
        self._max_tokens = max_tokens  
        return self  
  
    def with_engine(  
        self,  
        engine_cls: Type[CognitiveEngine],  
        **engine_kwargs: Any  
    ) -> "AgentBuilder":  
        if not issubclass(engine_cls, CognitiveEngine):  
            raise TypeError("engine_cls 必须继承 CognitiveEngine")  
        self._engine_cls = engine_cls  
        self._engine_kwargs.update(engine_kwargs)  
        return self  
  
    def with_system_prompt(self, prompt: str) -> "AgentBuilder":  
        # 统一放入 engine_kwargs，确保 Engine 可接收  
        self._engine_kwargs["system_prompt"] = prompt  
        return self  
  
    def with_toolbox_config(self, **config: Any) -> "AgentBuilder":  
        """  
        允许调用者自定义 ToolBox 的并发/超时等参数  
        """  
        self._toolbox_config.update(config)  
        return self  
  
    # ---------------- 构建流程 ----------------  
    def build(self) -> Agent:  
        if not self._model:  
            raise ConfigurationError("构建 Agent 前必须调用 with_model 指定模型")  
  
        toolbox = self._build_toolbox()  
        memory = self._build_memory()  
  
        return Agent(  
            model=self._model,  
            toolbox=toolbox,  
            memory=memory,  
            engine_cls=self._engine_cls,  
            event_bus=self._engine_kwargs.pop("event_bus", None),  
            **self._engine_kwargs  # 其余参数直接传给 Engine  
        )  
  
    def _build_toolbox(self) -> ToolBox:  
        # 根据工具名称去重，后注册的同名工具会覆盖前者  
        deduped: dict[str, BaseTool] = {}  
        for tool in self._tools:  
            deduped[tool.name] = tool  
  
        return ToolBox(  
            tools=list(deduped.values()),  
            **self._toolbox_config  
        )  
  
    def _build_memory(self) -> TokenMemory:  
        return TokenMemory(  
            session_id=self._session_id,  
            storage=self._storage,  
            max_tokens=self._max_tokens  
        )  
