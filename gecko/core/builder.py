# gecko/core/builder.py
# ... imports
from typing import Any, List
from gecko.core.agent import Agent
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.session import Session
from gecko.core.engine.react import ReActEngine # 默认引擎

class AgentBuilder:
    def __init__(self):
        self._model = None
        self._tools = []
        self._session_id = "default"
        self._storage = None
        self._max_tokens = 4000
        self._system_prompt = None
        # ... 其他

    def with_model(self, model):
        self._model = model
        return self

    def with_tools(self, tools: List[Any]):
        self._tools.extend(tools)
        return self
        
    def with_storage(self, storage):
        self._storage = storage
        return self
        
    def with_session_id(self, session_id: str):
        self._session_id = session_id
        return self

    def with_system_prompt(self, prompt: str):
        self._system_prompt = prompt
        return self

    def build(self) -> Agent:
        if not self._model:
            raise ValueError("Model is required")

        # 1. 构建组件
        toolbox = ToolBox(self._tools)
        
        memory = TokenMemory(
            session_id=self._session_id,
            storage=self._storage,
            max_tokens=self._max_tokens
        )
        
        # 2. 组装 Agent
        return Agent(
            model=self._model,
            toolbox=toolbox,
            memory=memory,
            engine_cls=ReActEngine,
            system_prompt=self._system_prompt
            # 可传递其他 kwargs 给 engine
        )