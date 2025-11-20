# gecko/core/agent.py
from __future__ import annotations

from typing import Any, List, Optional, Type
from pydantic import BaseModel

from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.session import Session
from gecko.core.events import EventBus, AppEvent
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.react import ReActEngine
from gecko.plugins.storage.interfaces import SessionInterface

class Agent:
    """
    Gecko Agent (v2.0 Refactored)
    组件容器：组装 ToolBox, Memory, Engine
    """
    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        engine_cls: Type[CognitiveEngine] = ReActEngine,
        event_bus: Optional[EventBus] = None,
        **engine_kwargs
    ):
        self.event_bus = event_bus or EventBus()
        
        # 核心组件
        self.toolbox = toolbox
        self.memory = memory
        
        # 初始化引擎 (依赖注入)
        self.engine = engine_cls(
            model=model, 
            toolbox=toolbox, 
            memory=memory,
            **engine_kwargs
        )

    async def run(self, messages: str | List[Message] | Dict) -> AgentOutput:
        """
        统一执行入口
        """
        # 1. 标准化输入
        if isinstance(messages, str):
            input_msgs = [Message(role="user", content=messages)]
        elif isinstance(messages, dict):
             # 兼容 workflow 传入的 context dict
             content = messages.get("input", str(messages))
             input_msgs = [Message(role="user", content=content)]
        elif isinstance(messages, list):
            input_msgs = messages
        else:
            raise ValueError("Invalid input type")

        # 2. 触发事件
        await self.event_bus.publish(AppEvent(type="run_started", data={"input": input_msgs}))

        try:
            # 3. 委托给引擎执行
            output = await self.engine.step(input_msgs)
            
            await self.event_bus.publish(AppEvent(type="run_completed", data={"output": output}))
            return output
            
        except Exception as e:
            await self.event_bus.publish(AppEvent(type="run_error", error=str(e)))
            raise e
    
    # 增加生命周期管理
    async def startup(self):
        """启动所有工具资源"""
        # 未来遍历 toolbox 调用 tool.startup()
        pass

    async def shutdown(self):
        """释放资源"""
        # 未来遍历 toolbox 调用 tool.shutdown()
        pass