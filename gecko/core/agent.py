# gecko/core/agent.py
from __future__ import annotations
from typing import Any, List, Optional, Type, Dict, AsyncIterator, Union
from pydantic import BaseModel

from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.events import EventBus, BaseEvent
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.react import ReActEngine

class AgentRunEvent(BaseEvent):
    type: str = "agent_run"
    
class Agent:
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
        self.toolbox = toolbox
        self.memory = memory
        self.engine = engine_cls(model=model, toolbox=toolbox, memory=memory, **engine_kwargs)

    async def run(
        self, 
        messages: str | List[Message] | List[Dict] | Dict,
        response_model: Optional[Type[BaseModel]] = None
    ) -> AgentOutput | BaseModel:
        
        # [优化] 1. 标准化输入：优先检查是否已经是 Message 列表，减少开销
        if isinstance(messages, list) and messages and isinstance(messages[0], Message):
            input_msgs = messages
        elif isinstance(messages, str):
            input_msgs = [Message(role="user", content=messages)]
        elif isinstance(messages, dict):
             content = messages.get("input", str(messages))
             input_msgs = [Message(role="user", content=content)]
        elif isinstance(messages, list):
            # 处理 List[Dict]
            input_msgs = [Message(**m) if isinstance(m, dict) else m for m in messages]
        else:
            raise ValueError(f"Invalid input type: {type(messages)}")

        # 发布事件
        await self.event_bus.publish(
            AgentRunEvent(type="run_started", data={"input_count": len(input_msgs)})
        )

        try:
            output = await self.engine.step(input_msgs, response_model=response_model)
            
            evt_data = output.model_dump() if isinstance(output, BaseModel) else output
            await self.event_bus.publish(
                AgentRunEvent(type="run_completed", data={"output": evt_data})
            )
            return output
            
        except Exception as e:
            await self.event_bus.publish(
                AgentRunEvent(type="run_error", error=str(e))
            )
            raise e

    async def stream(self, messages: str | List[Message]) -> AsyncIterator[str]:
        # 同样的输入优化
        if isinstance(messages, list) and messages and isinstance(messages[0], Message):
            input_msgs = messages
        elif isinstance(messages, str):
            input_msgs = [Message(role="user", content=messages)]
        else:
             input_msgs = messages # Fallback
            
        await self.event_bus.publish(AgentRunEvent(type="stream_started"))
        
        try:
            async for token in self.engine.step_stream(input_msgs):
                yield token
            
            await self.event_bus.publish(AgentRunEvent(type="stream_completed"))
            
        except Exception as e:
            await self.event_bus.publish(AgentRunEvent(type="stream_error", error=str(e)))
            raise e