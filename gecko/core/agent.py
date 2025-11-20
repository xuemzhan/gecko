# gecko/core/agent.py
from __future__ import annotations
from typing import Any, List, Optional, Type, Dict, AsyncIterator, Union
from pydantic import BaseModel

from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.events import EventBus, BaseEvent  # [修改] 引入通用事件基类
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.react import ReActEngine

# 定义具体的事件类型 (建议放在 agent.py 或单独的 events 定义文件中)
class AgentRunEvent(BaseEvent):
    type: str = "agent_run" # 默认值，实际使用时覆盖
    
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
        messages: str | List[Message] | Dict,
        response_model: Optional[Type[BaseModel]] = None
    ) -> AgentOutput | BaseModel:
        
        # 1. 标准化输入
        if isinstance(messages, str):
            input_msgs = [Message(role="user", content=messages)]
        elif isinstance(messages, dict):
             content = messages.get("input", str(messages))
             input_msgs = [Message(role="user", content=content)]
        elif isinstance(messages, list):
            input_msgs = messages
        else:
            raise ValueError("Invalid input type")

        # [修改] 发布事件：使用新版 EventBus API
        # wait=False 表示非阻塞发布（除非这对业务流至关重要，否则建议 False）
        await self.event_bus.publish(
            AgentRunEvent(type="run_started", data={"input": [m.to_api_payload() for m in input_msgs]})
        )

        try:
            output = await self.engine.step(input_msgs, response_model=response_model)
            
            # 准备事件数据
            evt_data = output.model_dump() if isinstance(output, BaseModel) else output
            
            await self.event_bus.publish(
                AgentRunEvent(type="run_completed", data={"output": evt_data})
            )
            return output
            
        except Exception as e:
            await self.event_bus.publish(
                AgentRunEvent(type="run_error", error=str(e), data={"input": str(input_msgs)})
            )
            raise e

    async def stream(self, messages: str | List[Message]) -> AsyncIterator[str]:
        if isinstance(messages, str):
            input_msgs = [Message(role="user", content=messages)]
        else:
            input_msgs = messages
            
        # 流式开始事件
        await self.event_bus.publish(AgentRunEvent(type="stream_started"))
        
        try:
            async for token in self.engine.step_stream(input_msgs):
                yield token
                # 可选：发布 token 事件（高频事件，慎用 wait=True）
                # await self.event_bus.publish(AgentRunEvent(type="stream_chunk", data={"chunk": token}))
            
            await self.event_bus.publish(AgentRunEvent(type="stream_completed"))
            
        except Exception as e:
            await self.event_bus.publish(AgentRunEvent(type="stream_error", error=str(e)))
            raise e