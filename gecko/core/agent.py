# gecko/core/agent.py  
from __future__ import annotations  
  
from typing import Any, Iterable, List, Optional, Type, Union  
  
from pydantic import BaseModel  
  
from gecko.core.events import AgentRunEvent, EventBus  
from gecko.core.message import Message  
from gecko.core.output import AgentOutput  
from gecko.core.toolbox import ToolBox  
from gecko.core.memory import TokenMemory  
from gecko.core.engine.base import CognitiveEngine  
from gecko.core.engine.react import ReActEngine  
from gecko.core.logging import get_logger  
from gecko.core.exceptions import AgentError  
  
logger = get_logger(__name__)  
  
  
class Agent:  
    """  
    Agent 对象负责在模型、工具箱、记忆之间协调一次推理任务。  
    """  
  
    def __init__(  
        self,  
        model: Any,  
        toolbox: ToolBox,  
        memory: TokenMemory,  
        engine_cls: Type[CognitiveEngine] = ReActEngine,  
        event_bus: Optional[EventBus] = None,  
        **engine_kwargs: Any,  
    ):  
        self.event_bus = event_bus or EventBus()  
        self.toolbox = toolbox  
        self.memory = memory  
        self.engine = engine_cls(  
            model=model,  
            toolbox=toolbox,  
            memory=memory,  
            **engine_kwargs  
        )  
  
    async def run(  
        self,  
        messages: str | Message | List[Message] | List[dict] | dict,  
        response_model: Optional[Type[BaseModel]] = None  
    ) -> AgentOutput | BaseModel:  
        """  
        单次推理入口：对多种输入格式统一转换为 Message 列表  
        """  
        input_msgs = self._normalize_messages(messages)  
  
        await self.event_bus.publish(  
            AgentRunEvent(type="run_started", data={"input_count": len(input_msgs)})  
        )  
  
        try:  
            output = await self.engine.step(input_msgs, response_model=response_model)  
            payload = self._serialize_output(output)  
  
            await self.event_bus.publish(  
                AgentRunEvent(type="run_completed", data={"output": payload})  
            )  
            return output  
  
        except Exception as e:  
            logger.exception("Agent run failed")  
            await self.event_bus.publish(  
                AgentRunEvent(type="run_error", error=str(e))  
            )  
            raise  
  
    async def stream(self, messages: str | Message | List[Message] | List[dict] | dict):  
        """  
        流式推理：共用同一套输入标准化逻辑  
        """  
        input_msgs = self._normalize_messages(messages)  
  
        await self.event_bus.publish(AgentRunEvent(type="stream_started"))  
        try:  
            async for chunk in self.engine.step_stream(input_msgs):   # type: ignore
                yield chunk  
            await self.event_bus.publish(AgentRunEvent(type="stream_completed"))  
        except Exception as e:  
            logger.exception("Agent stream failed")  
            await self.event_bus.publish(AgentRunEvent(type="stream_error", error=str(e)))  
            raise  
  
    # ---------------- 辅助方法 ----------------  
    def _normalize_messages(  
        self,  
        messages: str | Message | List[Message] | List[dict] | dict  
    ) -> List[Message]:  
        """  
        支持以下输入：  
        1. 字符串 -> 单条 user 消息  
        2. Message -> [Message]  
        3. List[Message] -> 原样返回  
        4. dict -> 若包含 role/content 则构建 Message，否则视为 {"input": "..."}  
        5. List[dict] -> 每个 dict 转为 Message  
        """  
        if isinstance(messages, Message):  
            return [messages]  
  
        if isinstance(messages, str):  
            return [Message.user(messages)]  
  
        if isinstance(messages, dict):  
            if "role" in messages:  
                return [Message(**messages)]  
            text = messages.get("input") or str(messages)  
            return [Message.user(text)]  
  
        if isinstance(messages, list):  
            if not messages:  
                raise AgentError("消息列表为空")  
            if isinstance(messages[0], Message):  
                return messages  # type: ignore # 已经是标准 Message  
            normalized = []  
            for item in messages:  
                if isinstance(item, Message):  
                    normalized.append(item)  
                elif isinstance(item, dict):  
                    normalized.append(Message(**item))  
                else:  
                    raise AgentError(f"无法识别的消息元素类型: {type(item)}")  
            return normalized  
  
        raise AgentError(f"不支持的消息类型: {type(messages)}")  
  
    def _serialize_output(self, output: AgentOutput | BaseModel) -> dict:  
        if hasattr(output, "model_dump"):  
            return output.model_dump()  
        return {"content": str(output)}  
