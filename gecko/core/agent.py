# gecko/core/agent.py  
from __future__ import annotations  
  
from typing import Any, AsyncIterator, Iterable, List, Optional, Type, Union  
  
from pydantic import BaseModel  
  
from gecko.core.events import AgentRunEvent, EventBus  
from gecko.core.events.types import AgentStreamEvent
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
            event_bus=self.event_bus,
            memory=memory,  
            **engine_kwargs
        )  
  
    # 修复缩进，使其成为类方法而不是 __init__ 的内部函数
    async def run(
        self,
        messages: str | Message | List[Message] | List[dict] | dict,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> AgentOutput | BaseModel:
        """
        单次推理入口：对多种输入格式统一转换为 Message 列表
        """
        input_msgs = self._normalize_messages(messages)

        # 新增：拼接用户可读文本，供 Guardrails / 日志使用
        raw_text = "\n".join(m.get_text_content() for m in input_msgs)

        await self.event_bus.publish(
            AgentRunEvent(
                type="run_started",
                data={
                    "input": raw_text,
                    "input_count": len(input_msgs),
                },
            )
        )

        try:
            output = await self.engine.step(
                input_msgs, 
                response_model=response_model,
                **kwargs)
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

    async def stream(
        self, 
        messages: str | Message | List[Message] | List[dict] | dict
    ) -> AsyncIterator[str]:
        """
        流式推理（简易模式）：仅返回文本 Token。
        
        [修复] 适配 Engine 的 AgentStreamEvent 输出。
        """
        input_msgs = self._normalize_messages(messages)
        raw_text = "\n".join(m.get_text_content() for m in input_msgs)

        await self.event_bus.publish(
            AgentRunEvent(
                type="stream_started",
                data={"input": raw_text, "input_count": len(input_msgs)},
            )
        )
        
        try:
            # 迭代 Engine 产生的高级事件
            async for event in self.engine.step_stream(input_msgs): # type: ignore
                # 1. 如果是 Token，直接 yield 给用户
                if event.type == "token":
                    yield event.content
                
                # 2. 如果是 Error，记录日志 (是否抛出取决于策略，这里选择记录不中断流，除非是致命错误)
                elif event.type == "error":
                    logger.warning(f"Stream error event: {event.content}")
                    # 可选：yield f"[Error: {event.content}]"
                
                # 3. 其他事件 (tool_input, result) 可以通过 EventBus 转发，
                #    或者在这里忽略，仅供 stream_events 方法使用。
            
            await self.event_bus.publish(AgentRunEvent(type="stream_completed"))
            
        except Exception as e:
            logger.exception("Agent stream failed")
            await self.event_bus.publish(AgentRunEvent(type="stream_error", error=str(e)))
            raise

    async def stream_events(
        self, 
        messages: str | Message | List[Message] | List[dict] | dict
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        [新增] 高级流式接口：返回完整的事件流 (Token, Tools, Result)。
        适用于需要展示“正在思考”、“正在调用工具”等状态的 UI。
        """
        input_msgs = self._normalize_messages(messages)
        
        try:
            async for event in self.engine.step_stream(input_msgs): # type: ignore
                yield event
        except Exception as e:
            # 即使发生异常，也尝试 yield 一个 error 事件给前端
            yield AgentStreamEvent(type="error", content=str(e))
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
