from typing import AsyncIterator
from gecko.core.message import Message
from gecko.core.output import AgentOutput

class AsyncRunner:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model  # 直接访问

    async def execute(self, messages: list[Message]) -> AgentOutput:
        response = await self.model.acompletion(  # 直接调用实例方法
            messages=[m.model_dump() for m in messages]
        )
        return AgentOutput(
            content=response.choices[0].message.content,
            raw=response
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[AgentOutput]:
        # 占位，真实流式稍后补
        yield await self.execute(messages)