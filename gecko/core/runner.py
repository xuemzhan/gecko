# gecko/core/runner.py
from __future__ import annotations
import asyncio
from typing import List
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.plugins.tools.executor import ToolExecutor
from gecko.plugins.tools.registry import ToolRegistry
from gecko.core.events import RunEvent

class AsyncRunner:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model

    async def execute(self, messages: List[Message], max_turns: int = 5) -> AgentOutput:
        current_messages = [m.model_dump() for m in messages]
        turn = 0
        while turn < max_turns:
            turn += 1
            if turn == 1 and self.agent.tools:
                tools_schema = [tool.parameters for tool in ToolRegistry.list_all()]
                if current_messages and "tools" not in current_messages[0]:
                    current_messages.insert(0, {"role": "system", "content": "", "tools": tools_schema})
           
            # 关键修复：强制 await 模型调用，消除 RuntimeWarning
            if asyncio.iscoroutinefunction(self.model.acompletion):
                response = await self.model.acompletion(messages=current_messages)
            else:
                response = self.model.acompletion(messages=current_messages)  # 同步模型 fallback

            if not response.choices:
                break
           
            choice = response.choices[0]
            message = choice.message
            if message.tool_calls:
                current_messages.append(message.model_dump())
                calls = []
                for tc in message.tool_calls:
                    try:
                        args = eval(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except:
                        args = {}
                    calls.append({"name": tc.function.name, "arguments": args})
                tool_results = await ToolExecutor.concurrent_execute(calls)
                for result in tool_results:
                    current_messages.append({
                        "role": "tool",
                        "name": result.tool_name,
                        "content": result.content
                    })
                continue
           
            return AgentOutput(content=message.content or "", raw=response)
       
        return AgentOutput(content="达到最大工具调用轮次", raw=response)