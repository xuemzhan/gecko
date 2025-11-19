from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pydantic import BaseModel as PydanticBaseModel

from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.runner import AsyncRunner
from gecko.core.session import Session
from gecko.core.events import AppEvent, EventBus
from gecko.core.exceptions import AgentError
from gecko.plugins.registry import model_registry, tool_registry, storage_registry

class BaseModel(PydanticBaseModel):
    """所有插件的基类，确保 Pydantic 验证"""

class Agent:
    """
    Gecko 核心 Agent 类：极简执行循环。
    - 无配置参数，所有通过 Builder 注入。
    - 负责消息处理、模型调用、工具执行、事件广播。
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        session: Optional[Session] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.tools = tools or []
        self.session = session or Session()
        self.event_bus = event_bus or EventBus()
        # 关键修复：Runner 只接收 agent 本身
        self.runner = AsyncRunner(self)   # ← 正确方式
        self.kwargs = kwargs
        

    async def run(self, messages: List[Message]) -> AgentOutput:
        try:
            await self.event_bus.publish(AppEvent(type="run_started", data={"messages": messages}))
            output = await self.runner.execute(messages)
            await self.event_bus.publish(AppEvent(type="run_completed", data={"output": output}))
            return output
        except Exception as e:
            error_msg = f"Agent run failed: {type(e).__name__}: {str(e)}"
            print(f"\n❌ Gecko Error: {error_msg}")  # 立即打印，永不吞噬
            await self.event_bus.publish(AppEvent(
                type="run_error",
                error=error_msg,           # 使用新字段
                data={"exception": str(e)}
            ))
            raise AgentError(error_msg) from e

    def sync_run(self, messages: List[Message]) -> AgentOutput:
        """同步包装（薄层）"""
        return asyncio.run(self.run(messages))

    async def stream(self, messages: List[Message]) -> AsyncIterator[AgentOutput]:
        """流式输出"""
        async for chunk in self.runner.stream(messages):
            yield chunk

    # 扩展方法：如 add_tool, 但鼓励用 Builder
    def add_tool(self, tool: BaseModel):
        self.tools.append(tool)
        
    