# gecko/core/runner.py
from __future__ import annotations
import asyncio
from typing import List
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable
from gecko.plugins.tools.executor import ToolExecutor
from gecko.plugins.tools.registry import ToolRegistry

class AsyncRunner:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model

    async def execute(self, new_messages: List[Message], max_turns: int = 5) -> AgentOutput:
        # 1. 加载历史记忆 (Load History)
        history: List[Message] = []
        if self.agent.storage and self.agent.session.session_id:
            session_data = await self.agent.storage.get(self.agent.session.session_id)
            if session_data and "messages" in session_data:
                # 反序列化消息
                raw_msgs = session_data["messages"]
                # 实现滑动窗口 (Sliding Window): 只取最近 N 条
                window_size = self.agent.memory_window
                if len(raw_msgs) > window_size:
                    raw_msgs = raw_msgs[-window_size:]
                
                for m in raw_msgs:
                    try:
                        history.append(Message(**m))
                    except Exception:
                        pass # 忽略格式错误的历史消息
        
        # 2. 构建完整上下文 (Context Construction)
        # 历史 + 新输入
        context_messages = history + new_messages
        
        # 转换为模型可读格式 (Dict)
        current_messages_dicts = [m.model_dump() for m in context_messages]
        
        turn = 0
        final_response = None
        
        # 3. 执行 ReAct 循环
        while turn < max_turns:
            turn += 1
            
            # 注入 System Prompt 和 工具定义 (仅在第一轮)
            if turn == 1 and self.agent.tools:
                tools_schema = [tool.parameters for tool in ToolRegistry.list_all()] # 简化：实际应只列出 agent.tools
                # 这里为了简化，假设 tool_registry 包含了所需工具。
                # 严谨做法：从 self.agent.tools 中提取 schema
                
                has_system = len(current_messages_dicts) > 0 and current_messages_dicts[0]["role"] == "system"
                if not has_system:
                    # 简单的 System Prompt
                    current_messages_dicts.insert(0, {
                        "role": "system", 
                        "content": "You are a helpful assistant.",
                        # 部分模型通过 extra_body 传 tools，部分通过 messages，这里视模型实现而定
                        # Litellm 会自动处理 tools 参数如果传入 acompletion
                    })

            # 调用模型
            # 注意：LiteLLM 等适配器通常将 tools 作为 kwargs 传入，而不是 message content
            # 这里为了简化，我们假设 model.acompletion 处理了 tools 转换
            response = await ensure_awaitable(
                self.model.acompletion, 
                messages=current_messages_dicts
            )
            
            if not hasattr(response, "choices") or not response.choices:
                # Mock 或 异常响应处理
                final_content = getattr(response, "content", str(response))
                final_response = AgentOutput(content=final_content, raw=response)
                break

            choice = response.choices[0]
            response_message = choice.message
            
            # 追加模型回复到上下文
            current_messages_dicts.append(response_message.model_dump())

            # 处理工具调用
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                calls = []
                for tc in response_message.tool_calls:
                    try:
                        args_str = tc.function.arguments
                        import json
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except:
                        args = {}
                    calls.append({"name": tc.function.name, "arguments": args})
                
                # 执行工具
                tool_results = await ToolExecutor.concurrent_execute(calls)
                
                for result in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "name": result.tool_name,
                        "content": result.content,
                        "tool_call_id": tc.id if hasattr(tc, 'id') else "call_null"
                    }
                    current_messages_dicts.append(tool_msg)
                continue # 进入下一轮，让模型根据工具结果生成回答
            
            # 没有工具调用，生成最终结果
            final_response = AgentOutput(content=response_message.content or "", raw=response)
            break
        
        if not final_response:
             final_response = AgentOutput(content="Max turns reached without final response")

        # 4. 保存记忆 (Save History)
        if self.agent.storage and self.agent.session.session_id:
            # 保存最新的 current_messages_dicts
            # 注意：这里保存了 System Prompt 和 Tool Calls 过程
            # 生产环境可能需要清洗（比如去掉中间步骤，只留 User/Assistant）
            await self.agent.storage.set(
                self.agent.session.session_id, 
                {"messages": current_messages_dicts}
            )

        return final_response

    async def stream(self, messages):
        # 简单的流式透传，Day 3 完善
        yield AgentOutput(content="Streaming not fully implemented in Day 1")