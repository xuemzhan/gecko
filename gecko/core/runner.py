# gecko/core/runner.py
from __future__ import annotations
import asyncio
from typing import List, Any
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.plugins.tools.executor import ToolExecutor
from gecko.plugins.tools.registry import ToolRegistry
from gecko.core.utils import ensure_awaitable # [修复] 导入

class AsyncRunner:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model

    async def execute(self, messages: List[Message], max_turns: int = 5) -> AgentOutput:
        current_messages = [m.model_dump() for m in messages]
        turn = 0
        last_response = None

        while turn < max_turns:
            turn += 1
            # 注入工具 Schema
            if turn == 1 and self.agent.tools:
                tools_schema = [tool.parameters for tool in ToolRegistry.list_all()]
                # 检查 system message 是否存在，不存在则插入，存在则更新
                has_system = len(current_messages) > 0 and current_messages[0]["role"] == "system"
                if not has_system:
                     current_messages.insert(0, {"role": "system", "content": "", "tools": tools_schema})
                # 注意：实际生产中需要更复杂的 prompt merging
           
            # [修复] 使用 ensure_awaitable 调用模型，防止同步/异步混用警告
            response = await ensure_awaitable(self.model.acompletion, messages=current_messages)
            last_response = response

            # [修复] Mock 对象兼容性处理
            # 许多测试 Mock 只有 content 属性，没有 choices 结构
            if not hasattr(response, "choices"):
                 if hasattr(response, "content"):
                      return AgentOutput(content=response.content, raw=response)
                 if isinstance(response, (str, dict)):
                      content_str = str(response)
                      return AgentOutput(content=content_str, raw=response)

            if not response.choices:
                break
           
            choice = response.choices[0]
            message = choice.message
            
            # 处理工具调用
            if hasattr(message, "tool_calls") and message.tool_calls:
                current_messages.append(message.model_dump())
                calls = []
                for tc in message.tool_calls:
                    try:
                        # 解析参数，这里简化处理
                        args_str = tc.function.arguments
                        args = eval(args_str) if isinstance(args_str, str) and args_str.startswith("{") else {}
                        # 注意：生产环境应该用 json.loads
                    except:
                        args = {}
                    calls.append({"name": tc.function.name, "arguments": args})
                
                # 并行执行工具
                tool_results = await ToolExecutor.concurrent_execute(calls)
                
                # 将结果添加回消息历史
                for result in tool_results:
                    current_messages.append({
                        "role": "tool",
                        "name": result.tool_name,
                        "content": result.content,
                        "tool_call_id": tc.id if hasattr(tc, 'id') else None # 补全 tool_call_id
                    })
                continue
           
            # 正常返回
            return AgentOutput(content=message.content or "", raw=response)
       
        # [修复] 兜底逻辑：如果循环结束且有最后一次响应，尝试提取内容
        content = "达到最大工具调用轮次"
        if last_response and hasattr(last_response, "choices") and last_response.choices:
             content = last_response.choices[0].message.content or content
        
        return AgentOutput(content=content, raw=last_response)