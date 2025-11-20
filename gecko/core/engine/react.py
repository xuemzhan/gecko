# gecko/core/engine/react.py
from __future__ import annotations
import json
from typing import List, Dict, Any

from gecko.core.engine.base import CognitiveEngine
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable
from gecko.core.prompt import PromptTemplate

# 默认 Prompt 模板
DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Available Tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

Answer the user's request. Use tools if necessary.
"""

class ReActEngine(CognitiveEngine):
    """
    标准的 Reasoning + Acting 循环引擎
    """
    def __init__(self, model, toolbox, memory, max_turns: int = 5, system_prompt: str | PromptTemplate = None):
        super().__init__(model, toolbox, memory)
        self.max_turns = max_turns
        
        # 初始化 Prompt
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

    async def step(self, input_messages: List[Message]) -> AgentOutput:
        # 1. 准备上下文 (Context Construction)
        # 加载历史并修剪
        if self.memory.storage and self.memory.session_id:
            raw_data = await self.memory.storage.get(self.memory.session_id)
            history_raw = raw_data.get("messages", []) if raw_data else []
            history = await self.memory.get_history(history_raw)
        else:
            history = []
            
        # 2. 构造 System Prompt
        # 如果历史里没有 System Prompt，或者是第一轮，则生成一个新的
        has_system = any(m.role == "system" for m in history)
        if not has_system:
            tools_schema = self.toolbox.to_openai_schema()
            sys_content = self.prompt_template.format(tools=tools_schema)
            history.insert(0, Message(role="system", content=sys_content))

        # 合并当前输入
        current_context = history + input_messages
        
        # 3. 进入执行循环
        turn = 0
        final_response = None
        
        # 临时保存本轮产生的中间消息（用于回写记忆）
        new_generated_messages = [] 

        while turn < self.max_turns:
            turn += 1
            
            # 调用 LLM
            # 注意：LiteLLM 接收 dict 列表
            messages_payload = [m.model_dump() for m in current_context]
            
            # 注入 tools 定义 (OpenAI 格式)
            tools_schema = self.toolbox.to_openai_schema()
            kwargs = {}
            if tools_schema:
                kwargs["tools"] = tools_schema
                kwargs["tool_choice"] = "auto"

            response = await ensure_awaitable(
                self.model.acompletion, 
                messages=messages_payload,
                **kwargs
            )
            
            # 解析响应
            choice = response.choices[0]
            msg_data = choice.message.model_dump()
            response_message = Message(**msg_data)
            
            # 追加到上下文
            current_context.append(response_message)
            new_generated_messages.append(response_message)

            # 检查是否有工具调用
            if response_message.tool_calls:
                # 执行工具
                for tc in response_message.tool_calls:
                    func_name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    call_id = tc["id"]
                    
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        # 使用 ToolBox 执行
                        content = await self.toolbox.execute(func_name, args)
                        is_error = False
                    except Exception as e:
                        content = f"Error executing {func_name}: {str(e)}"
                        is_error = True

                    # 构造 Tool Message
                    tool_msg = Message(
                        role="tool",
                        content=str(content),
                        tool_call_id=call_id,
                        name=func_name
                    )
                    current_context.append(tool_msg)
                    new_generated_messages.append(tool_msg)
                
                # 工具执行完，继续下一轮循环（让 LLM看结果）
                continue
            
            # 没有工具调用，结束
            final_response = AgentOutput(
                content=response_message.content or "",
                raw=response
            )
            break
        
        if not final_response:
            final_response = AgentOutput(content="Max turns reached.")

        # 4. 保存记忆 (Save Memory)
        # 将新产生的对话（用户输入 + 中间思考 + 最终结果）追加存档
        if self.memory.storage and self.memory.session_id:
            # 这里简单处理：读取旧的 + 新的 = 全量保存
            # 实际生产中可能会做更细粒度的 append
            full_to_save = [m.model_dump() for m in current_context]
            await self.memory.storage.set(
                self.memory.session_id, 
                {"messages": full_to_save}
            )

        return final_response