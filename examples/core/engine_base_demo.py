import asyncio
import time
from typing import Any, AsyncIterator, List, Type, Dict

from pydantic import BaseModel, Field

# Gecko 核心组件导入
from gecko.core.engine.base import CognitiveEngine, AgentOutput
from gecko.core.message import Message
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox
from gecko.core.protocols import (
    ModelProtocol, 
    CompletionResponse, 
    CompletionChoice, 
    StreamChunk
)

# ==========================================
# 1. 模拟组件 (Mock Components)
# ==========================================

class MockModel(ModelProtocol):
    """
    一个简单的 Mock 模型，实现了 ModelProtocol。
    它只是回显用户的输入，或者生成预定义的流式数据。
    """
    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs) -> CompletionResponse:
        # 模拟网络延迟
        await asyncio.sleep(0.1)
        
        last_content = messages[-1]["content"]
        response_text = f"Mock Response to: {last_content}"
        
        return CompletionResponse(
            choices=[
                CompletionChoice(message={"role": "assistant", "content": response_text})
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} # type: ignore
        )

    async def astream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[StreamChunk]:
        # 模拟流式输出
        full_text = "This is a streaming response from the mock model."
        for word in full_text.split():
            await asyncio.sleep(0.05)
            yield StreamChunk(
                choices=[{"delta": {"content": word + " "}}]
            )

    def count_tokens(self, text_or_messages) -> int:
        # 简单模拟：按字符数估算，不阻塞主线程
        if isinstance(text_or_messages, list):
            text = "".join(str(m.get("content", "")) for m in text_or_messages)
        else:
            text = str(text_or_messages)
        return len(text) // 4

# ==========================================
# 2. 自定义引擎实现 (Custom Engine)
# ==========================================

class DemoEngine(CognitiveEngine):
    """
    继承 CognitiveEngine 的演示引擎。
    必须实现 step() 方法。
    """
    
    async def step(self, input_messages: List[Message], **kwargs) -> AgentOutput:
        """
        实现核心推理逻辑（带统计修复）
        """
        # ⏱️ 1. 开始计时
        start_time = time.time()
        
        # 2. 验证输入
        self.validate_input(input_messages)
        
        # 3. 触发 before_step hook
        await self.before_step(input_messages, **kwargs)

        # 4. 准备数据
        formatted_msgs = [m.to_openai_format() for m in input_messages]
        
        try:
            # 5. 调用模型
            response = await self.model.acompletion(formatted_msgs)
            content = response.choices[0].message["content"]
            
            # 获取 token 使用量 (MockModel 返回了 usage)
            # 如果 response.usage 是对象则取属性，如果是字典则取键值
            usage_info = response.usage
            total_tokens = 0
            if isinstance(usage_info, dict):
                total_tokens = usage_info.get("total_tokens", 0)
            elif hasattr(usage_info, "total_tokens"):
                total_tokens = usage_info.total_tokens # type: ignore

            # 6. 构建输出
            output = AgentOutput(
                content=content,
                metadata={"finish_reason": "stop"}
            )
            
            # 7. 触发 after_step hook
            await self.after_step(input_messages, output, **kwargs)
            
            # ✅ 8. [修复点] 手动更新统计信息
            duration = time.time() - start_time
            if self.stats:
                self.stats.add_step(duration, tokens=total_tokens)
            
            return output
            
        except Exception as e:
            # ✅ 9. [修复点] 记录错误统计
            if self.stats:
                self.stats.errors += 1
                
            # 错误处理 hook
            await self.on_error(e, input_messages)
            raise

    async def step_stream(self, input_messages: List[Message], **kwargs) -> AsyncIterator[str]: # type: ignore
        """
        覆盖流式推理方法
        """
        formatted_msgs = [m.to_openai_format() for m in input_messages]
        
        async for chunk in self.model.astream(formatted_msgs): # type: ignore
            content = chunk.content
            if content:
                yield content

    async def step_structured(
        self, 
        input_messages: List[Message], 
        response_model: Type[BaseModel], 
        **kwargs
    ) -> BaseModel:
        """
        覆盖结构化输出方法 (模拟实现)
        """
        # 模拟：直接返回一个伪造的结构化对象
        # 实际场景中这里会调用 StructureEngine
        print(f"   [Engine] Parsing structured output for {response_model.__name__}...")
        await asyncio.sleep(0.1)
        
        return response_model(
            reasoning="Simulated reasoning",
            score=95,
            tags=["demo", "mock"]
        )

# ==========================================
# 3. 辅助数据结构
# ==========================================

class AnalysisResult(BaseModel):
    """用于测试结构化输出的模型"""
    reasoning: str = Field(description="思考过程")
    score: int = Field(description="评分")
    tags: List[str] = Field(description="标签")

# ==========================================
# 4. 主演示流程
# ==========================================

async def main():
    print("🚀 Starting Engine Base Demo...\n")

    # --- 初始化依赖 ---
    model = MockModel()
    toolbox = ToolBox() # 空工具箱
    # 注意：这里简单 mock memory，实际应传入 SessionInterface 实现
    memory = TokenMemory(session_id="demo_session", max_tokens=1000)

    # --- 使用上下文管理器初始化引擎 ---
    print("1️⃣  Testing Context Manager & Basic Step")
    async with DemoEngine(model, toolbox, memory) as engine:
        
        # --- 设置 Hooks ---
        async def my_before_hook(messages, **kwargs):
            print(f"   [Hook] Before step: Processing {len(messages)} messages")

        async def my_after_hook(messages, output, **kwargs):
            print(f"   [Hook] After step: Generated {len(output.content)} chars")

        engine.before_step_hook = my_before_hook
        engine.after_step_hook = my_after_hook

        # --- 测试普通推理 ---
        user_msg = Message.user("Hello Gecko!")
        print(f"   User: {user_msg.content}")
        
        output = await engine.step([user_msg])
        print(f"   Agent: {output.content}\n")

        # --- 测试流式推理 ---
        print("2️⃣  Testing Streaming")
        print("   Agent (Stream): ", end="", flush=True)
        async for token in engine.step_stream([Message.user("Stream me!")]):
            print(token, end="", flush=True)
        print("\n")

        # --- 测试结构化输出 ---
        print("3️⃣  Testing Structured Output")
        result = await engine.step_structured(
            [Message.user("Analyze this")], 
            response_model=AnalysisResult
        )
        print(f"   Result: {result.model_dump_json()}\n")

        # --- 查看统计 ---
        print("4️⃣  Execution Stats")
        stats = engine.get_stats()
        print(f"   Total Steps: {stats['total_steps']}") # type: ignore
        print(f"   Total Time:  {stats['total_time']:.4f}s") # type: ignore
        print(f"   Avg Time:    {stats['avg_step_time']:.4f}s") # type: ignore
        print(f"   Errors:      {stats['errors']}") # type: ignore

if __name__ == "__main__":
    asyncio.run(main())