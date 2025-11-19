# examples/fast_dev_demo.py
import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air  # 任意模型

async def main():
    agent = (AgentBuilder()
             .with_model(glm_4_5_air(temperature=0.3))
             .with_session_storage_url("sqlite://./dev_sessions.db")   # Session 持久化
             .with_vector_storage_url("lancedb://./dev_vector_db")     # Vector RAG
             .build())

    # 第一次运行：记住名字
    output1 = await agent.run([Message(role="user", content="我叫张三，以后叫我老张")])
    print("第一次：", output1.content)

    # 第二次运行：验证记忆恢复
    output2 = await agent.run([Message(role="user", content="我叫什么名字？")])
    print("第二次（应叫老张）：", output2.content)

if __name__ == "__main__":
    asyncio.run(main())