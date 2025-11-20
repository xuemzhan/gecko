# examples/fast_dev_demo.py
import asyncio
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air
from gecko.plugins.storage.sqlite import SQLiteSessionStorage # [新增] 显式引入存储后端

async def main():
    # [修改] 初始化存储实例 (不再只传 URL 字符串)
    storage = SQLiteSessionStorage("sqlite://./dev_sessions.db")

    agent = (AgentBuilder()
             .with_model(glm_4_5_air(temperature=0.3))
             .with_session_id("user_123")              # [新增] 指定会话 ID
             .with_storage(storage)                    # [修改] 注入存储实例
             # .with_tools([KnowledgeTool(...)])       # [说明] Vector RAG 现在建议作为 Tool 注入
             .build())

    # 第一次运行：记住名字
    print("--- Round 1 ---")
    output1 = await agent.run([Message(role="user", content="我叫张三，以后叫我老张")])
    print("AI:", output1.content)

    # 第二次运行：验证记忆恢复 (TokenMemory 会自动从 SQLite 加载历史)
    print("\n--- Round 2 ---")
    output2 = await agent.run([Message(role="user", content="我叫什么名字？")])
    print("AI:", output2.content)

if __name__ == "__main__":
    asyncio.run(main())