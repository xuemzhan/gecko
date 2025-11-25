# examples/fast_dev_demo.py
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Fix] Import ZhipuChat
from gecko.plugins.models.presets.zhipu import ZhipuChat
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    
    # [Fix] 创建 Storage 实例
    storage = SQLiteStorage("sqlite://./dev_sessions.db")
    await storage.initialize() # 确保存储初始化

    try:
        # [Fix] 创建 Model 实例
        model = ZhipuChat(api_key=api_key or "mock_key", model="glm-4-flash")

        agent = (AgentBuilder()
                 .with_model(model)
                 .with_session_id("user_123")
                 .with_storage(storage)
                 .build())

        # 第一次运行：记住名字
        print("--- Round 1 ---")
        output1 = await agent.run([Message(role="user", content="我叫张三，以后叫我老张")])
        print("AI:", output1.content) # type: ignore

        # 第二次运行：验证记忆恢复
        print("\n--- Round 2 ---")
        output2 = await agent.run([Message(role="user", content="我叫什么名字？")])
        print("AI:", output2.content) # type: ignore
        
    finally:
        await storage.shutdown()
        # 清理临时文件
        if os.path.exists("dev_sessions.db"):
            try: os.remove("dev_sessions.db")
            except: pass

if __name__ == "__main__":
    asyncio.run(main())