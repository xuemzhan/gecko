# examples/fast_dev_demo.py
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Fix] 使用标准的类导入，而非旧版 helper 函数
from gecko.plugins.models import ZhipuChat
# [Fix] 使用工厂方法创建存储
from gecko.plugins.storage.factory import create_storage

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Skipping: ZHIPU_API_KEY not set")
        return

    # [Fix] 使用 factory 创建并初始化存储
    # 注意：SQLite URL 相对路径需要 3 个斜杠 (sqlite:///./path)
    # 验证 Bug #7 (SQLite 注册) 是否修复，工厂应能正确加载 sqlite
    storage = await create_storage("sqlite:///./dev_sessions.db")

    agent = (AgentBuilder()
             # [Fix] 显式实例化模型
             .with_model(ZhipuChat(api_key=api_key, model="glm-4-flash", temperature=0.3))
             .with_session_id("user_123")
             .with_storage(storage) # type: ignore
             .build())

    # 第一次运行
    print("--- Round 1 ---")
    output1 = await agent.run([Message(role="user", content="我叫张三，以后叫我老张")])
    print("AI:", output1.content) # type: ignore

    # 第二次运行：验证记忆恢复
    print("\n--- Round 2 ---")
    output2 = await agent.run([Message(role="user", content="我叫什么名字？")])
    print("AI:", output2.content) # type: ignore
    
    # 清理
    await storage.shutdown()
    if os.path.exists("dev_sessions.db"):
        try:
            os.remove("dev_sessions.db")
            # 清理可能存在的锁文件和 WAL 文件
            if os.path.exists("dev_sessions.db.lock"): os.remove("dev_sessions.db.lock")
            if os.path.exists("dev_sessions.db-wal"): os.remove("dev_sessions.db-wal")
            if os.path.exists("dev_sessions.db-shm"): os.remove("dev_sessions.db-shm")
        except OSError:
            pass

if __name__ == "__main__":
    asyncio.run(main())