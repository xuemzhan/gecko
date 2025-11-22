# examples/redis_storage_demo.py
import asyncio
import os
from gecko.plugins.storage.factory import create_storage
from gecko.core.logging import setup_logging

# 需要运行真实的 Redis: docker run -p 6379:6379 redis
REDIS_URL = os.getenv("GECKO_REDIS_URL", "redis://localhost:6379/0")

async def main():
    setup_logging(level="INFO")
    
    print(f"🔌 Connecting to {REDIS_URL}...")
    
    try:
        # 使用工厂创建
        storage = await create_storage(REDIS_URL)
        print("✅ Storage initialized")
        
        session_id = "demo_user_007"
        
        # 写入
        print("💾 Writing data...")
        await storage.set(session_id, {
            "user": "Bond",
            "mission": "Secret",
            "active": True
        })
        
        # 读取
        print("📖 Reading data...")
        data = await storage.get(session_id)
        print(f"   Result: {data}")
        
        # 清理
        print("🧹 Deleting data...")
        await storage.delete(session_id)
        
        # 再次读取
        data = await storage.get(session_id)
        print(f"   After delete: {data}")
        
        await storage.shutdown()
        print("👋 Shutdown complete")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Tip: Ensure Redis is running or set GECKO_REDIS_URL")

if __name__ == "__main__":
    asyncio.run(main())