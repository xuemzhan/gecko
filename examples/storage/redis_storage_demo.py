# examples/storage/redis_storage_demo.py
import asyncio
import os
from gecko.plugins.storage.factory import create_storage
from gecko.core.exceptions import StorageError, ConfigurationError

# 需要运行真实的 Redis，否则演示连接失败
# export GECKO_REDIS_URL="redis://localhost:6379/0"
REDIS_URL = os.getenv("GECKO_REDIS_URL", "redis://localhost:6379/0")

async def main():
    print(f"🔌 Connecting to {REDIS_URL} with TTL=10s...")
    
    # 1. 正常流程
    try:
        # 附加参数演示
        url_with_ttl = f"{REDIS_URL}?ttl=10"
        storage = await create_storage(url_with_ttl)
        
        session_id = "demo_user_007"
        
        print("💾 Writing data (with 10s TTL)...")
        await storage.set(session_id, { # type: ignore
            "user": "Bond",
            "mission": "Secret"
        })
        
        data = await storage.get(session_id) # type: ignore
        print(f"📖 Read success: {data}")
        
        await storage.shutdown()
        
    except ImportError:
        print("⚠️  Redis client not installed. Run: pip install redis")
        return
    except (ConnectionError, StorageError) as e:
        print(f"⚠️  Redis not available: {e}")
        print("   (Skipping normal test, proceeding to error handling demo)")

    # 2. [新特性] 错误处理演示
    print("\n🛡️  Error Handling Demo (Invalid Host)")
    try:
        # 故意使用不可达的地址
        bad_url = "redis://non-existent-host:6379/0"
        print(f"   Attempting to connect to {bad_url}...")
        
        # 工厂应该抛出 StorageError
        bad_storage = await create_storage(bad_url)
        
    except StorageError as e:
        print(f"✅ Caught expected StorageError: {e}")
        print("   The application handled the connection failure gracefully.")
    except Exception as e:
        print(f"❌ Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())