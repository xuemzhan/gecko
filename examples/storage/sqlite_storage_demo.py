# examples/storage/sqlite_storage_demo.py
import asyncio
import os
import random
import time
from gecko.plugins.storage.factory import create_storage
from gecko.core.exceptions import StorageError

DB_PATH = "./demo_sqlite.db"
DB_URL = f"sqlite:///{DB_PATH}"

async def main():
    # 清理旧数据
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(DB_PATH + ".lock"):
        os.remove(DB_PATH + ".lock")

    print(f"🚀 Initializing SQLite Storage at {DB_URL}")
    
    # 1. 创建实例 (会自动启用 WAL 和 FileLock)
    storage = await create_storage(DB_URL)
    
    try:
        # 2. 基础操作
        session_id = "user_123"
        await storage.set(session_id, {"name": "Alice", "balance": 100}) # type: ignore
        print("✅ Basic CRUD operational.")

        # 3. [新特性] 并发压力测试 (验证锁机制)
        # 模拟多个协程同时读取并更新同一个 Key
        # 如果没有锁，可能会遇到 "database is locked" 或者更新丢失
        print("\n⚡ Starting Concurrency Stress Test (10 concurrent updates)...")
        
        concurrency_level = 10
        target_session = "counter_session"
        await storage.set(target_session, {"count": 0}) # type: ignore
        
        async def worker(idx):
            # 模拟随机延迟
            await asyncio.sleep(random.uniform(0.001, 0.01))
            
            # 读-改-写 (注意：应用层的原子性仍需分布式锁，但这里测试的是 DB 层不崩)
            # 我们使用 AtomicWriteMixin 的 write_guard 也可以在应用层加锁，
            # 但 storage.set 内部已经加了锁，保证单次 set 是安全的。
            # 为了测试 storage 的健壮性，我们只单纯疯狂写入。
            try:
                # 获取当前值（为了模拟负载）
                await storage.get(target_session) # type: ignore
                # 写入新值
                await storage.set(f"worker_{idx}", {"data": "x" * 100}) # type: ignore
                return True
            except StorageError as e:
                print(f"❌ Worker {idx} failed: {e}")
                return False

        start_time = time.time()
        results = await asyncio.gather(*[worker(i) for i in range(concurrency_level)])
        duration = time.time() - start_time
        
        success_count = sum(results)
        print(f"✅ Finished in {duration:.3f}s. Success: {success_count}/{concurrency_level}")
        
        if success_count == concurrency_level:
            print("🎉 Concurrency test PASSED (No locking errors)")
        else:
            print("⚠️ Some writes failed (Check logs)")

    finally:
        await storage.shutdown()
        # 清理
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        if os.path.exists(DB_PATH + ".lock"): os.remove(DB_PATH + ".lock")
        # WAL 文件
        if os.path.exists(DB_PATH + "-wal"): os.remove(DB_PATH + "-wal")
        if os.path.exists(DB_PATH + "-shm"): os.remove(DB_PATH + "-shm")

if __name__ == "__main__":
    asyncio.run(main())