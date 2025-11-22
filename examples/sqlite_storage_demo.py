# examples/sqlite_storage_demo.py
import asyncio
import os
import time
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

async def main():
    db_path = "./demo_sqlite.db"
    url = f"sqlite:///{db_path}"
    
    print(f"🚀 Initializing SQLite Storage at {url}")
    
    # 1. 创建实例
    storage = SQLiteStorage(url)
    
    try:
        # 2. 初始化 (建表, WAL)
        await storage.initialize()
        
        # 3. 写入测试
        print("\n💾 Saving session data...")
        session_id = "user_session_123"
        data = {
            "name": "Gecko Agent",
            "role": "Assistant",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "metadata": {"timestamp": time.time()}
        }
        await storage.set(session_id, data)
        print("✅ Saved.")
        
        # 4. 读取测试
        print("\n📖 Reading session data...")
        loaded_data = await storage.get(session_id)
        print(f"✅ Loaded: {loaded_data['name']} (History: {len(loaded_data['history'])} msgs)")
        
        # 5. 更新测试
        print("\n🔄 Updating session data...")
        loaded_data["metadata"]["updated"] = True
        await storage.set(session_id, loaded_data)
        
        # 6. 并发测试 (验证是否阻塞)
        print("\n⚡ Testing concurrency (Non-blocking check)...")
        start_time = time.time()
        
        async def background_writer(idx):
            # 模拟写入
            await storage.set(f"bg_sess_{idx}", {"idx": idx})
            return idx

        # 同时发起 10 个写操作
        tasks = [background_writer(i) for i in range(10)]
        # 同时做一个 Sleep 模拟 Event Loop 其他任务
        tasks.append(asyncio.sleep(0.1))
        
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        print(f"✅ Concurrency test passed in {duration:.3f}s")
        
    finally:
        # 7. 关闭
        await storage.shutdown()
        # 清理
        if os.path.exists(db_path):
            os.remove(db_path)
            # WAL 模式会产生 .wal 和 .shm 文件
            if os.path.exists(db_path + "-wal"): os.remove(db_path + "-wal")
            if os.path.exists(db_path + "-shm"): os.remove(db_path + "-shm")
        print("\n👋 Cleanup done.")

if __name__ == "__main__":
    asyncio.run(main())