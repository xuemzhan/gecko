# examples/storage/custom_storage_demo.py
import asyncio
import json
import os
from typing import Any, Dict, Optional

# 导入 Gecko 的基类和 Mixin
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import (
    ThreadOffloadMixin,
    AtomicWriteMixin,
    JSONSerializerMixin
)
from gecko.plugins.storage.registry import register_storage

# ================= 自定义实现 =================

@register_storage("myjson")
class SimpleJsonStorage(
    AbstractStorage,
    SessionInterface,
    ThreadOffloadMixin,  # 1. 自动将 IO 放入线程池
    AtomicWriteMixin,    # 2. 自动提供 FileLock 和 AsyncLock
    JSONSerializerMixin  # 3. 提供 _serialize/_deserialize
):
    """
    一个极其简单但健壮的 JSON 文件存储
    URL: myjson://./data.json
    """
    
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        # 解析路径: myjson://./data.json -> ./data.json
        self.file_path = url.replace("myjson://", "")
        
        # [关键] 配置 FileLock，这样即使多个进程同时操作这个文件也不会坏
        self.setup_multiprocess_lock(self.file_path)

    async def initialize(self) -> None:
        """初始化：确保文件存在"""
        if not os.path.exists(self.file_path):
            # 使用 run_sync 在线程中执行文件写入
            await self._run_sync(self._write_file, {})
        self._is_initialized = True
        print(f"[Init] Storage ready at {self.file_path}")

    async def shutdown(self) -> None:
        self._is_initialized = False

    # --- 核心逻辑 (全部是同步写法，由 Mixin 处理异步) ---

    def _read_file(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_file(self, data: Dict[str, Any]):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # --- 接口实现 ---

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        # 读操作：只需要卸载到线程池，不需要加写锁
        data = await self._run_sync(self._read_file)
        return data.get(session_id)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        # 写操作逻辑
        def _do_update():
            data = self._read_file()
            data[session_id] = state
            self._write_file(data)
            return len(data)

        # [关键] 使用 write_guard 保护临界区 (包含 FileLock)
        async with self.write_guard():
            count = await self._run_sync(_do_update)
            print(f"   [Write] Saved session {session_id}. Total sessions: {count}")

    async def delete(self, session_id: str) -> None:
        def _do_delete():
            data = self._read_file()
            if session_id in data:
                del data[session_id]
                self._write_file(data)

        async with self.write_guard():
            await self._run_sync(_do_delete)

# ================= 测试流程 =================

async def main():
    db_file = "demo_custom.json"
    url = f"myjson://{db_file}"
    
    # 清理环境
    if os.path.exists(db_file): os.remove(db_file)
    if os.path.exists(db_file + ".lock"): os.remove(db_file + ".lock")

    print(f"🚀 Testing Custom Storage: {url}")
    
    # 1. 实例化 (无需工厂，直接用类演示，或通过 create_storage 也可以)
    storage = SimpleJsonStorage(url)
    await storage.initialize()

    try:
        # 2. 并发写入测试
        print("\n⚡ Starting Concurrent Write Test...")
        
        async def worker(idx):
            # 模拟并发 Agent 写入
            await storage.set(f"user_{idx}", {"score": idx * 10}) # type: ignore
        
        # 启动 10 个并发任务
        # 如果没有 AtomicWriteMixin，这里大概率会报 JSONDecodeError 或内容损坏
        await asyncio.gather(*[worker(i) for i in range(10)])
        
        # 3. 验证结果
        print("\n🔍 Verifying Data...")
        all_data = await storage._run_sync(storage._read_file) # type: ignore
        print(f"   Total Records: {len(all_data)}")
        
        assert len(all_data) == 10
        assert all_data["user_9"]["score"] == 90
        print("✅ Data integrity check passed!")

    finally:
        await storage.shutdown()
        # 清理
        if os.path.exists(db_file): os.remove(db_file)
        if os.path.exists(db_file + ".lock"): os.remove(db_file + ".lock")

if __name__ == "__main__":
    asyncio.run(main())