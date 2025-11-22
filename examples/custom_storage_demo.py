# examples/custom_storage_demo.py
import asyncio
import os
import json
import tempfile
import shutil
from typing import Dict, Any
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.mixins import ThreadOffloadMixin, AtomicWriteMixin, JSONSerializerMixin
from gecko.plugins.storage.interfaces import SessionInterface

class JsonFileStorage(
    AbstractStorage, 
    SessionInterface, 
    ThreadOffloadMixin, 
    AtomicWriteMixin, 
    JSONSerializerMixin
):
    """
    演示：一个简单的基于 JSON 文件的存储实现
    
    特性：
    1. 线程安全（ThreadOffload）
    2. 进程内写锁（AtomicWrite）
    3. 自动序列化
    4. [修复] 原子文件写入，防止并发读写时的竞态崩溃
    """
    
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        # 解析 path: json://./data.json -> ./data.json
        self.file_path = url.replace("json://", "")
    
    async def initialize(self) -> None:
        print(f"[Init] Checking file: {self.file_path}")
        # 在线程中检查文件
        await self._run_sync(self._ensure_file)

    async def shutdown(self) -> None:
        print("[Shutdown] Storage closed")

    def _ensure_file(self):
        """同步文件检查逻辑"""
        if not os.path.exists(self.file_path):
            self._write_json_atomically({})

    def _write_json_atomically(self, data: Dict[str, Any]):
        """
        [核心修复] 原子写入文件
        
        步骤：
        1. 写入临时文件
        2. 原子重命名覆盖原文件
        
        这防止了 reader 在 writer 截断文件但未完成写入时读取到空文件。
        """
        dir_name = os.path.dirname(self.file_path) or "."
        # 在同一目录下创建临时文件，确保原子重命名可行（跨分区无法原子重命名）
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8", suffix=".tmp") as tmp_f:
            json.dump(data, tmp_f, ensure_ascii=False, indent=2)
            tmp_name = tmp_f.name
        
        # 原子替换
        try:
            shutil.move(tmp_name, self.file_path)
        except Exception:
            # 清理残余
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    async def get(self, session_id: str) -> Dict[str, Any] | None:
        def _read():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get(session_id)
            except (FileNotFoundError, json.JSONDecodeError):
                return None
        
        # 卸载到线程池读取
        return await self._run_sync(_read)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        def _write():
            # 1. 读取最新数据 (在写锁保护下，确保读-改-写的一致性)
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            
            # 2. 更新内存
            data[session_id] = state
            
            # 3. 原子写入磁盘
            self._write_json_atomically(data)

        # 加锁 + 线程卸载
        async with self.write_guard():
            await self._run_sync(_write)

    async def delete(self, session_id: str) -> None:
        def _delete():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return

            if session_id in data:
                del data[session_id]
                self._write_json_atomically(data)
        
        async with self.write_guard():
            await self._run_sync(_delete)

async def main():
    file_path = "demo_sessions.json"
    # 清理旧文件防止干扰
    if os.path.exists(file_path):
        os.remove(file_path)

    storage = JsonFileStorage(f"json://{file_path}")
    
    try:
        # 1. 初始化
        await storage.initialize()
        
        # 2. 写入数据
        print("Writing session...")
        await storage.set("user_1", {"name": "Alice", "history": ["Hi"]})
        
        # 3. 读取数据
        data = await storage.get("user_1")
        print(f"Read session: {data}")
        
        # 4. 并发写入测试
        print("Testing concurrent writes...")
        async def update_age(age):
            # 模拟业务逻辑：读 -> 改 -> 写
            # 注意：这里的业务逻辑本身在极端并发下存在"更新丢失"风险，
            # 但我们要测试的是 storage 层不会崩溃。
            s = await storage.get("user_1") or {}
            s["age"] = age
            await storage.set("user_1", s)
            print(f"Updated age to {age}")

        # 并发执行 5 次，增加压力
        await asyncio.gather(
            update_age(20),
            update_age(25),
            update_age(30),
            update_age(35),
            update_age(40)
        )
        
        final_data = await storage.get("user_1")
        print(f"Final data: {final_data}")
        
    finally:
        await storage.shutdown()
        # 清理文件
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    asyncio.run(main())