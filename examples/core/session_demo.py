# examples/session_demo.py
import asyncio
import os
from gecko.core.session import Session, SessionManager
from gecko.plugins.storage.backends.sqlite import SQLiteStorage


async def main():
    print("=== Gecko Session 示例 ===\n")
    
    # 1. 基础会话使用
    print("1. 基础会话")
    session = Session(session_id="demo_session")
    
    # 设置状态
    session.set("user_name", "Alice")
    session.set("preferences", {"theme": "dark", "language": "zh"})
    
    # 获取状态
    name = session.get("user_name")
    prefs = session.get("preferences")
    print(f"   用户: {name}")
    print(f"   偏好: {prefs}\n")
    
    # 2. 使用字典语法
    print("2. 字典语法")
    session["score"] = 100
    session["level"] = 5
    
    print(f"   Score: {session['score']}")
    print(f"   Level: {session['level']}")
    print(f"   Keys: {session.keys()}\n")
    
    # 3. TTL 和过期
    print("3. TTL 和过期")
    temp_session = Session(session_id="temp", ttl=5)  # 5 秒后过期
    temp_session.set("data", "temporary")
    
    print(f"   是否过期: {temp_session.is_expired()}")
    print(f"   剩余时间: {temp_session.metadata.time_to_expire():.1f}s")
    
    # 延长 TTL
    temp_session.extend_ttl(10)
    print(f"   延长后剩余: {temp_session.metadata.time_to_expire():.1f}s\n")
    
    # 4. 标签管理
    print("4. 标签管理")
    session.add_tag("premium")
    session.add_tag("verified")
    
    print(f"   标签: {session.metadata.tags}")
    print(f"   是否 premium: {session.has_tag('premium')}\n")
    
    # 5. 会话信息
    print("5. 会话信息")
    info = session.get_info() # type: ignore
    print(f"   访问次数: {info['access_count']}")
    print(f"   状态键数: {info['state_keys']}")
    print(f"   创建时间: {info['created_at']}\n")
    
    # 6. 会话克隆
    print("6. 会话克隆")
    cloned = session.clone(new_id="cloned_session")
    print(f"   原始: {session.session_id}")
    print(f"   克隆: {cloned.session_id}")
    print(f"   克隆的数据: {cloned.get('user_name')}\n")
    
    # 7. 持久化与一致性测试 [修改]
    print("7. 持久化与一致性测试")
    
    # [修复] 使用更健壮的数据库路径处理
    db_file = "session_demo.db"
    # 获取当前脚本所在目录的绝对路径，或者使用当前工作目录的绝对路径
    # 这里使用 os.getcwd() 确保在当前运行目录下创建
    db_path = os.path.join(os.getcwd(), db_file)
    db_url = f"sqlite:///{db_path}"
    
    # [修复] 启动前清理旧文件，防止锁残留
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError as e:
            print(f"⚠️ Warning: Could not remove old DB file: {e}")
    
    # 使用绝对路径 URL 初始化
    # 注意：SQLiteSessionStorage 类可能需要导入，或者使用 factory create_storage
    # 假设这里使用的是 gecko.plugins.storage.backends.sqlite.SQLiteStorage
    from gecko.plugins.storage.backends.sqlite import SQLiteStorage
    storage = SQLiteStorage(db_url) 
    
    try:
        await storage.initialize()
        
        persistent_session = Session(
            session_id="race_test",
            storage=storage,
            auto_save=False 
        )
    
        # 初始状态
        persistent_session.set("counter", 0)
        
        print("   启动并发修改任务...")
        
        # 模拟：保存的同时修改数据
        async def save_task():
            print("   [Task 1] 开始保存 (Counter=0)...")
            await persistent_session.save()
            print("   [Task 1] 保存完成")

        async def modify_task():
            # 稍微延迟，确保 save 已经进入 IO 等待
            await asyncio.sleep(0.01)
            print("   [Task 2] 修改数据 (Counter->999)...")
            persistent_session.set("counter", 999)

        # 并发执行
        await asyncio.gather(save_task(), modify_task())
        
        # 验证存储中的数据
        # 期望：存储中是保存开始时的快照 (0)，而不是修改后的 (999)
        # 注意：由于我们是并发的，如果 save 很快完成，可能存的是 0；如果 modify 先执行，存的是 999。
        # 但优化后的 save 方法会先同步快照，所以只要 save 先被调度，它就会锁定当前状态 0。
        
        loaded_data = await storage.get("race_test")
        print(f"   存储中的 Counter: {loaded_data['state']['counter']}") # type: ignore
        print(f"   内存中的 Counter: {persistent_session.get('counter')}")
    
    finally:
        await storage.shutdown()
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass
    print()
    
    # 8. 会话管理器 [修复开始]
    print("8. 会话管理器")
    
    # [修复] 重新创建一个 Storage 实例，或者重用变量但必须重新 initialize
    # 既然文件被删了，必须重新 init 来建表
    # 建议：为了清晰，重新实例化
    
    # 使用内存数据库演示管理器功能，避免文件残留问题
    manager_storage = SQLiteStorage("sqlite:///:memory:")
    await manager_storage.initialize() # 必须显式初始化
    
    manager = SessionManager(
        storage=manager_storage, # 使用新的 storage 实例
        default_ttl=3600,
        auto_cleanup=True
    )
    
    # 创建多个会话
    s1 = await manager.create_session(user="Alice", score=100)
    s2 = await manager.create_session(user="Bob", score=200)
    
    print(f"   活跃会话数: {manager.get_active_count()}")
    
    # 获取会话
    retrieved = await manager.get_session(s1.session_id)
    print(f"   检索到的用户: {retrieved.get('user')}\n") # type: ignore
    
    # 9. 批量更新
    print("9. 批量更新")
    # 注意：这里的 session 变量是第 1 部分创建的内存 Session，
    # 它没有绑定 storage，所以 update 是安全的内存操作
    session.update({
        "last_login": "2024-01-01",
        "login_count": 42,
        "status": "active"
    })
    print(f"   更新后的键: {session.keys()}\n")
    
    # 10. 清理
    print("10. 会话清理")
    await manager.shutdown()
    # 同时也关闭底层的 manager_storage
    await manager_storage.shutdown()
    print("   管理器已关闭，所有会话已保存")


if __name__ == "__main__":
    asyncio.run(main())