# examples/session_demo.py
import asyncio
from gecko.core.session import Session, SessionManager
from gecko.plugins.storage.sqlite import SQLiteSessionStorage


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
    info = session.get_info()
    print(f"   访问次数: {info['access_count']}")
    print(f"   状态键数: {info['state_keys']}")
    print(f"   创建时间: {info['created_at']}\n")
    
    # 6. 会话克隆
    print("6. 会话克隆")
    cloned = session.clone(new_id="cloned_session")
    print(f"   原始: {session.session_id}")
    print(f"   克隆: {cloned.session_id}")
    print(f"   克隆的数据: {cloned.get('user_name')}\n")
    
    # 7. 持久化
    print("7. 持久化到存储")
    storage = SQLiteSessionStorage("sqlite://:memory:")
    
    persistent_session = Session(
        session_id="persistent",
        storage=storage,
        auto_save=True
    )
    persistent_session.set("important_data", "must be saved")
    
    await persistent_session.save()
    print("   会话已保存\n")
    
    # 8. 会话管理器
    print("8. 会话管理器")
    manager = SessionManager(
        storage=storage,
        default_ttl=3600,  # 1 小时
        auto_cleanup=True
    )
    
    # 创建多个会话
    s1 = await manager.create_session(user="Alice", score=100)
    s2 = await manager.create_session(user="Bob", score=200)
    
    print(f"   活跃会话数: {manager.get_active_count()}")
    
    # 获取会话
    retrieved = await manager.get_session(s1.session_id)
    print(f"   检索到的用户: {retrieved.get('user')}\n")
    
    # 9. 批量更新
    print("9. 批量更新")
    session.update({
        "last_login": "2024-01-01",
        "login_count": 42,
        "status": "active"
    })
    print(f"   更新后的键: {session.keys()}\n")
    
    # 10. 清理
    print("10. 会话清理")
    await manager.shutdown()
    print("   管理器已关闭，所有会话已保存")


if __name__ == "__main__":
    asyncio.run(main())