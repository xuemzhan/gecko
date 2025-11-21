# tests/core/test_session.py
import pytest
import asyncio
import time
from gecko.core.session import Session, SessionManager, SessionMetadata


class TestSessionMetadata:
    """SessionMetadata 测试"""
    
    def test_create_metadata(self):
        """测试创建元数据"""
        meta = SessionMetadata(session_id="test")
        
        assert meta.session_id == "test"
        assert meta.access_count == 0
        assert meta.ttl is None
    
    def test_is_expired(self):
        """测试过期检查"""
        # 不过期
        meta1 = SessionMetadata(session_id="test1", ttl=None)
        assert not meta1.is_expired()
        
        # 已过期
        meta2 = SessionMetadata(session_id="test2", ttl=1)
        meta2.created_at = time.time() - 2
        assert meta2.is_expired()
    
    def test_touch(self):
        """测试 touch 更新"""
        meta = SessionMetadata(session_id="test")
        
        initial_count = meta.access_count
        meta.touch()
        
        assert meta.access_count == initial_count + 1


class TestSession:
    """Session 测试"""
    
    def test_create_session(self):
        """测试创建会话"""
        session = Session(session_id="test_session")
        
        assert session.session_id == "test_session"
        assert len(session.state) == 0
    
    def test_get_set(self):
        """测试 get/set"""
        session = Session()
        
        session.set("key1", "value1")
        assert session.get("key1") == "value1"
        assert session.get("key2", "default") == "default"
    
    def test_dict_syntax(self):
        """测试字典语法"""
        session = Session()
        
        session["key1"] = "value1"
        assert session["key1"] == "value1"
        assert "key1" in session
    
    def test_update(self):
        """测试批量更新"""
        session = Session()
        
        session.update({"a": 1, "b": 2, "c": 3})
        
        assert session.get("a") == 1
        assert session.get("b") == 2
        assert len(session.keys()) == 3
    
    def test_delete(self):
        """测试删除"""
        session = Session()
        session.set("key", "value")
        
        result = session.delete("key")
        
        assert result is True
        assert session.get("key") is None
    
    def test_clear(self):
        """测试清空"""
        session = Session()
        session.update({"a": 1, "b": 2})
        
        session.clear()
        
        assert len(session.state) == 0
    
    def test_is_expired(self):
        """测试过期"""
        session = Session(ttl=1)
        
        assert not session.is_expired()
        
        # 修改创建时间
        session.metadata.created_at = time.time() - 2
        assert session.is_expired()
    
    def test_extend_ttl(self):
        """测试延长 TTL"""
        session = Session(ttl=10)
        
        session.extend_ttl(5)
        
        assert session.metadata.ttl == 15
    
    def test_tags(self):
        """测试标签"""
        session = Session()
        
        session.add_tag("premium")
        session.add_tag("verified")
        
        assert session.has_tag("premium")
        assert session.has_tag("verified")
        
        session.remove_tag("premium")
        assert not session.has_tag("premium")
    
    def test_clone(self):
        """测试克隆"""
        session = Session(session_id="original")
        session.set("data", "value")
        session.add_tag("tag1")
        
        cloned = session.clone(new_id="cloned")
        
        assert cloned.session_id == "cloned"
        assert cloned.get("data") == "value"
        assert cloned.has_tag("tag1")
    
    def test_to_dict(self):
        """测试序列化"""
        session = Session()
        session.set("key", "value")
        
        data = session.to_dict()
        
        assert "state" in data
        assert "metadata" in data
        assert data["state"]["key"] == "value"
    
    def test_from_dict(self):
        """测试反序列化"""
        session = Session()
        
        data = {
            "state": {"key": "value"},
            "metadata": {
                "session_id": "test",
                "access_count": 5
            }
        }
        
        session.from_dict(data)
        
        assert session.get("key") == "value"
        assert session.metadata.access_count == 5


class TestSessionManager:
    """SessionManager 测试"""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """测试创建会话"""
        manager = SessionManager(auto_cleanup=False)
        
        session = await manager.create_session(user="Alice")
        
        assert session is not None
        assert session.get("user") == "Alice"
        assert manager.get_active_count() == 1
    
    @pytest.mark.asyncio
    async def test_get_session(self):
        """测试获取会话"""
        manager = SessionManager(auto_cleanup=False)
        
        created = await manager.create_session(session_id="test_id")
        retrieved = await manager.get_session("test_id")
        
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """测试获取不存在的会话"""
        manager = SessionManager(auto_cleanup=False)
        
        result = await manager.get_session("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_create_if_missing(self):
        """测试自动创建"""
        manager = SessionManager(auto_cleanup=False)
        
        session = await manager.get_session("new_id", create_if_missing=True)
        
        assert session is not None
        assert session.session_id == "new_id"
    
    @pytest.mark.asyncio
    async def test_destroy_session(self):
        """测试销毁会话"""
        manager = SessionManager(auto_cleanup=False)
        
        session = await manager.create_session(session_id="destroy_me")
        result = await manager.destroy_session("destroy_me")
        
        assert result is True
        assert await manager.get_session("destroy_me") is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """测试清理过期会话"""
        manager = SessionManager(auto_cleanup=False, default_ttl=1)
        
        # 创建会话
        session = await manager.create_session()
        
        # 设置为已过期
        session.metadata.created_at = time.time() - 2
        
        # 清理
        count = await manager.cleanup_expired()
        
        assert count == 1
        assert manager.get_active_count() == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """测试关闭"""
        manager = SessionManager(auto_cleanup=True)
        
        await manager.create_session()
        await manager.shutdown()
        
        # 验证清理任务已取消
        if manager._cleanup_task:
            assert manager._cleanup_task.cancelled() or manager._cleanup_task.done()