# tests/core/test_utils.py
import threading
from typing import Any
from pydantic import BaseModel
import pytest
import asyncio
from gecko.core.utils import (
    ensure_awaitable,
    retry,
    safe_dict,
    merge_dicts,
    safe_serialize_context,
    truncate,
    format_size,
    format_duration,
    Timer,
    chunk_list,
    flatten_list,
    deduplicate,
    get_function_args,
    has_argument,
)


class TestEnsureAwaitable:
    """ensure_awaitable 测试"""
    
    @pytest.mark.asyncio
    async def test_sync_function(self):
        """测试同步函数"""
        def sync_func(x):
            return x * 2
        
        result = await ensure_awaitable(sync_func, 5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """测试异步函数"""
        async def async_func(x):
            return x * 3
        
        result = await ensure_awaitable(async_func, 5)
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_with_timeout(self):
        """测试超时"""
        async def slow_func():
            await asyncio.sleep(2)
            return "done"
        
        with pytest.raises(asyncio.TimeoutError):
            await ensure_awaitable(slow_func, timeout=0.5)


class TestRetry:
    """重试测试"""
    
    @pytest.mark.asyncio
    async def test_retry_success(self):
        """测试成功重试"""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        async def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = await sometimes_fails()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_all_fail(self):
        """测试所有尝试都失败"""
        @retry(max_attempts=2, delay=0.1)
        async def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await always_fails()


class TestDataConversion:
    """数据转换测试"""
    
    def test_safe_dict_simple(self):
        """测试简单对象转换"""
        class Simple:
            def __init__(self):
                self.name = "test"
                self.value = 123
        
        obj = Simple()
        data = safe_dict(obj)
        
        assert data["name"] == "test"
        assert data["value"] == 123
    
    def test_merge_dicts_shallow(self):
        """测试浅合并"""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        
        result = merge_dicts(d1, d2)
        
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4
    
    def test_merge_dicts_deep(self):
        """测试深合并"""
        d1 = {"a": {"x": 1}}
        d2 = {"a": {"y": 2}}
        
        result = merge_dicts(d1, d2, deep=True)
        
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 2


class TestStringUtils:
    """字符串工具测试"""
    
    def test_truncate(self):
        """测试截断"""
        text = "A" * 200
        result = truncate(text, max_length=50)
        
        assert len(result) == 50
        assert result.endswith("...")
    
    def test_format_size(self):
        """测试大小格式化"""
        assert "1.00 KB" in format_size(1024)
        assert "1.00 MB" in format_size(1048576)
    
    def test_format_duration(self):
        """测试时长格式化"""
        assert "5.0s" in format_duration(5)
        assert "1m" in format_duration(65)
        assert "1h" in format_duration(3665)


class TestTimer:
    """计时器测试"""
    
    @pytest.mark.asyncio
    async def test_timer(self):
        """测试计时器"""
        with Timer("test", log=False) as t:
            await asyncio.sleep(0.1)
        
        assert t.elapsed >= 0.1
        assert t.elapsed < 0.2


class TestListUtils:
    """列表工具测试"""
    
    def test_chunk_list(self):
        """测试分块"""
        chunks = chunk_list([1, 2, 3, 4, 5], chunk_size=2)
        
        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [3, 4]
        assert chunks[2] == [5]
    
    def test_flatten_list(self):
        """测试展平"""
        nested = [[1, 2], [3, 4], [5]]
        flat = flatten_list(nested)
        
        assert flat == [1, 2, 3, 4, 5]
    
    def test_deduplicate(self):
        """测试去重"""
        items = [1, 2, 2, 3, 1, 4]
        unique = deduplicate(items)
        
        assert unique == [1, 2, 3, 4]
    
    def test_deduplicate_with_key(self):
        """测试按键去重"""
        items = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
            {"id": 1, "name": "C"},
        ]
        unique = deduplicate(items, key=lambda x: x["id"])
        
        assert len(unique) == 2
        assert unique[0]["id"] == 1
        assert unique[1]["id"] == 2


class TestFunctionUtils:
    """函数工具测试"""
    
    def test_get_function_args(self):
        """测试获取函数参数"""
        def test_func(a, b, c=None):
            pass
        
        args = get_function_args(test_func)
        
        assert "a" in args
        assert "b" in args
        assert "c" in args
    
    def test_has_argument(self):
        """测试检查参数"""
        def test_func(a, b):
            pass
        
        assert has_argument(test_func, "a")
        assert not has_argument(test_func, "c")

class TestSerializationUtils:
    """[New] 序列化工具测试"""

    def test_safe_serialize_basic(self):
        """测试基础类型和 Pydantic 对象"""
        class MyModel(BaseModel):
            name: str = "test"
        
        data = {
            "a": 1,
            "b": MyModel(),
            "c": [1, 2]
        }
        
        clean = safe_serialize_context(data)
        assert clean["a"] == 1
        assert clean["b"] == {"name": "test"} # Pydantic 转 dict
        assert clean["c"] == [1, 2]

    def test_safe_serialize_unserializable(self):
        """[Core] 测试不可序列化对象的安全降级"""
        lock = threading.Lock()
        data = {
            "valid": "data",
            "dangerous": lock,
            "nested": {"inner_lock": lock}
        }
        
        clean = safe_serialize_context(data)
        
        assert clean["valid"] == "data"
        
        # 验证被替换为标记字典，而不是抛出异常
        assert isinstance(clean["dangerous"], dict)
        assert clean["dangerous"].get("__gecko_unserializable__") is True
        assert "lock" in clean["dangerous"]["type"].lower()
        
        # 验证递归处理
        assert clean["nested"]["inner_lock"].get("__gecko_unserializable__") is True

    def test_safe_serialize_recursion_limit(self):
        """测试防止无限递归"""
        # 构造循环引用
        d = {}
        d["self"] = d
    
        # 应该优雅处理
        clean = safe_serialize_context(d)
        
        # [修改] clean['self'] 本身还是一个 dict (因为递归在更深层被截断)，只要不抛错且是 dict 即可
        # 结构会是 {'self': {'self': ... {'self': "<...>"}}}
        assert isinstance(clean["self"], dict)
        # 验证确实有内容
        assert "self" in clean["self"]

def test_safe_serialize_unserializable():
    """测试不可序列化对象的降级处理"""
    import threading
    lock = threading.Lock()
    data = {"lock": lock}
    
    clean = safe_serialize_context(data)
    
    # 必须变为 dict 标记，且不抛错
    assert isinstance(clean["lock"], dict)
    assert clean["lock"].get("__gecko_unserializable__") is True

def test_safe_serialize_complex_objects():
    """[New] 测试复杂对象的序列化清洗能力"""
    import threading
    from pydantic import BaseModel, Field, PrivateAttr
    
    class UnserializableModel(BaseModel):
        name: str
        # [FIX] 使用 PrivateAttr 或 default_factory 防止 Pydantic 深拷贝默认值
        _lock: Any = PrivateAttr(default_factory=threading.Lock)

    lock = threading.Lock()
    
    data = {
        "normal": "value",
        "nested": {
            "lock_obj": lock,
            "list": [1, lock, 2]
        },
        "pydantic_obj": UnserializableModel(name="test")
    }
    
    clean = safe_serialize_context(data)
    
    # 1. 验证正常数据保留
    assert clean["normal"] == "value"
    assert clean["nested"]["list"][0] == 1
    
    # 2. 验证锁对象被转换为标记字典
    lock_marker = clean["nested"]["lock_obj"]
    assert isinstance(lock_marker, dict)
    assert lock_marker.get("__gecko_unserializable__") is True
    assert "lock" in lock_marker.get("type", "")
    
    # 3. 验证列表中的锁也被处理
    assert clean["nested"]["list"][1].get("__gecko_unserializable__") is True
    
    # 4. 验证 Pydantic 对象被转为 dict
    assert clean["pydantic_obj"] == {"name": "test"}