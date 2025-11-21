# tests/core/test_utils.py
import pytest
import asyncio
from gecko.core.utils import (
    ensure_awaitable,
    retry,
    safe_dict,
    merge_dicts,
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