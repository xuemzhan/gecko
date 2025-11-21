# gecko/core/utils.py
"""
通用工具函数库

提供框架常用的工具函数，包括：
- 异步/同步统一处理
- 重试机制
- 超时控制
- 数据转换
- 字符串处理
- 装饰器

优化点：
1. 扩展工具函数集合
2. 添加超时和重试支持
3. 提供数据转换工具
4. 添加常用装饰器
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ===== 异步/同步统一处理 =====

async def ensure_awaitable(
    func: Callable[..., T | Awaitable[T]],
    *args,
    timeout: Optional[float] = None,
    **kwargs
) -> T:
    """
    统一处理同步/异步函数调用
    
    参数:
        func: 可调用对象（同步或异步）
        *args: 位置参数
        timeout: 超时时间（秒），None 表示无限制
        **kwargs: 关键字参数
    
    返回:
        函数执行结果
    
    异常:
        asyncio.TimeoutError: 超时
        Exception: 函数执行异常
    
    示例:
        ```python
        # 同步函数
        result = await ensure_awaitable(sync_func, arg1, arg2)
        
        # 异步函数
        result = await ensure_awaitable(async_func, arg1, arg2)
        
        # 带超时
        result = await ensure_awaitable(func, timeout=5.0)
        ```
    """
    # 确定是否为协程函数
    if asyncio.iscoroutinefunction(func):
        coro = func(*args, **kwargs)
    else:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            coro = result
        else:
            # 同步函数，直接返回结果
            return result
    
    # 执行异步函数（带可选超时）
    if timeout:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Function execution timeout",
                func=getattr(func, "__name__", str(func)),
                timeout=timeout
            )
            raise
    else:
        return await coro


def run_sync(coro: Awaitable[T]) -> T:
    """
    在同步上下文中运行异步函数
    
    参数:
        coro: 协程对象
    
    返回:
        执行结果
    
    示例:
        ```python
        async def async_func():
            return "result"
        
        # 在同步代码中调用
        result = run_sync(async_func())
        ```
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建新的事件循环
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # 没有事件循环，创建新的
        return asyncio.run(coro)


# ===== 重试机制 =====

async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs
) -> T:
    """
    异步函数重试装饰器
    
    参数:
        func: 异步函数
        *args: 位置参数
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 退避倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        **kwargs: 关键字参数
    
    返回:
        函数执行结果
    
    异常:
        最后一次尝试的异常
    
    示例:
        ```python
        async def unstable_api_call():
            # 可能失败的 API 调用
            ...
        
        result = await retry_async(
            unstable_api_call,
            max_attempts=5,
            delay=2.0,
            backoff=2.0
        )
        ```
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt >= max_attempts:
                logger.error(
                    "All retry attempts failed",
                    func=getattr(func, "__name__", str(func)),
                    attempts=max_attempts,
                    error=str(e)
                )
                raise
            
            logger.warning(
                "Function failed, retrying",
                func=getattr(func, "__name__", str(func)),
                attempt=attempt,
                max_attempts=max_attempts,
                delay=current_delay,
                error=str(e)
            )
            
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception as callback_error:
                    logger.warning("Retry callback failed", error=str(callback_error))
            
            await asyncio.sleep(current_delay)
            current_delay *= backoff
    
    raise last_exception


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    重试装饰器（支持同步和异步）
    
    参数:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 退避倍数
        exceptions: 需要重试的异常类型
    
    示例:
        ```python
        @retry(max_attempts=5, delay=2.0)
        async def unstable_function():
            # 可能失败的操作
            ...
        ```
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_async(
                    func,
                    *args,
                    max_attempts=max_attempts,
                    delay=delay,
                    backoff=backoff,
                    exceptions=exceptions,
                    **kwargs
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt >= max_attempts:
                            raise
                        time.sleep(current_delay)
                        current_delay *= backoff
                
                raise last_exception
            
            return sync_wrapper
    
    return decorator


# ===== 数据转换 =====

def safe_dict(obj: Any, max_depth: int = 3, _current_depth: int = 0) -> Any:
    """
    安全地将对象转换为字典（递归处理）
    
    参数:
        obj: 要转换的对象
        max_depth: 最大递归深度
        _current_depth: 当前深度（内部使用）
    
    返回:
        字典或可序列化的值
    
    示例:
        ```python
        class MyClass:
            def __init__(self):
                self.name = "test"
                self.value = 123
        
        obj = MyClass()
        data = safe_dict(obj)
        # {"name": "test", "value": 123}
        ```
    """
    if _current_depth >= max_depth:
        return str(obj)[:100]
    
    # Pydantic 模型
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    
    # 字典
    if isinstance(obj, dict):
        return {
            str(k): safe_dict(v, max_depth, _current_depth + 1)
            for k, v in obj.items()
        }
    
    # 列表/元组
    if isinstance(obj, (list, tuple)):
        return [safe_dict(item, max_depth, _current_depth + 1) for item in obj]
    
    # 基本类型
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # 对象属性
    if hasattr(obj, "__dict__"):
        return {
            k: safe_dict(v, max_depth, _current_depth + 1)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    
    # 其他情况：转字符串
    return str(obj)[:200]


def merge_dicts(*dicts: Dict, deep: bool = False) -> Dict:
    """
    合并多个字典
    
    参数:
        *dicts: 要合并的字典
        deep: 是否深度合并
    
    返回:
        合并后的字典
    
    示例:
        ```python
        d1 = {"a": 1, "b": {"x": 1}}
        d2 = {"b": {"y": 2}, "c": 3}
        
        # 浅合并
        result = merge_dicts(d1, d2)
        # {"a": 1, "b": {"y": 2}, "c": 3}
        
        # 深合并
        result = merge_dicts(d1, d2, deep=True)
        # {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}
        ```
    """
    if not dicts:
        return {}
    
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
        
        for key, value in d.items():
            if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    
    return result


# ===== 字符串处理 =====

def truncate(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    截断文本
    
    参数:
        text: 文本
        max_length: 最大长度
        suffix: 后缀
    
    返回:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_size(size_bytes: int) -> str:
    """
    格式化字节大小
    
    参数:
        size_bytes: 字节数
    
    返回:
        可读的大小字符串
    
    示例:
        ```python
        format_size(1024)       # "1.00 KB"
        format_size(1048576)    # "1.00 MB"
        format_size(1073741824) # "1.00 GB"
        ```
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    格式化时长
    
    参数:
        seconds: 秒数
    
    返回:
        可读的时长字符串
    
    示例:
        ```python
        format_duration(65)    # "1m 5s"
        format_duration(3665)  # "1h 1m 5s"
        ```
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    return f"{hours}h {minutes}m {secs}s"


def compute_hash(text: str, algorithm: str = "md5") -> str:
    """
    计算文本的哈希值
    
    参数:
        text: 文本
        algorithm: 算法（md5/sha1/sha256）
    
    返回:
        哈希值（十六进制字符串）
    
    示例:
        ```python
        hash_value = compute_hash("Hello, World!")
        ```
    """
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


# ===== 性能监控 =====

class Timer:
    """
    简单的计时器（上下文管理器）
    
    示例:
        ```python
        with Timer("操作名称") as t:
            # 执行耗时操作
            do_something()
        
        print(f"耗时: {t.elapsed:.2f}s")
        ```
    """
    
    def __init__(self, name: str = "Timer", log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.log:
            logger.info(
                "Timer completed",
                name=self.name,
                elapsed=f"{self.elapsed:.3f}s"
            )
        
        return False


def timing(func: Callable) -> Callable:
    """
    计时装饰器（支持同步和异步）
    
    示例:
        ```python
        @timing
        async def slow_function():
            await asyncio.sleep(1)
        ```
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                logger.info(
                    "Function executed",
                    func=func.__name__,
                    elapsed=f"{elapsed:.3f}s"
                )
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                logger.info(
                    "Function executed",
                    func=func.__name__,
                    elapsed=f"{elapsed:.3f}s"
                )
        return sync_wrapper


# ===== 函数签名工具 =====

def get_function_args(func: Callable) -> List[str]:
    """
    获取函数的参数名列表
    
    参数:
        func: 函数对象
    
    返回:
        参数名列表
    """
    sig = inspect.signature(func)
    return [
        name for name, param in sig.parameters.items()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def has_argument(func: Callable, arg_name: str) -> bool:
    """
    检查函数是否有指定参数
    
    参数:
        func: 函数对象
        arg_name: 参数名
    
    返回:
        是否存在该参数
    """
    return arg_name in get_function_args(func)


# ===== 其他工具 =====

def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    将列表分块
    
    参数:
        lst: 列表
        chunk_size: 每块大小
    
    返回:
        分块后的列表
    
    示例:
        ```python
        chunks = chunk_list([1, 2, 3, 4, 5], chunk_size=2)
        # [[1, 2], [3, 4], [5]]
        ```
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested: List[List[T]]) -> List[T]:
    """
    展平嵌套列表
    
    参数:
        nested: 嵌套列表
    
    返回:
        展平后的列表
    
    示例:
        ```python
        flat = flatten_list([[1, 2], [3, 4], [5]])
        # [1, 2, 3, 4, 5]
        ```
    """
    return [item for sublist in nested for item in sublist]


def deduplicate(
    items: List[T],
    key: Optional[Callable[[T], Any]] = None
) -> List[T]:
    """
    列表去重（保持顺序）
    
    参数:
        items: 列表
        key: 可选的键函数
    
    返回:
        去重后的列表
    
    示例:
        ```python
        # 简单去重
        unique = deduplicate([1, 2, 2, 3, 1])
        # [1, 2, 3]
        
        # 按属性去重
        users = [{"id": 1, "name": "A"}, {"id": 1, "name": "B"}]
        unique = deduplicate(users, key=lambda u: u["id"])
        ```
    """
    seen = set()
    result = []
    
    for item in items:
        k = key(item) if key else item
        
        # 处理不可哈希的情况
        try:
            if k not in seen:
                seen.add(k)
                result.append(item)
        except TypeError:
            # 不可哈希，使用 == 比较
            if not any(k == s for s in seen):
                seen.add(k)
                result.append(item)
    
    return result


# ===== 向后兼容导出 =====

__all__ = [
    # 异步工具
    "ensure_awaitable",
    "run_sync",
    # 重试
    "retry",
    "retry_async",
    # 数据转换
    "safe_dict",
    "merge_dicts",
    # 字符串
    "truncate",
    "format_size",
    "format_duration",
    "compute_hash",
    # 性能
    "Timer",
    "timing",
    # 函数工具
    "get_function_args",
    "has_argument",
    # 列表工具
    "chunk_list",
    "flatten_list",
    "deduplicate",
]