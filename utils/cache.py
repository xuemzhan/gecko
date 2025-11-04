# agno/utils/cache.py

"""
缓存工具模块

该模块提供了通用的缓存功能，旨在提高重复计算或IO密集型操作的性能。

主要功能:
- cache_result: 基于文件系统的缓存装饰器，支持TTL过期机制
"""

import functools
import hashlib
import inspect
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

# 常量定义
DEFAULT_CACHE_DIR_NAME = "agno_cache"
DEFAULT_CACHE_TTL = 3600  # 1小时
CACHE_FILE_EXTENSION = ".json"
CACHE_TIMESTAMP_KEY = "timestamp"
CACHE_RESULT_KEY = "result"
HASH_ALGORITHM = "md5"
ENCODING = "utf-8"

# 日志配置
logger = logging.getLogger(__name__)


def log_debug(message: str) -> None:
    """记录调试级别日志"""
    logger.debug(message)


def log_warning(message: str) -> None:
    """记录警告级别日志"""
    logger.warning(message)


# 类型变量
T = TypeVar("T")


# --- 缓存装饰器 ---

def cache_result(
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    创建基于文件的函数结果缓存装饰器
    
    缓存键基于函数模块、名称和所有参数生成唯一哈希值。
    参数会被规范化，确保相同的参数值产生相同的缓存键。
    
    注意：
    - 返回值必须是 JSON 可序列化的
    - 参数应该是可序列化的以确保缓存键稳定
    
    Args:
        enable_cache: 是否启用缓存，False 时直接执行原函数
        cache_dir: 缓存文件存储目录，None 则使用系统临时目录
        cache_ttl: 缓存生存时间（秒），默认 3600 秒
        
    Returns:
        装饰器函数
        
    Examples:
        >>> @cache_result(cache_ttl=60)
        ... def expensive_operation(x: int) -> int:
        ...     return x * 2
        >>> expensive_operation(5)
        10
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # 未启用缓存，直接执行
            if not enable_cache:
                return func(*args, **kwargs)
            
            # 准备缓存环境
            func_cache_dir = _get_function_cache_dir(func, cache_dir)
            cache_file = _get_cache_file_path(
                func, func_cache_dir, args, kwargs
            )
            
            # 尝试读取缓存
            cached_result = _read_cache(cache_file, cache_ttl, func.__name__)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            log_debug(f"函数 '{func.__name__}' 未命中缓存，正在执行...")
            result = func(*args, **kwargs)
            
            # 写入缓存
            _write_cache(cache_file, result, func.__name__)
            
            return result
        
        return wrapper
    
    return decorator


# --- 缓存路径管理 ---

def _get_function_cache_dir(
    func: Callable,
    cache_dir: Optional[str] = None
) -> Path:
    """
    获取函数的缓存目录路径
    
    为每个函数创建独立子目录，避免哈希冲突。
    
    Args:
        func: 被装饰的函数
        cache_dir: 自定义缓存根目录
        
    Returns:
        函数缓存目录的 Path 对象
    """
    # 确定基础缓存目录
    if cache_dir:
        base_dir = Path(cache_dir)
    else:
        base_dir = Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIR_NAME
    
    # 为函数创建独立目录
    func_dir = base_dir / func.__module__ / func.__qualname__
    
    # 创建目录
    try:
        func_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_warning(f"创建缓存目录失败: {e}")
        # 回退到临时目录
        func_dir = Path(tempfile.gettempdir()) / "agno_cache_fallback"
        func_dir.mkdir(parents=True, exist_ok=True)
    
    return func_dir


def _get_cache_file_path(
    func: Callable,
    cache_dir: Path,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Path:
    """
    生成缓存文件路径
    
    Args:
        func: 被装饰的函数
        cache_dir: 缓存目录
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        缓存文件的 Path 对象
    """
    cache_key = _generate_cache_key(func, args, kwargs)
    return cache_dir / f"{cache_key}{CACHE_FILE_EXTENSION}"


def _generate_cache_key(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> str:
    """
    生成缓存键的哈希值
    
    基于函数标识和规范化的参数生成唯一键。
    参数会被规范化以确保相同参数值产生相同的键。
    
    Args:
        func: 被装饰的函数
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        MD5 哈希字符串
    """
    # 规范化参数
    normalized_kwargs = _normalize_arguments(func, args, kwargs)
    
    # 序列化规范化后的参数
    params_repr = _serialize_kwargs(normalized_kwargs)
    
    # 构建键字符串
    key_str = f"{func.__module__}.{func.__qualname__}:{params_repr}"
    
    # 生成哈希
    return hashlib.md5(key_str.encode(ENCODING)).hexdigest()


def _normalize_arguments(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    规范化函数参数
    
    将位置参数和关键字参数统一转换为关键字参数形式。
    这确保了 func(1, 2) 和 func(a=1, b=2) 产生相同的缓存键。
    
    Args:
        func: 被装饰的函数
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        规范化后的参数字典
    """
    try:
        # 获取函数签名
        sig = inspect.signature(func)
        
        # 绑定参数
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        # 返回规范化的参数字典
        return dict(bound.arguments)
    
    except (ValueError, TypeError) as e:
        # 如果绑定失败，回退到原始方式
        log_warning(f"参数规范化失败: {e}，使用原始参数")
        
        # 简单合并 args 和 kwargs
        result = {}
        
        # 将位置参数转换为字典（使用索引作为键）
        for i, arg in enumerate(args):
            result[f"_arg_{i}"] = arg
        
        # 添加关键字参数
        result.update(kwargs)
        
        return result


def _serialize_args(args: Tuple[Any, ...]) -> str:
    """
    序列化位置参数
    
    Args:
        args: 位置参数元组
        
    Returns:
        序列化后的字符串
    """
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        # JSON 序列化失败，回退到字符串表示
        log_warning("参数不是 JSON 可序列化的，使用 str() 表示")
        return str(args)


def _serialize_kwargs(kwargs: Dict[str, Any]) -> str:
    """
    序列化关键字参数
    
    Args:
        kwargs: 关键字参数字典
        
    Returns:
        序列化后的字符串
    """
    try:
        return json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        # JSON 序列化失败，回退到排序后的字符串表示
        log_warning("关键字参数不是 JSON 可序列化的，使用 str() 表示")
        return str(sorted(kwargs.items()))


# --- 缓存读写操作 ---

def _read_cache(
    cache_file: Path,
    cache_ttl: int,
    func_name: str
) -> Optional[Any]:
    """
    读取缓存文件
    
    Args:
        cache_file: 缓存文件路径
        cache_ttl: 缓存生存时间
        func_name: 函数名（用于日志）
        
    Returns:
        缓存的结果，如果缓存不存在或过期则返回 None
    """
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "r", encoding=ENCODING) as f:
            cache_data = json.load(f)
        
        # 验证缓存数据结构
        if not isinstance(cache_data, dict):
            log_warning(f"缓存数据格式无效: {cache_file}")
            return None
        
        # 检查是否包含必需的键
        if CACHE_TIMESTAMP_KEY not in cache_data or CACHE_RESULT_KEY not in cache_data:
            log_warning(f"缓存数据缺少必需字段: {cache_file}")
            return None
        
        # 检查是否过期
        timestamp = cache_data.get(CACHE_TIMESTAMP_KEY, 0)
        age = time.time() - timestamp
        
        if age <= cache_ttl:
            cache_key = cache_file.stem
            log_debug(
                f"函数 '{func_name}' 命中缓存 "
                f"(key: {cache_key[:8]}..., age: {age:.1f}s)"
            )
            return cache_data[CACHE_RESULT_KEY]
        else:
            log_debug(
                f"函数 '{func_name}' 的缓存已过期 "
                f"(age: {age:.1f}s > ttl: {cache_ttl}s)"
            )
            return None
    
    except json.JSONDecodeError as e:
        log_warning(f"解析缓存文件失败 {cache_file}: {e}")
        return None
    
    except (IOError, OSError) as e:
        log_warning(f"读取缓存文件失败 {cache_file}: {e}")
        return None
    
    except Exception as e:
        log_warning(f"读取缓存时发生未知错误: {e}")
        return None


def _write_cache(
    cache_file: Path,
    result: Any,
    func_name: str
) -> None:
    """
    写入缓存文件
    
    Args:
        cache_file: 缓存文件路径
        result: 要缓存的结果
        func_name: 函数名（用于日志）
    """
    cache_data = {
        CACHE_TIMESTAMP_KEY: time.time(),
        CACHE_RESULT_KEY: result
    }
    
    try:
        # 使用临时文件确保原子性写入
        temp_file = cache_file.with_suffix(".tmp")
        
        with open(temp_file, "w", encoding=ENCODING) as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        # 原子性重命名
        temp_file.replace(cache_file)
        
        log_debug(f"已为函数 '{func_name}' 创建缓存")
    
    except (TypeError, ValueError) as e:
        log_warning(f"函数 '{func_name}' 的返回值不是 JSON 可序列化的: {e}")
    
    except (IOError, OSError) as e:
        log_warning(f"写入缓存文件失败 {cache_file}: {e}")
    
    except Exception as e:
        log_warning(f"写入缓存时发生未知错误: {e}")


# --- 缓存管理工具 ---

def clear_cache(cache_dir: Optional[str] = None) -> int:
    """
    清除缓存目录中的所有文件
    
    Args:
        cache_dir: 缓存目录，None 则使用默认目录
        
    Returns:
        删除的文件数量
    """
    if cache_dir:
        dir_path = Path(cache_dir)
    else:
        dir_path = Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIR_NAME
    
    if not dir_path.exists():
        return 0
    
    count = 0
    try:
        for item in dir_path.rglob("*"):
            if item.is_file():
                try:
                    item.unlink()
                    count += 1
                except OSError:
                    pass
        
        log_debug(f"已清除 {count} 个缓存文件")
        return count
    
    except Exception as e:
        log_warning(f"清除缓存时发生错误: {e}")
        return count


def get_cache_info(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    获取缓存统计信息
    
    Args:
        cache_dir: 缓存目录，None 则使用默认目录
        
    Returns:
        包含缓存统计信息的字典
    """
    if cache_dir:
        dir_path = Path(cache_dir)
    else:
        dir_path = Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIR_NAME
    
    if not dir_path.exists():
        return {
            "exists": False,
            "total_files": 0,
            "total_size": 0,
            "path": str(dir_path)
        }
    
    total_files = 0
    total_size = 0
    
    try:
        for item in dir_path.rglob("*"):
            if item.is_file():
                total_files += 1
                total_size += item.stat().st_size
    except Exception as e:
        log_warning(f"获取缓存信息时发生错误: {e}")
    
    return {
        "exists": True,
        "total_files": total_files,
        "total_size": total_size,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "path": str(dir_path)
    }


# --- 测试代码 ---

def _test_basic_caching() -> None:
    """测试基本缓存功能"""
    print("\n[1] 测试基本缓存功能:")
    
    temp_cache_dir = Path(tempfile.gettempdir()) / "test_agno_cache_basic"
    call_count = {"value": 0}
    
    @cache_result(cache_dir=str(temp_cache_dir), cache_ttl=60)
    def sample_function(x: int, y: int) -> Dict[str, Any]:
        call_count["value"] += 1
        time.sleep(0.1)
        return {"sum": x + y, "product": x * y}
    
    try:
        # 第一次调用
        result1 = sample_function(2, 3)
        assert call_count["value"] == 1
        assert result1["sum"] == 5
        print("  ✓ 第一次调用执行成功")
        
        # 第二次调用（应命中缓存）
        result2 = sample_function(2, 3)
        assert call_count["value"] == 1
        assert result2 == result1
        print("  ✓ 第二次调用命中缓存")
        
        # 不同参数（应重新执行）
        result3 = sample_function(3, 4)
        assert call_count["value"] == 2
        assert result3["sum"] == 7
        print("  ✓ 不同参数重新执行")
        
    finally:
        # 清理
        import shutil
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)


def _test_cache_expiration() -> None:
    """测试缓存过期"""
    print("\n[2] 测试缓存过期:")
    
    temp_cache_dir = Path(tempfile.gettempdir()) / "test_agno_cache_ttl"
    call_count = {"value": 0}
    
    @cache_result(cache_dir=str(temp_cache_dir), cache_ttl=1)
    def quick_expire_function(x: int) -> int:
        call_count["value"] += 1
        return x * 2
    
    try:
        # 第一次调用
        result1 = quick_expire_function(5)
        assert result1 == 10
        assert call_count["value"] == 1
        print("  ✓ 创建缓存")
        
        # 等待过期
        time.sleep(1.1)
        
        # 再次调用（缓存应已过期）
        result2 = quick_expire_function(5)
        assert result2 == 10
        assert call_count["value"] == 2
        print("  ✓ 缓存过期后重新执行")
        
    finally:
        # 清理
        import shutil
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)


def _test_cache_disabled() -> None:
    """测试禁用缓存"""
    print("\n[3] 测试禁用缓存:")
    
    call_count = {"value": 0}
    
    @cache_result(enable_cache=False)
    def no_cache_function(x: int) -> int:
        call_count["value"] += 1
        return x * 2
    
    # 多次调用
    result1 = no_cache_function(5)
    result2 = no_cache_function(5)
    
    assert result1 == 10
    assert result2 == 10
    assert call_count["value"] == 2
    print("  ✓ 禁用缓存时每次都执行")


def _test_kwargs_caching() -> None:
    """测试关键字参数缓存"""
    print("\n[4] 测试关键字参数规范化:")
    
    temp_cache_dir = Path(tempfile.gettempdir()) / "test_agno_cache_kwargs"
    call_count = {"value": 0}
    
    @cache_result(cache_dir=str(temp_cache_dir))
    def kwargs_function(a: int, b: int = 10) -> int:
        call_count["value"] += 1
        return a + b
    
    try:
        # 使用位置参数和关键字参数
        result1 = kwargs_function(5, b=10)
        assert result1 == 15
        assert call_count["value"] == 1
        print("  ✓ 位置参数 + 关键字参数")
        
        # 全部使用关键字参数（应命中缓存，因为参数被规范化了）
        result2 = kwargs_function(a=5, b=10)
        assert result2 == 15
        assert call_count["value"] == 1, f"Expected 1 but got {call_count['value']}"
        print("  ✓ 全关键字参数命中缓存")
        
        # 使用默认值（应命中缓存）
        result3 = kwargs_function(5)
        assert result3 == 15
        assert call_count["value"] == 1
        print("  ✓ 使用默认值命中缓存")
        
        # 不同的参数值
        result4 = kwargs_function(5, 20)
        assert result4 == 25
        assert call_count["value"] == 2
        print("  ✓ 不同参数值重新执行")
        
    finally:
        # 清理
        import shutil
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)


def _test_cache_management() -> None:
    """测试缓存管理功能"""
    print("\n[5] 测试缓存管理:")
    
    temp_cache_dir = Path(tempfile.gettempdir()) / "test_agno_cache_mgmt"
    
    @cache_result(cache_dir=str(temp_cache_dir))
    def mgmt_function(x: int) -> int:
        return x * 2
    
    try:
        # 创建一些缓存
        mgmt_function(1)
        mgmt_function(2)
        mgmt_function(3)
        
        # 获取缓存信息
        info = get_cache_info(str(temp_cache_dir))
        assert info["exists"]
        assert info["total_files"] >= 3
        print(f"  ✓ 缓存信息: {info['total_files']} 文件, "
              f"{info['total_size_mb']} MB")
        
        # 清除缓存
        count = clear_cache(str(temp_cache_dir))
        assert count >= 3
        print(f"  ✓ 清除了 {count} 个缓存文件")
        
    finally:
        # 清理
        import shutil
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)


def _test_edge_cases() -> None:
    """测试边缘情况"""
    print("\n[6] 测试边缘情况:")
    
    passed = 0
    failed = 0
    
    temp_cache_dir = Path(tempfile.gettempdir()) / "test_agno_cache_edge"
    
    # 测试不可序列化的参数
    try:
        @cache_result(cache_dir=str(temp_cache_dir))
        def non_serializable_args(obj: object) -> str:
            return str(obj)
        
        result = non_serializable_args(object())
        assert isinstance(result, str)
        print("  ✓ 不可序列化参数处理正常")
        passed += 1
    except Exception as e:
        print(f"  ✗ 不可序列化参数测试失败: {e}")
        failed += 1
    
    # 测试不可序列化的返回值
    try:
        call_count = {"value": 0}
        
        @cache_result(cache_dir=str(temp_cache_dir))
        def non_serializable_result() -> object:
            call_count["value"] += 1
            return object()
        
        result1 = non_serializable_result()
        result2 = non_serializable_result()
        # 由于无法缓存，应该执行两次
        assert call_count["value"] == 2
        print("  ✓ 不可序列化返回值处理正常")
        passed += 1
    except Exception as e:
        print(f"  ✗ 不可序列化返回值测试失败: {e}")
        failed += 1
    
    # 清理
    import shutil
    if temp_cache_dir.exists():
        shutil.rmtree(temp_cache_dir)
    
    print(f"\n  边缘情况测试: {passed} 通过, {failed} 失败")


def _run_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("运行 agno/utils/cache.py 测试")
    print("=" * 60)
    
    _test_basic_caching()
    _test_cache_expiration()
    _test_cache_disabled()
    _test_kwargs_caching()
    _test_cache_management()
    _test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    _run_tests()