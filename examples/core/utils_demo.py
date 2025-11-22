# examples/utils_demo.py
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
    timing,
    chunk_list,
    deduplicate,
)


# 示例函数
def sync_function(x):
    return x * 2


async def async_function(x):
    await asyncio.sleep(0.1)
    return x * 3


@retry(max_attempts=3, delay=0.5)
async def unstable_function(fail_count=0):
    """模拟不稳定的函数"""
    if fail_count > 0:
        raise ValueError("Intentional failure")
    return "Success"


@timing
async def slow_function():
    """带计时的函数"""
    await asyncio.sleep(1)
    return "Done"


async def main():
    print("=== Gecko Utils 示例 ===\n")
    
    # 1. ensure_awaitable
    print("1. 统一异步/同步调用")
    result1 = await ensure_awaitable(sync_function, 5)
    result2 = await ensure_awaitable(async_function, 5)
    print(f"   同步函数结果: {result1}")
    print(f"   异步函数结果: {result2}\n")
    
    # 2. 重试机制
    print("2. 重试机制")
    try:
        result = await unstable_function(fail_count=0)
        print(f"   成功: {result}\n")
    except Exception as e:
        print(f"   失败: {e}\n")
    
    # 3. 数据转换
    print("3. 数据转换")
    class TestClass:
        def __init__(self):
            self.name = "test"
            self.value = 123
    
    obj = TestClass()
    data = safe_dict(obj)
    print(f"   对象转字典: {data}\n")
    
    # 4. 字典合并
    print("4. 字典合并")
    d1 = {"a": 1, "b": {"x": 1}}
    d2 = {"b": {"y": 2}, "c": 3}
    merged = merge_dicts(d1, d2, deep=True)
    print(f"   深度合并: {merged}\n")
    
    # 5. 字符串处理
    print("5. 字符串处理")
    long_text = "A" * 200
    truncated = truncate(long_text, max_length=50)
    print(f"   截断文本: {truncated}\n")
    
    print(f"   文件大小: {format_size(1048576)}")
    print(f"   时长: {format_duration(3665)}\n")
    
    # 6. 计时器
    print("6. 计时器")
    with Timer("测试操作") as t:
        await asyncio.sleep(0.5)
    print(f"   耗时: {t.elapsed:.2f}s\n")
    
    # 7. 计时装饰器
    print("7. 计时装饰器")
    result = await slow_function()
    print(f"   结果: {result}\n")
    
    # 8. 列表工具
    print("8. 列表工具")
    chunks = chunk_list([1, 2, 3, 4, 5, 6, 7], chunk_size=3)
    print(f"   分块: {chunks}")
    
    unique = deduplicate([1, 2, 2, 3, 1, 4, 3])
    print(f"   去重: {unique}\n")


if __name__ == "__main__":
    asyncio.run(main())