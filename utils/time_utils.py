# agno/utils/time_utils.py

"""
时间和计时工具模块

该模块提供了与时间和性能测量相关的实用工具。
主要功能包括：
- 获取当前本地和UTC时间的函数。
- 一个用于精确测量代码块执行时间的 Timer 类。
"""

from datetime import datetime, timezone
from time import perf_counter
from typing import Optional, Any


# --- 日期时间函数 ---

def current_datetime() -> datetime:
    """获取当前本地日期和时间。"""
    return datetime.now()


def current_datetime_utc() -> datetime:
    """获取当前带有时区信息的 UTC 时间（推荐用于日志和存储）。"""
    return datetime.now(timezone.utc)


def current_datetime_utc_str() -> str:
    """获取当前 UTC 时间的 ISO 8601 格式字符串（不含微秒和时区）。"""
    return current_datetime_utc().strftime("%Y-%m-%dT%H:%M:%S")


# --- 性能计时器 ---

class Timer:
    """
    高精度代码执行时间测量工具。

    基于 `time.perf_counter()`，支持手动控制或作为上下文管理器使用。

    示例:
        # 手动模式
        t = Timer()
        t.start()
        ... # 耗时操作
        t.stop()
        print(f"耗时: {t.elapsed:.4f} 秒")

        # 上下文管理器模式
        with Timer() as t:
            ... # 耗时操作
        print(f"耗时: {t.elapsed:.4f} 秒")
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self._elapsed_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        """
        返回经过的时间（秒）。
        - 若已停止：返回固定耗时；
        - 若正在运行：返回实时耗时；
        - 若未启动：返回 0.0。
        """
        if self._elapsed_time is not None:
            return self._elapsed_time
        if self.start_time is not None:
            return perf_counter() - self.start_time
        return 0.0

    def start(self) -> None:
        """启动计时器。若已启动，则重置并重新开始。"""
        self.start_time = perf_counter()
        self._elapsed_time = None

    def stop(self) -> float:
        """停止计时器并返回总耗时。若未启动，返回 0.0。"""
        if self.start_time is None:
            return 0.0
        self._elapsed_time = perf_counter() - self.start_time
        return self._elapsed_time

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# --- 测试代码 ---
if __name__ == "__main__":
    import time

    print("--- 正在运行 agno/utils/time_utils.py 的测试代码 ---")

    # 1. 日期时间函数
    print("\n[1] 日期时间函数:")
    print(f"  本地时间: {current_datetime()}")
    print(f"  UTC时间: {current_datetime_utc()}")
    print(f"  UTC字符串: {current_datetime_utc_str()}")

    # 2. Timer 测试
    print("\n[2] Timer 测试:")

    # 手动模式
    print("  手动模式:")
    t1 = Timer()
    t1.start()
    time.sleep(0.2)
    print(f"    运行中耗时: {t1.elapsed:.4f} 秒")
    t1.stop()
    print(f"    停止后耗时: {t1.elapsed:.4f} 秒")

    # 上下文管理器
    print("\n  上下文管理器模式:")
    with Timer() as t2:
        time.sleep(0.3)
    print(f"    耗时: {t2.elapsed:.4f} 秒")

    # 未启动的 Timer
    print(f"\n  未启动 Timer 的 elapsed: {Timer().elapsed}")

    print("\n--- 测试结束 ---")

    