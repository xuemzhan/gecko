# gecko/core/metrics.py
"""
指标收集系统

提供轻量级的应用指标收集能力。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class MetricSample:
    """指标样本"""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """计数器（只增不减）"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加计数"""
        key = self._make_key(labels)
        with self._lock:
            self._values[key] += value

    def get(self, **labels) -> float:
        """获取当前值"""
        key = self._make_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        """重置"""
        with self._lock:
            self._values.clear()

    @staticmethod
    def _make_key(labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))


class Gauge:
    """仪表（可增可减）"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels) -> None:
        """设置值"""
        key = self._make_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加"""
        key = self._make_key(labels)
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, **labels) -> None:
        """减少"""
        key = self._make_key(labels)
        with self._lock:
            self._values[key] -= value

    def get(self, **labels) -> float:
        """获取值"""
        key = self._make_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    @staticmethod
    def _make_key(labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))


class Histogram:
    """直方图（用于延迟等分布统计）"""

    DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: tuple = DEFAULT_BUCKETS
    ):
        self.name = name
        self.description = description
        self.buckets = tuple(sorted(buckets))

        self._counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """记录观测值"""
        key = self._make_key(labels)
        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

    def get_stats(self, **labels) -> Dict[str, float]:
        """获取统计"""
        key = self._make_key(labels)
        with self._lock:
            total = self._totals.get(key, 0)
            sum_val = self._sums.get(key, 0.0)

            return {
                "count": total,
                "sum": sum_val,
                "avg": sum_val / total if total > 0 else 0.0,
            }

    @contextmanager
    def time(self, **labels) -> Iterator[None]:
        """计时上下文管理器"""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start, **labels)

    @staticmethod
    def _make_key(labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))


class MetricsRegistry:
    """
    指标注册中心
    
    示例:
        ```python
        metrics = MetricsRegistry()
        
        requests = metrics.counter("requests_total", "Total requests")
        latency = metrics.histogram("request_latency", "Request latency")
        
        requests.inc(endpoint="/api/chat")
        with latency.time(endpoint="/api/chat"):
            await process_request()
        ```
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        """获取或创建计数器"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """获取或创建仪表"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: tuple = Histogram.DEFAULT_BUCKETS
    ) -> Histogram:
        """获取或创建直方图"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]

    def collect(self) -> Dict[str, Any]:
        """收集所有指标"""
        with self._lock:
            return {
                "counters": {
                    name: dict(c._values) for name, c in self._counters.items()
                },
                "gauges": {
                    name: dict(g._values) for name, g in self._gauges.items()
                },
                "histograms": {
                    name: h.get_stats() for name, h in self._histograms.items()
                },
            }

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            for c in self._counters.values():
                c.reset()
            self._gauges.clear()
            self._histograms.clear()


# 全局实例
_registry: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """获取全局指标注册中心"""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "MetricSample",
    "get_metrics",
]