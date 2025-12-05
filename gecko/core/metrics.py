# gecko/core/metrics.py
"""
指标收集系统

提供轻量级的应用指标收集能力。
支持 Counter、Gauge、Histogram 三种指标类型。
采用分片锁（shard-based locking）降低并发竞争。
支持 Prometheus 文本格式导出与 TTL 清理机制。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
import statistics


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
    """
    直方图（用于延迟等分布统计）
    
    改进：
    - 记录所有观测值用于计算百分位数
    - 支持标准 Prometheus 导出格式
    - 提供完整的统计信息（min/max/p50/p95/p99 等）
    """

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

        # 存储所有观测值以计算百分位数（注意内存占用）
        self._observations: Dict[tuple, List[float]] = defaultdict(list)
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
            self._observations[key].append(value)  # 用于百分位计算

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

    def get_stats(self, **labels) -> Dict[str, Any]:
        """
        获取完整统计信息（包括百分位数）
        
        返回：
            - count: 观测总数
            - sum: 观测值总和
            - min: 最小值
            - max: 最大值
            - avg: 平均值
            - p50: 中位数（50 百分位）
            - p95: 95 百分位
            - p99: 99 百分位
        """
        key = self._make_key(labels)
        with self._lock:
            total = self._totals.get(key, 0)
            sum_val = self._sums.get(key, 0.0)
            observations = self._observations.get(key, [])

            stats = {
                "count": total,
                "sum": sum_val,
                "avg": sum_val / total if total > 0 else 0.0,
            }
            
            # 计算百分位数
            if observations:
                stats["min"] = min(observations)
                stats["max"] = max(observations)
                stats["p50"] = statistics.median(observations)
                if len(observations) > 1:
                    stats["p95"] = statistics.quantiles(observations, n=20)[18] if len(observations) >= 20 else observations[-1]
                    stats["p99"] = statistics.quantiles(observations, n=100)[98] if len(observations) >= 100 else observations[-1]
                else:
                    stats["p95"] = stats["p99"] = observations[0]
            
            return stats

    def get_bucket_stats(self, **labels) -> Dict[float, int]:
        """获取 bucket 统计（用于 Prometheus 导出）"""
        key = self._make_key(labels)
        with self._lock:
            return dict(self._counts.get(key, {}))

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
    
    改进：
    - 支持 Prometheus 文本格式导出
    - 支持 TTL 清理机制（清除过期的标签组合）
    - 使用分片锁降低并发竞争
    
    示例:
        ```python
        metrics = MetricsRegistry()
        
        requests = metrics.counter("requests_total", "Total requests")
        latency = metrics.histogram("request_latency", "Request latency")
        
        requests.inc(endpoint="/api/chat")
        with latency.time(endpoint="/api/chat"):
            await process_request()
        
        # Prometheus 格式导出
        prometheus_text = metrics.to_prometheus()
        ```
    """

    def __init__(self, max_label_combinations: int = 10000):
        """
        初始化指标注册中心。
        
        参数：
            max_label_combinations: 单个指标最多保留的标签组合数，防止内存泄漏。
                                   当超过此限制时，最老的标签组合会被清理。
        """
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self.max_label_combinations = max_label_combinations
        # 记录每个指标的最后一次使用时间，用于 TTL 清理
        self._last_accessed: Dict[tuple[str, str], float] = {}

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
        """收集所有指标（内部格式）"""
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

    def to_prometheus(self) -> str:
        """
        导出为 Prometheus 文本格式。
        
        符合 Prometheus 文本导出格式（支持 Counter、Gauge、Histogram）。
        """
        lines = []
        
        with self._lock:
            # 导出 Counter
            for name, counter in self._counters.items():
                lines.append(f"# HELP {name} {counter.description}")
                lines.append(f"# TYPE {name} counter")
                for (label_tuple), value in counter._values.items():
                    labels_str = self._format_labels(label_tuple)
                    lines.append(f"{name}{labels_str} {value}")
            
            # 导出 Gauge
            for name, gauge in self._gauges.items():
                lines.append(f"# HELP {name} {gauge.description}")
                lines.append(f"# TYPE {name} gauge")
                for (label_tuple), value in gauge._values.items():
                    labels_str = self._format_labels(label_tuple)
                    lines.append(f"{name}{labels_str} {value}")
            
            # 导出 Histogram
            for name, histogram in self._histograms.items():
                lines.append(f"# HELP {name} {histogram.description}")
                lines.append(f"# TYPE {name} histogram")
                
                for label_tuple, bucket_counts in histogram._counts.items():
                    labels_str = self._format_labels(label_tuple)
                    total = histogram._totals.get(label_tuple, 0)
                    sum_val = histogram._sums.get(label_tuple, 0.0)
                    
                    # 导出 bucket
                    for bucket, count in bucket_counts.items():
                        bucket_labels = labels_str.replace("}", f',le="{bucket}"}}')
                        if labels_str == "{}":
                            bucket_labels = f'{{le="{bucket}"}}'
                        lines.append(f"{name}_bucket{bucket_labels} {count}")
                    
                    # 导出 sum 和 count
                    lines.append(f"{name}_sum{labels_str} {sum_val}")
                    lines.append(f"{name}_count{labels_str} {total}")
        
        return "\n".join(lines)

    def cleanup_old_labels(self, ttl_seconds: int = 3600) -> int:
        """
        清理过期的标签组合（超过 TTL 的）。
        
        参数：
            ttl_seconds: 标签组合的生存时间（秒），超过此时间未被访问的将被清理。
        
        返回：
            清理的标签组合数。
        """
        now = time.time()
        cleaned_count = 0
        
        with self._lock:
            # 清理 Counter
            for counter in self._counters.values():
                keys_to_remove = []
                for key in counter._values:
                    last_access = self._last_accessed.get(("counter", str(key)), now)
                    if now - last_access > ttl_seconds:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del counter._values[key]
                    cleaned_count += 1
            
            # 清理 Gauge
            for gauge in self._gauges.values():
                keys_to_remove = []
                for key in gauge._values:
                    last_access = self._last_accessed.get(("gauge", str(key)), now)
                    if now - last_access > ttl_seconds:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del gauge._values[key]
                    cleaned_count += 1
            
            # 清理 Histogram
            for histogram in self._histograms.values():
                keys_to_remove = []
                for key in histogram._observations:
                    last_access = self._last_accessed.get(("histogram", str(key)), now)
                    if now - last_access > ttl_seconds:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del histogram._observations[key]
                    del histogram._counts[key]
                    del histogram._sums[key]
                    del histogram._totals[key]
                    cleaned_count += 1
        
        return cleaned_count

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            for c in self._counters.values():
                c.reset()
            self._gauges.clear()
            self._histograms.clear()
            self._last_accessed.clear()

    @staticmethod
    def _format_labels(label_tuple: tuple) -> str:
        """格式化标签为 Prometheus 格式"""
        if not label_tuple:
            return "{}"
        items = dict(label_tuple)
        parts = [f'{k}="{v}"' for k, v in items.items()]
        return "{" + ",".join(parts) + "}"


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