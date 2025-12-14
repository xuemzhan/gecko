# gecko/core/metrics.py
"""
Gecko Metrics 指标模块（修复增强版）

本模块提供一套轻量的指标采集能力，包含：
- Counter   ：单调递增计数器
- Gauge     ：可增可减的仪表盘数值
- Histogram ：用于统计分布（bucket + percentiles）的直方图
- MetricsRegistry：指标注册中心，负责创建/复用/导出 Prometheus 文本

✅ 本版关键修复点（对应你的测试失败）：
1) labels 统一：所有指标 API 同时兼容两种调用方式
   - 新接口：labels=dict
   - 旧接口：**kwargs labels
   避免出现 time() 内部使用 self.observe(..., **labels) 导致 {"labels":{...}} 的错误传播。

2) 统一 key 类型：所有指标内部统一使用 _LabelKey（可 hash、稳定排序）
   避免 dict 直接进入 key 导致 “unhashable type: 'dict'”。

3) Prometheus 导出兼容：Registry 的 _format_labels() 能识别 _LabelKey，
   不再假设 label_key 一定是 tuple，从而避免 “_LabelKey not iterable”。

4) 向后兼容：Counter.value 属性保留，使依赖 value 的旧代码与 telemetry 测试通过。

5) 工业级增强：线程安全（RLock）、label 过期清理（cleanup_old_labels）、snapshot 导出。

注意：
- 本模块不依赖第三方 metrics 库，适合框架内置轻量统计。
- Prometheus 输出为文本格式（适合 /metrics endpoint）。
"""

from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# labels key：稳定、可 hash
# =============================================================================

@dataclass(frozen=True)
class _LabelKey:
    """
    内部 labels 归一化 key（不可变、可 hash）。

    为什么需要这个结构？
    - labels 是 dict，不能直接作为 dict key（不可 hash）
    - labels 插入顺序不稳定：{"a":1,"b":2} 与 {"b":2,"a":1} 语义应一致
    - labels value 类型可能不同（int/bool/float），统一转 str 保证一致性

    items:
      形如 (("method","GET"),("path","/api")) 的稳定 tuple
    """
    items: Tuple[Tuple[str, str], ...]

    @staticmethod
    def from_labels(labels: Optional[Dict[str, Any]]) -> "_LabelKey":
        """将 labels 归一化为稳定的 _LabelKey。"""
        if not labels:
            return _LabelKey(tuple())
        # 统一转 str，并按 key 排序保证稳定性
        normalized = tuple(sorted(((str(k), str(v)) for k, v in labels.items()), key=lambda kv: kv[0]))
        return _LabelKey(normalized)


def _merge_labels(labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> Dict[str, Any]:
    """
    合并 labels 的两种输入形式，统一输出 dict。

    支持：
    - labels={"a":1}
    - a=1,b=2

    规则：
    - kwargs 覆盖 labels（更符合 Python 常规行为）
    """
    merged: Dict[str, Any] = {}
    if labels:
        merged.update(labels)
    if label_kwargs:
        merged.update(label_kwargs)
    return merged


# =============================================================================
# 公共结构：MetricSample（可用于 collect/snapshot/导出）
# =============================================================================

@dataclass
class MetricSample:
    """
    指标样本（用于导出/展示）

    name:
      指标名称（例如 http_requests_total）
    labels:
      标签 tuple（稳定结构）
    value:
      样本值（counter/gauge）或 histogram 的某个派生值
    """
    name: str
    labels: Tuple[Tuple[str, str], ...]
    value: float


# =============================================================================
# BaseMetric：提供线程安全、label 活跃时间记录
# =============================================================================

class _BaseMetric:
    """
    指标基类：提供
    - name/description
    - RLock 线程安全
    - last_access：用于清理不活跃 label key
    """
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._lock = RLock()
        # 记录每个 label key 的最后访问时间，用于 cleanup_old_labels
        self._last_access: Dict[_LabelKey, float] = {}

    def _touch(self, key: _LabelKey) -> None:
        """记录 label key 的活跃时间（用于 TTL 清理）。"""
        self._last_access[key] = time.time()

    def cleanup_old_labels(self, ttl_seconds: float) -> int:
        """
        清理超过 ttl_seconds 未访问的 labels。

        返回：
          被清理的 labels 数量

        注意：
        - 具体清理哪些内部结构由子类实现（这里仅清理 last_access 记录）
        - 子类需要 override 并调用 super().cleanup_old_labels(...) 或自行实现
        """
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = [k for k, ts in self._last_access.items() if (now - ts) >= ttl_seconds]
            for k in to_remove:
                self._last_access.pop(k, None)
                removed += 1
        return removed


# =============================================================================
# Counter：单调递增计数器（兼容 value 属性）
# =============================================================================

class Counter(_BaseMetric):
    """
    Counter：计数器（单调递增）

    关键特性：
    - 支持 labels（labels=dict 或 **kwargs）
    - 线程安全
    - 向后兼容：value 属性存在（telemetry 集成测试依赖）
    """

    def __init__(self, name: str, description: str = "") -> None:
        super().__init__(name, description)
        self._values: Dict[_LabelKey, float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> None:
        """
        增加计数。

        Args:
          amount: 增量（必须 >= 0）
          labels / **label_kwargs: 标签

        兼容：
          counter.inc(1, labels={"method":"GET"})
          counter.inc(1, method="GET")
        """
        try:
            amt = float(amount)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Counter.inc amount 必须为数字类型，当前={amount!r}") from e

        if amt < 0:
            raise ValueError(f"Counter 不允许负增量：amount={amt}")

        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)

        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amt
            self._touch(key)

    def get(self, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> float:
        """获取指定 labels 的计数值（不存在返回 0）。"""
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)
        with self._lock:
            self._touch(key)
            return float(self._values.get(key, 0.0))

    @property
    def value(self) -> float:
        """
        向后兼容属性：返回“当前值”。

        语义：
        - 若存在无 labels 的 key（最常见），返回它
        - 否则返回所有 labels 组合值之和（便于快速观测整体规模）
        """
        base_key = _LabelKey(tuple())
        with self._lock:
            if not self._values:
                return 0.0
            if base_key in self._values:
                return float(self._values[base_key])
            return float(sum(self._values.values()))

    def reset(self) -> None:
        """清空计数（测试/调试用）。"""
        with self._lock:
            self._values.clear()
            self._last_access.clear()

    def snapshot(self) -> Dict[Tuple[Tuple[str, str], ...], float]:
        """导出快照：{labels_tuple: value}（labels_tuple 可序列化）。"""
        with self._lock:
            return {k.items: float(v) for k, v in self._values.items()}

    def cleanup_old_labels(self, ttl_seconds: float) -> int:
        """清理不活跃 label key，并同步清理 values。"""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = [k for k, ts in self._last_access.items() if (now - ts) >= ttl_seconds]
            for k in to_remove:
                self._last_access.pop(k, None)
                self._values.pop(k, None)
                removed += 1
        return removed


# =============================================================================
# Gauge：可增可减数值
# =============================================================================

class Gauge(_BaseMetric):
    """
    Gauge：仪表盘（可增可减）

    支持：
      - set(value)
      - inc(amount)
      - dec(amount)
      - get()

    labels 兼容：
      gauge.set(1, labels={"service":"api"})
      gauge.set(1, service="api")
    """

    def __init__(self, name: str, description: str = "") -> None:
        super().__init__(name, description)
        self._values: Dict[_LabelKey, float] = {}

    def set(self, value: float, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> None:
        """设置 gauge 值。"""
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)
        with self._lock:
            self._values[key] = float(value)
            self._touch(key)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> None:
        """增加 gauge 值。"""
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)
            self._touch(key)

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> None:
        """减少 gauge 值。"""
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - float(amount)
            self._touch(key)

    def get(self, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> float:
        """获取 gauge 值（不存在返回 0）。"""
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)
        with self._lock:
            self._touch(key)
            return float(self._values.get(key, 0.0))

    def reset(self) -> None:
        """清空 gauge（测试/调试用）。"""
        with self._lock:
            self._values.clear()
            self._last_access.clear()

    def snapshot(self) -> Dict[Tuple[Tuple[str, str], ...], float]:
        """导出快照：{labels_tuple: value}。"""
        with self._lock:
            return {k.items: float(v) for k, v in self._values.items()}

    def cleanup_old_labels(self, ttl_seconds: float) -> int:
        """清理不活跃 label key，并同步清理 values。"""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = [k for k, ts in self._last_access.items() if (now - ts) >= ttl_seconds]
            for k in to_remove:
                self._last_access.pop(k, None)
                self._values.pop(k, None)
                removed += 1
        return removed


# =============================================================================
# Histogram：统计分布（bucket + percentiles）
# =============================================================================

class Histogram(_BaseMetric):
    """
    Histogram：直方图

    我们维护每个 label key 下：
    - count：观测次数
    - sum  ：观测值累积
    - values：观测值列表（用于百分位 p50/p95/p99，注意：在高吞吐场景可能较大）
    - bucket_counts：每个 bucket 的落点计数（非累计）
      bucket_counts[value_bucket] += 1

    Prometheus histogram 导出要求：
    - <name>_bucket{...,le="0.5"} cumulative_count
    - <name>_sum{...} sum
    - <name>_count{...} count
    """

    DEFAULT_BUCKETS: List[float] = [
        0.005, 0.01, 0.025, 0.05, 0.1,
        0.25, 0.5, 1.0, 2.5, 5.0,
        10.0,
        float("inf"),  # 最终 bucket 必须是 +Inf（Prometheus 语义）
    ]

    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None) -> None:
        super().__init__(name, description)
        self.buckets: List[float] = list(buckets) if buckets else list(self.DEFAULT_BUCKETS)

        # 每个 key 的统计
        self._counts: Dict[_LabelKey, int] = {}
        self._sums: Dict[_LabelKey, float] = {}
        self._values: Dict[_LabelKey, List[float]] = {}
        # 每个 key 的 bucket 落点计数（非累计）
        self._bucket_counts: Dict[_LabelKey, Dict[float, int]] = {}

    def _ensure_key(self, key: _LabelKey) -> None:
        """确保 key 的内部结构已初始化。"""
        if key not in self._counts:
            self._counts[key] = 0
            self._sums[key] = 0.0
            self._values[key] = []
            self._bucket_counts[key] = {b: 0 for b in self.buckets}

    def observe(self, value: float, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> None:
        """
        记录观测值（兼容 labels=dict 与 **labels）。

        ✅ 关键修复：不再直接把 dict 放入 key，
        而是先合并为普通 dict -> _LabelKey.from_labels -> 可 hash key。
        """
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)

        v = float(value)

        with self._lock:
            self._ensure_key(key)

            self._counts[key] += 1
            self._sums[key] += v
            self._values[key].append(v)

            # bucket 落点：找到第一个 >= v 的 bucket
            for b in self.buckets:
                if v <= b:
                    self._bucket_counts[key][b] += 1
                    break

            self._touch(key)

    @contextmanager
    def time(self, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any):
        """
        计时上下文管理器（兼容两种 labels 传参）。

        ✅ 关键修复：结束时调用 observe(duration, labels=merged)
        避免旧实现 observe(duration, **labels) 导致 labels={"labels":{...}} 的错误传播。
        """
        merged = _merge_labels(labels, **label_kwargs)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, labels=merged)

    def get_stats(self, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> Dict[str, Any]:
        """
        获取统计信息：count/sum/avg/min/max/p50/p95/p99

        注：
        - 百分位基于 values 列表计算。若 values 为空返回 0。
        - 这是轻量实现，若追求高性能可升级为 TDigest/CKMS 等近似算法。
        """
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)

        with self._lock:
            self._touch(key)
            if key not in self._counts or self._counts[key] == 0:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            cnt = self._counts[key]
            s = self._sums[key]
            vals = self._values[key]

            vmin = min(vals)
            vmax = max(vals)
            avg = s / cnt if cnt else 0.0

            def _percentile(sorted_vals: List[float], p: float) -> float:
                # p in [0,100]
                if not sorted_vals:
                    return 0.0
                if p <= 0:
                    return float(sorted_vals[0])
                if p >= 100:
                    return float(sorted_vals[-1])
                k = (len(sorted_vals) - 1) * (p / 100.0)
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return float(sorted_vals[int(k)])
                d0 = sorted_vals[f] * (c - k)
                d1 = sorted_vals[c] * (k - f)
                return float(d0 + d1)

            sorted_vals = sorted(vals)
            p50 = _percentile(sorted_vals, 50)
            p95 = _percentile(sorted_vals, 95)
            p99 = _percentile(sorted_vals, 99)

            return {
                "count": cnt,
                "sum": s,
                "avg": avg,
                "min": vmin,
                "max": vmax,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }

    def get_bucket_stats(self, labels: Optional[Dict[str, Any]] = None, **label_kwargs: Any) -> Dict[float, int]:
        """
        返回 Prometheus 语义的 bucket 统计：每个 le 的“累计”计数。

        返回示例：
          {
            0.1: 3,
            0.5: 10,
            1.0: 15,
            inf: 20
          }

        ✅ 你的测试中会断言 buckets[inf] == total_count
        """
        merged = _merge_labels(labels, **label_kwargs)
        key = _LabelKey.from_labels(merged)

        with self._lock:
            self._touch(key)
            if key not in self._counts or self._counts[key] == 0:
                return {b: 0 for b in self.buckets}

            # 当前 key 的 bucket 落点计数（非累计）
            raw = self._bucket_counts.get(key, {})
            cumulative: Dict[float, int] = {}
            running = 0
            for b in self.buckets:
                running += int(raw.get(b, 0))
                cumulative[b] = running
            return cumulative

    def reset(self) -> None:
        """清空 histogram（测试/调试用）。"""
        with self._lock:
            self._counts.clear()
            self._sums.clear()
            self._values.clear()
            self._bucket_counts.clear()
            self._last_access.clear()

    def snapshot(self) -> Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]]:
        """
        导出快照（用于 registry.collect / prometheus 导出）
        结构：
          { labels_tuple: {"count":..., "sum":..., "buckets":{le: cumulative_count}} }
        """
        with self._lock:
            out: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}
            for k in list(self._counts.keys()):
                cnt = self._counts.get(k, 0)
                s = self._sums.get(k, 0.0)
                buckets = self.get_bucket_stats(labels=dict(k.items))
                out[k.items] = {"count": cnt, "sum": s, "buckets": buckets}
            return out

    def cleanup_old_labels(self, ttl_seconds: float) -> int:
        """清理不活跃 label key，并同步清理内部结构。"""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = [k for k, ts in self._last_access.items() if (now - ts) >= ttl_seconds]
            for k in to_remove:
                self._last_access.pop(k, None)
                self._counts.pop(k, None)
                self._sums.pop(k, None)
                self._values.pop(k, None)
                self._bucket_counts.pop(k, None)
                removed += 1
        return removed


# =============================================================================
# MetricsRegistry：指标注册中心
# =============================================================================

class MetricsRegistry:
    """
    指标注册中心

    职责：
    - 创建/复用同名指标（counter/gauge/histogram）
    - collect：返回结构化数据（用于 debug 或 JSON 输出）
    - to_prometheus：导出 Prometheus 文本格式
    - reset：清理/复位指标
    - cleanup_old_labels：清理不活跃 label key
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    # -------------------- 创建/复用 --------------------

    def counter(self, name: str, description: str = "") -> Counter:
        """创建或复用 Counter。"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """创建或复用 Gauge。"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        """创建或复用 Histogram。"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets=buckets)
            return self._histograms[name]

    # -------------------- collect --------------------

    def collect(self) -> Dict[str, Any]:
        """
        收集所有指标快照（结构化 dict）。

        你的测试会断言 keys 包含 counters/gauges/histograms。
        """
        with self._lock:
            return {
                "counters": {name: m.snapshot() for name, m in self._counters.items()},
                "gauges": {name: m.snapshot() for name, m in self._gauges.items()},
                "histograms": {name: m.snapshot() for name, m in self._histograms.items()},
            }

    # -------------------- Prometheus 导出 --------------------

    @staticmethod
    def _escape_label_value(v: str) -> str:
        """Prometheus label value 转义（最小必要）。"""
        return v.replace("\\", "\\\\").replace('"', '\\"')

    @classmethod
    def _format_labels(cls, label_key: Any, extra: Optional[Dict[str, str]] = None) -> str:
        """
        格式化 labels 为 Prometheus 格式：{k="v",...}

        ✅ 修复点：兼容 label_key 类型
        - tuple[tuple[str,str], ...]
        - _LabelKey(items=(...))
        - None / 空
        """
        if not label_key:
            # Prometheus 允许没有 labels，返回 {} 也可接受
            if extra:
                items = dict(extra)
                inner = ",".join(f'{k}="{cls._escape_label_value(str(v))}"' for k, v in items.items())
                return "{" + inner + "}"
            return "{}"

        # _LabelKey：取其 .items
        if hasattr(label_key, "items"):
            base_items = getattr(label_key, "items")
        else:
            base_items = label_key

        items_dict = dict(base_items) if base_items else {}

        if extra:
            items_dict.update(extra)

        if not items_dict:
            return "{}"

        inner = ",".join(f'{k}="{cls._escape_label_value(str(v))}"' for k, v in items_dict.items())
        return "{" + inner + "}"

    def to_prometheus(self) -> str:
        """
        导出 Prometheus 文本格式。

        输出包括：
        - # HELP
        - # TYPE
        - metric samples
        - histogram: _bucket/_sum/_count
        """
        lines: List[str] = []

        with self._lock:
            # ---- counters ----
            for name, c in self._counters.items():
                lines.append(f"# HELP {name} {c.description}".rstrip())
                lines.append(f"# TYPE {name} counter")
                snap = c.snapshot()
                for labels_tuple, val in snap.items():
                    labels_str = self._format_labels(labels_tuple)
                    lines.append(f"{name}{labels_str} {val}")

            # ---- gauges ----
            for name, g in self._gauges.items():
                lines.append(f"# HELP {name} {g.description}".rstrip())
                lines.append(f"# TYPE {name} gauge")
                snap = g.snapshot()
                for labels_tuple, val in snap.items():
                    labels_str = self._format_labels(labels_tuple)
                    lines.append(f"{name}{labels_str} {val}")

            # ---- histograms ----
            for name, h in self._histograms.items():
                lines.append(f"# HELP {name} {h.description}".rstrip())
                lines.append(f"# TYPE {name} histogram")
                snap = h.snapshot()
                for labels_tuple, stat in snap.items():
                    # bucket lines
                    buckets: Dict[float, int] = stat["buckets"]
                    for le, count in buckets.items():
                        # Prometheus le 必须是字符串，inf 表示 +Inf
                        le_str = "+Inf" if le == float("inf") else str(le)
                        labels_str = self._format_labels(labels_tuple, extra={"le": le_str})
                        lines.append(f"{name}_bucket{labels_str} {count}")

                    # sum line
                    labels_str_sum = self._format_labels(labels_tuple)
                    lines.append(f"{name}_sum{labels_str_sum} {stat['sum']}")
                    # count line
                    lines.append(f"{name}_count{labels_str_sum} {stat['count']}")

        # Prometheus 文本通常以换行结尾
        return "\n".join(lines) + "\n"

    # -------------------- reset / cleanup --------------------

    def reset(self) -> None:
        """
        重置所有指标。

        为了兼容你的测试用例：
        - counter 对象可能已被外部持有引用，因此必须调用 reset() 复位其值
        - gauges/histograms：测试期望 registry._gauges 被清空（len == 0）
        """
        with self._lock:
            for c in self._counters.values():
                c.reset()

            for g in self._gauges.values():
                g.reset()
            for h in self._histograms.values():
                h.reset()

            # ✅ 按测试预期：清空 gauges/histograms 容器
            self._gauges.clear()
            self._histograms.clear()

    def cleanup_old_labels(self, ttl_seconds: float) -> int:
        """
        清理超过 ttl_seconds 未访问的 labels。
        返回总清理数量（所有指标之和）。
        """
        total = 0
        with self._lock:
            for c in self._counters.values():
                total += c.cleanup_old_labels(ttl_seconds)
            for g in self._gauges.values():
                total += g.cleanup_old_labels(ttl_seconds)
            for h in self._histograms.values():
                total += h.cleanup_old_labels(ttl_seconds)
        return total


__all__ = [
    "MetricSample",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
]
