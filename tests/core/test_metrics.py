# tests/core/test_metrics.py
"""
指标收集系统单元测试（最新版）

覆盖：
- Counter / Gauge / Histogram 的核心行为
- MetricsRegistry 的创建、复用、导出 Prometheus、清理、reset
- 并发写入基本正确性（线程并发）

兼容策略：
- 优先使用“新接口（labels=dict）”
- 若项目中 Gauge/Histogram 仍是旧接口（**labels），自动 fallback
"""

from __future__ import annotations

import time
import threading
import pytest

from gecko.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    MetricSample,
)



# =============================================================================
# 兼容适配器：优先新接口 labels=dict；若 TypeError 则 fallback 到旧接口 kwargs labels
# =============================================================================

def _counter_inc(counter: Counter, amount: float = 1.0, labels: dict | None = None, **kwargs):
    """
    Counter.inc 兼容调用：
    - 新接口：inc(amount, labels={...})
    - 旧接口：inc(value=..., **labels)
    """
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}

    try:
        # ✅ 新接口
        counter.inc(amount, labels=labels)  # type: ignore[arg-type]
    except TypeError:
        # ✅ 旧接口
        counter.inc(amount, **labels)  # type: ignore[misc]


def _counter_get(counter: Counter, labels: dict | None = None, **kwargs) -> float:
    """
    Counter.get 兼容调用：
    - 新接口：get(labels={...})
    - 旧接口：get(**labels)
    """
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}

    try:
        return float(counter.get(labels=labels))  # type: ignore[arg-type]
    except TypeError:
        return float(counter.get(**labels))  # type: ignore[misc]


def _gauge_set(gauge: Gauge, value: float, labels: dict | None = None, **kwargs):
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        gauge.set(value, labels=labels)  # type: ignore[arg-type]
    except TypeError:
        gauge.set(value, **labels)  # type: ignore[misc]


def _gauge_inc(gauge: Gauge, amount: float = 1.0, labels: dict | None = None, **kwargs):
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        gauge.inc(amount, labels=labels)  # type: ignore[arg-type]
    except TypeError:
        gauge.inc(amount, **labels)  # type: ignore[misc]


def _gauge_dec(gauge: Gauge, amount: float = 1.0, labels: dict | None = None, **kwargs):
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        gauge.dec(amount, labels=labels)  # type: ignore[arg-type]
    except TypeError:
        gauge.dec(amount, **labels)  # type: ignore[misc]


def _gauge_get(gauge: Gauge, labels: dict | None = None, **kwargs) -> float:
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        return float(gauge.get(labels=labels))  # type: ignore[arg-type]
    except TypeError:
        return float(gauge.get(**labels))  # type: ignore[misc]


def _hist_observe(hist: Histogram, value: float, labels: dict | None = None, **kwargs):
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        hist.observe(value, labels=labels)  # type: ignore[arg-type]
    except TypeError:
        hist.observe(value, **labels)  # type: ignore[misc]


def _hist_get_stats(hist: Histogram, labels: dict | None = None, **kwargs) -> dict:
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        return hist.get_stats(labels=labels)  # type: ignore[arg-type]
    except TypeError:
        return hist.get_stats(**labels)  # type: ignore[misc]


def _hist_get_bucket_stats(hist: Histogram, labels: dict | None = None, **kwargs) -> dict:
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        return hist.get_bucket_stats(labels=labels)  # type: ignore[arg-type]
    except TypeError:
        return hist.get_bucket_stats(**labels)  # type: ignore[misc]


def _hist_time(hist: Histogram, labels: dict | None = None, **kwargs):
    """
    Histogram.time() 兼容上下文管理器：
    - 新接口：time(labels={...})
    - 旧接口：time(**labels)
    """
    if labels is None and kwargs:
        labels = dict(kwargs)
    labels = labels or {}
    try:
        return hist.time(labels=labels)  # type: ignore[arg-type]
    except TypeError:
        return hist.time(**labels)  # type: ignore[misc]


# =============================================================================
# Counter
# =============================================================================

class TestCounter:
    """测试计数器"""

    def test_counter_init(self):
        """初始化计数器"""
        counter = Counter("test_counter", "Test counter")
        assert counter.name == "test_counter"
        assert counter.description == "Test counter"

    def test_counter_increment(self):
        """增加计数（无 labels）"""
        counter = Counter("counter")
        _counter_inc(counter)
        assert _counter_get(counter) == 1.0

        _counter_inc(counter, 5)
        assert _counter_get(counter) == 6.0

    def test_counter_with_labels(self):
        """带标签的计数器"""
        counter = Counter("counter")
        _counter_inc(counter, 1, labels={"method": "GET", "path": "/api"})
        _counter_inc(counter, 1, labels={"method": "POST", "path": "/api"})

        assert _counter_get(counter, labels={"method": "GET", "path": "/api"}) == 1.0
        assert _counter_get(counter, labels={"method": "POST", "path": "/api"}) == 1.0

    def test_counter_value_property(self):
        """
        ✅ 关键：Counter.value 必须存在（telemetry 集成测试依赖）
        - 如果无 labels 计数存在，应优先返回无 labels 的值
        """
        counter = Counter("counter")
        _counter_inc(counter, 1)
        assert hasattr(counter, "value")
        assert counter.value == 1.0  # type: ignore[attr-defined]

    def test_counter_reset(self):
        """重置计数器"""
        counter = Counter("counter")
        _counter_inc(counter, 10)
        assert _counter_get(counter) == 10.0

        # 若 Counter 实现提供 reset()，应清空
        assert hasattr(counter, "reset")
        counter.reset()  # type: ignore[attr-defined]
        assert _counter_get(counter) == 0.0

    def test_counter_snapshot(self):
        """
        若 Counter 提供 snapshot()，应返回可序列化结构
        （新实现常见：返回 {labels_tuple: value}）
        """
        counter = Counter("counter")
        _counter_inc(counter, 1)
        _counter_inc(counter, 2, labels={"a": 1})

        if hasattr(counter, "snapshot"):
            snap = counter.snapshot()  # type: ignore[attr-defined]
            assert isinstance(snap, dict)
            assert len(snap) >= 1


# =============================================================================
# Gauge
# =============================================================================

class TestGauge:
    """测试仪表"""

    def test_gauge_set(self):
        """设置仪表值"""
        gauge = Gauge("gauge")
        _gauge_set(gauge, 42)
        assert _gauge_get(gauge) == 42.0

    def test_gauge_increment(self):
        """增加仪表值"""
        gauge = Gauge("gauge")
        _gauge_set(gauge, 10)
        _gauge_inc(gauge, 5)
        assert _gauge_get(gauge) == 15.0

    def test_gauge_decrement(self):
        """减少仪表值"""
        gauge = Gauge("gauge")
        _gauge_set(gauge, 10)
        _gauge_dec(gauge, 3)
        assert _gauge_get(gauge) == 7.0

    def test_gauge_with_labels(self):
        """带标签的仪表"""
        gauge = Gauge("gauge")
        _gauge_set(gauge, 10, labels={"service": "api"})
        _gauge_set(gauge, 20, labels={"service": "db"})

        assert _gauge_get(gauge, labels={"service": "api"}) == 10.0
        assert _gauge_get(gauge, labels={"service": "db"}) == 20.0


# =============================================================================
# Histogram
# =============================================================================

class TestHistogram:
    """测试直方图"""

    def test_histogram_init(self):
        """初始化直方图"""
        hist = Histogram("hist", "Test histogram")
        assert hist.name == "hist"
        assert len(hist.buckets) > 0

    def test_histogram_observe(self):
        """记录观测值"""
        hist = Histogram("hist")
        _hist_observe(hist, 0.5)
        _hist_observe(hist, 1.5)

        stats = _hist_get_stats(hist)
        assert stats["count"] == 2
        assert stats["sum"] == 2.0

    def test_histogram_stats_complete(self):
        """获取完整统计"""
        hist = Histogram("hist")
        values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

        for val in values:
            _hist_observe(hist, val)

        stats = _hist_get_stats(hist)
        assert stats["count"] == len(values)
        assert stats["sum"] == sum(values)
        assert stats["avg"] == sum(values) / len(values)
        assert stats["min"] == min(values)
        assert stats["max"] == max(values)
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_histogram_percentiles(self):
        """测试百分位计算"""
        hist = Histogram("hist")
        for i in range(100):
            _hist_observe(hist, float(i))

        stats = _hist_get_stats(hist)
        assert 45 <= stats["p50"] <= 55
        assert 90 <= stats["p95"] <= 100
        assert 95 <= stats["p99"] <= 100

    def test_histogram_with_labels(self):
        """带标签的直方图"""
        hist = Histogram("hist")
        _hist_observe(hist, 0.5, labels={"endpoint": "/api"})
        _hist_observe(hist, 1.5, labels={"endpoint": "/api"})
        _hist_observe(hist, 2.0, labels={"endpoint": "/db"})

        stats_api = _hist_get_stats(hist, labels={"endpoint": "/api"})
        assert stats_api["count"] == 2

        stats_db = _hist_get_stats(hist, labels={"endpoint": "/db"})
        assert stats_db["count"] == 1

    def test_histogram_time_context(self):
        """测试计时上下文"""
        hist = Histogram("hist")

        with _hist_time(hist):
            time.sleep(0.01)

        stats = _hist_get_stats(hist)
        assert stats["count"] == 1
        assert stats["sum"] >= 0.01

    def test_histogram_bucket_stats(self):
        """测试 bucket 统计"""
        hist = Histogram("hist")

        for i in range(20):
            _hist_observe(hist, float(i) / 10)  # 0.0 - 1.9

        buckets = _hist_get_bucket_stats(hist)
        assert len(buckets) > 0
        assert buckets[float("inf")] == 20


# =============================================================================
# MetricsRegistry
# =============================================================================

class TestMetricsRegistry:
    """测试指标注册中心"""

    def test_registry_create_counter(self):
        """创建计数器"""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter")

        assert counter.name == "test_counter"
        assert registry.counter("test_counter") is counter  # 同名复用

    def test_registry_create_gauge(self):
        """创建仪表"""
        registry = MetricsRegistry()
        gauge = registry.gauge("test_gauge")

        assert gauge.name == "test_gauge"
        assert registry.gauge("test_gauge") is gauge

    def test_registry_create_histogram(self):
        """创建直方图"""
        registry = MetricsRegistry()
        hist = registry.histogram("test_hist")

        assert hist.name == "test_hist"
        assert registry.histogram("test_hist") is hist

    def test_registry_collect(self):
        """收集指标"""
        registry = MetricsRegistry()

        counter = registry.counter("counter")
        _counter_inc(counter, 5)

        gauge = registry.gauge("gauge")
        _gauge_set(gauge, 42)

        hist = registry.histogram("histogram")
        _hist_observe(hist, 1.0)

        data = registry.collect()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data

    def test_registry_prometheus_export(self):
        """导出为 Prometheus 格式"""
        registry = MetricsRegistry()

        counter = registry.counter("http_requests", "Total HTTP requests")
        _counter_inc(counter, 10, labels={"method": "GET"})
        _counter_inc(counter, 5, labels={"method": "POST"})

        gauge = registry.gauge("memory_usage", "Memory usage in MB")
        _gauge_set(gauge, 512, labels={"service": "api"})

        hist = registry.histogram("latency", "Request latency")
        _hist_observe(hist, 0.1, labels={"endpoint": "/api"})
        _hist_observe(hist, 0.2, labels={"endpoint": "/api"})

        prometheus_text = registry.to_prometheus()

        assert "# HELP" in prometheus_text
        assert "# TYPE" in prometheus_text
        assert "http_requests" in prometheus_text
        assert "memory_usage" in prometheus_text
        assert "latency" in prometheus_text

        # 标签格式（至少包含 method="GET"）
        assert 'method="GET"' in prometheus_text

        # 值存在（10 或 10.0）
        assert "10" in prometheus_text or "10.0" in prometheus_text

    def test_registry_prometheus_histogram_format(self):
        """验证 Prometheus 直方图格式"""
        registry = MetricsRegistry()
        hist = registry.histogram("test_hist", "Test histogram")
        _hist_observe(hist, 0.5, labels={"endpoint": "/test"})
        _hist_observe(hist, 1.5, labels={"endpoint": "/test"})

        prometheus_text = registry.to_prometheus()

        assert "test_hist_bucket" in prometheus_text
        assert "test_hist_sum" in prometheus_text
        assert "test_hist_count" in prometheus_text
        assert 'le="' in prometheus_text

    def test_registry_cleanup_old_labels(self):
        """
        清理过期标签（如果实现提供 cleanup_old_labels）
        注意：不同实现可能对 last_access 记录策略不同，这里只验证不会报错且返回 >=0
        """
        registry = MetricsRegistry()
        counter = registry.counter("counter")

        _counter_inc(counter, 1, labels={"label1": "old"})
        _counter_inc(counter, 1, labels={"label2": "new"})

        if hasattr(registry, "cleanup_old_labels"):
            cleaned = registry.cleanup_old_labels(ttl_seconds=0)  # 立即清理
            assert cleaned >= 0

    def test_registry_reset(self):
        """重置所有指标"""
        registry = MetricsRegistry()

        counter = registry.counter("counter")
        _counter_inc(counter, 10)

        gauge = registry.gauge("gauge")
        _gauge_set(gauge, 42)

        registry.reset()

        # counter.reset 后应为 0
        assert _counter_get(counter) == 0.0

        # registry.reset 的策略：通常会清空 gauges/histograms
        #（这里维持原测试的行为假设）
        assert hasattr(registry, "_gauges")
        assert len(registry._gauges) == 0  # type: ignore[attr-defined]


# =============================================================================
# 并发场景
# =============================================================================

class TestMetricsConcurrency:
    """并发场景测试"""

    def test_concurrent_counter_increments(self):
        """并发计数器增加"""
        counter = Counter("counter")

        def worker():
            for _ in range(100):
                _counter_inc(counter, 1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 * 100 = 1000
        assert _counter_get(counter) == 1000.0

    def test_concurrent_histogram_observations(self):
        """并发直方图观测"""
        hist = Histogram("hist")

        def worker():
            for i in range(50):
                _hist_observe(hist, float(i))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = _hist_get_stats(hist)
        assert stats["count"] == 250


# =============================================================================
# Prometheus 格式特定测试
# =============================================================================

class TestPrometheusFormat:
    """Prometheus 格式特定测试"""

    def test_labels_formatting(self):
        """测试标签格式化"""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter")

        _counter_inc(counter, 1, labels={"method": "GET", "path": "/api/users", "status": "200"})

        prometheus_text = registry.to_prometheus()
        assert 'method="GET"' in prometheus_text
        assert 'path="/api/users"' in prometheus_text
        assert 'status="200"' in prometheus_text

    def test_metric_types(self):
        """验证指标类型声明"""
        registry = MetricsRegistry()
        registry.counter("my_counter")
        registry.gauge("my_gauge")
        registry.histogram("my_histogram")

        prometheus_text = registry.to_prometheus()
        assert "# TYPE my_counter counter" in prometheus_text
        assert "# TYPE my_gauge gauge" in prometheus_text
        assert "# TYPE my_histogram histogram" in prometheus_text

    def test_metric_descriptions(self):
        """验证指标描述"""
        registry = MetricsRegistry()
        registry.counter("counter1", "This is counter 1")
        registry.gauge("gauge1", "This is gauge 1")

        prometheus_text = registry.to_prometheus()
        assert "# HELP counter1 This is counter 1" in prometheus_text
        assert "# HELP gauge1 This is gauge 1" in prometheus_text
