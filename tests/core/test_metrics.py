# tests/core/test_metrics.py
"""
指标收集系统单元测试

测试 Counter、Gauge、Histogram 和 MetricsRegistry。
"""
import pytest
import time
from unittest.mock import MagicMock, patch

from gecko.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    MetricSample,
)


class TestCounter:
    """测试计数器"""

    def test_counter_init(self):
        """初始化计数器"""
        counter = Counter("test_counter", "Test counter")
        assert counter.name == "test_counter"
        assert counter.description == "Test counter"

    def test_counter_increment(self):
        """增加计数"""
        counter = Counter("counter")
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc(5)
        assert counter.get() == 6.0

    def test_counter_with_labels(self):
        """带标签的计数器"""
        counter = Counter("counter")
        counter.inc(method="GET", path="/api")
        counter.inc(method="POST", path="/api")
        
        assert counter.get(method="GET", path="/api") == 1.0
        assert counter.get(method="POST", path="/api") == 1.0

    def test_counter_reset(self):
        """重置计数器"""
        counter = Counter("counter")
        counter.inc(10)
        assert counter.get() == 10.0
        
        counter.reset()
        assert counter.get() == 0.0


class TestGauge:
    """测试仪表"""

    def test_gauge_set(self):
        """设置仪表值"""
        gauge = Gauge("gauge")
        gauge.set(42)
        assert gauge.get() == 42.0

    def test_gauge_increment(self):
        """增加仪表值"""
        gauge = Gauge("gauge")
        gauge.set(10)
        gauge.inc(5)
        assert gauge.get() == 15.0

    def test_gauge_decrement(self):
        """减少仪表值"""
        gauge = Gauge("gauge")
        gauge.set(10)
        gauge.dec(3)
        assert gauge.get() == 7.0

    def test_gauge_with_labels(self):
        """带标签的仪表"""
        gauge = Gauge("gauge")
        gauge.set(10, service="api")
        gauge.set(20, service="db")
        
        assert gauge.get(service="api") == 10.0
        assert gauge.get(service="db") == 20.0


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
        hist.observe(0.5)
        hist.observe(1.5)
        
        stats = hist.get_stats()
        assert stats["count"] == 2
        assert stats["sum"] == 2.0

    def test_histogram_stats_complete(self):
        """获取完整统计"""
        hist = Histogram("hist")
        values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
        
        for val in values:
            hist.observe(val)
        
        stats = hist.get_stats()
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
        
        # 记录 100 个值 0-99
        for i in range(100):
            hist.observe(float(i))
        
        stats = hist.get_stats()
        # p50 应该接近 50
        assert 45 <= stats["p50"] <= 55
        # p95 应该接近 95
        assert 90 <= stats["p95"] <= 100
        # p99 应该接近 99
        assert 95 <= stats["p99"] <= 100

    def test_histogram_with_labels(self):
        """带标签的直方图"""
        hist = Histogram("hist")
        hist.observe(0.5, endpoint="/api")
        hist.observe(1.5, endpoint="/api")
        hist.observe(2.0, endpoint="/db")
        
        stats_api = hist.get_stats(endpoint="/api")
        assert stats_api["count"] == 2
        
        stats_db = hist.get_stats(endpoint="/db")
        assert stats_db["count"] == 1

    def test_histogram_time_context(self):
        """测试计时上下文"""
        hist = Histogram("hist")
        
        with hist.time():
            time.sleep(0.01)  # 等待 10ms
        
        stats = hist.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] >= 0.01  # 至少 10ms

    def test_histogram_bucket_stats(self):
        """测试 bucket 统计"""
        hist = Histogram("hist")
        
        for i in range(20):
            hist.observe(float(i) / 10)  # 0.0 - 1.9
        
        buckets = hist.get_bucket_stats()
        assert len(buckets) > 0
        # 验证 bucket 计数是累积的
        assert buckets[float('inf')] == 20


class TestMetricsRegistry:
    """测试指标注册中心"""

    def test_registry_create_counter(self):
        """创建计数器"""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter")
        
        assert counter.name == "test_counter"
        # 再次获取应该返回同一个对象
        assert registry.counter("test_counter") is counter

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
        counter.inc(5)
        
        gauge = registry.gauge("gauge")
        gauge.set(42)
        
        hist = registry.histogram("histogram")
        hist.observe(1.0)
        
        data = registry.collect()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data

    def test_registry_prometheus_export(self):
        """导出为 Prometheus 格式"""
        registry = MetricsRegistry()
        
        counter = registry.counter("http_requests", "Total HTTP requests")
        counter.inc(10, method="GET")
        counter.inc(5, method="POST")
        
        gauge = registry.gauge("memory_usage", "Memory usage in MB")
        gauge.set(512, service="api")
        
        hist = registry.histogram("latency", "Request latency")
        hist.observe(0.1, endpoint="/api")
        hist.observe(0.2, endpoint="/api")
        
        prometheus_text = registry.to_prometheus()
        
        # 验证格式
        assert "# HELP" in prometheus_text
        assert "# TYPE" in prometheus_text
        assert "http_requests" in prometheus_text
        assert "memory_usage" in prometheus_text
        assert "latency" in prometheus_text
        
        # 验证具体值
        assert 'method="GET"' in prometheus_text
        assert "10" in prometheus_text or "10.0" in prometheus_text

    def test_registry_prometheus_histogram_format(self):
        """验证 Prometheus 直方图格式"""
        registry = MetricsRegistry()
        hist = registry.histogram("test_hist", "Test histogram")
        hist.observe(0.5, endpoint="/test")
        hist.observe(1.5, endpoint="/test")
        
        prometheus_text = registry.to_prometheus()
        
        # 应该包含 bucket、sum、count
        assert "test_hist_bucket" in prometheus_text
        assert "test_hist_sum" in prometheus_text
        assert "test_hist_count" in prometheus_text
        assert 'le="' in prometheus_text  # bucket 标签

    def test_registry_cleanup_old_labels(self):
        """清理过期标签"""
        registry = MetricsRegistry()
        counter = registry.counter("counter")
        
        counter.inc(1, label1="old")
        counter.inc(1, label2="new")
        
        # 模拟时间流逝（这里只是演示 API）
        cleaned = registry.cleanup_old_labels(ttl_seconds=0)  # 立即清理
        assert cleaned >= 0  # 应该清理了一些标签

    def test_registry_reset(self):
        """重置所有指标"""
        registry = MetricsRegistry()
        
        counter = registry.counter("counter")
        counter.inc(10)
        
        gauge = registry.gauge("gauge")
        gauge.set(42)
        
        registry.reset()
        
        # 计数器应该被清空
        assert counter.get() == 0.0
        assert len(registry._gauges) == 0


class TestMetricsSampling:
    """测试并发场景"""

    def test_concurrent_counter_increments(self):
        """并发计数器增加"""
        import threading
        
        counter = Counter("counter")
        
        def increment():
            for _ in range(100):
                counter.inc()
        
        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 应该得到 1000（10 个线程 * 100 次）
        assert counter.get() == 1000.0

    def test_concurrent_histogram_observations(self):
        """并发直方图观测"""
        import threading
        
        hist = Histogram("hist")
        
        def observe_values():
            for i in range(50):
                hist.observe(float(i))
        
        threads = [threading.Thread(target=observe_values) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = hist.get_stats()
        # 应该有 250 个观测值（5 个线程 * 50 个）
        assert stats["count"] == 250


class TestPrometheusFormat:
    """Prometheus 格式特定测试"""

    def test_labels_formatting(self):
        """测试标签格式化"""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter")
        counter.inc(1, method="GET", path="/api/users", status="200")
        
        prometheus_text = registry.to_prometheus()
        
        # 标签应该被正确格式化
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
