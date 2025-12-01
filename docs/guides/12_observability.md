# 可观测性 (Observability)

Gecko v0.3.1 深度集成了 **OpenTelemetry (OTel)**，让你可以像监控微服务一样监控 Agent。

## 快速配置

```python
from gecko.core.telemetry import TelemetryConfig, configure_telemetry
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 配置 OTLP 导出器 (例如连接到 Jaeger, SigNoz, Uptrace 等)
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")

config = TelemetryConfig(
    service_name="my-agent-app",
    environment="production",
    enabled=True,
    exporter=exporter
)

# 全局初始化 (建议在应用启动时调用)
configure_telemetry(config)
```

## 追踪视图

启用后，Gecko 会自动记录以下 Span：

*   **Workflow**: 整个工作流的执行耗时。
*   **Agent**: Agent `run` 的思考过程。
*   **Engine**: ReAct 循环的每一轮 (Thought / Action / Observation)。
*   **Tool**: 每个工具的调用参数和耗时。
*   **Model**: LLM API 的请求延迟和 Token 消耗。
*   **Storage**: 数据库读写操作。

## 日志关联

Gecko 的结构化日志系统 (`gecko.core.logging`) 会自动感知 Trace Context。

```python
from gecko.core.logging import get_logger

logger = get_logger(__name__)

async def my_tool_function():
    # 这条日志会自动附带当前的 trace_id 和 span_id
    logger.info("Processing data", user_id="u123")
```

在日志聚合平台中，你可以直接通过 `trace_id` 过滤出该请求的所有相关日志。