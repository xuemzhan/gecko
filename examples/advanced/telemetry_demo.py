import asyncio
from gecko.core.telemetry import configure_telemetry, TelemetryConfig, get_telemetry
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

async def main():
    # 1. 配置遥测 (打印到控制台或导出到 Jaeger/OTLP)
    config = TelemetryConfig(
        service_name="gecko-demo",
        enabled=True,
        exporter=ConsoleSpanExporter()  # 取消注释，并实例化
    )
    telemetry = configure_telemetry(config)
    
    # 2. 使用装饰器追踪
    @telemetry.trace_async("complex_operation")
    async def run_task(data):
        # 3. 手动添加事件
        telemetry.add_event("processing_start", {"data_len": len(data)})
        await asyncio.sleep(0.1)
        
        # 4. 嵌套 Span
        async with telemetry.async_span("inner_step", {"step": 1}) as span:
            span.set_attribute("result", "success") # type: ignore
    
    await run_task("hello gecko")
    print("Trace generated. (Check console if Exporter configured)")

if __name__ == "__main__":
    asyncio.run(main())