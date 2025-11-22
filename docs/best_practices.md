# 最佳实践

## 1. 并发安全

在自定义 Storage 或 Tool 时，请注意 Gecko 是异步并发环境。
*   **不要阻塞 Event Loop**: 如果有耗时的同步 I/O（如文件读写），请继承 `ThreadOffloadMixin` 并使用 `self._run_sync()`。
*   **使用锁**: 如果需要修改共享状态，使用 `asyncio.Lock`。如果涉及跨进程文件写入，参考 `AtomicWriteMixin`。

## 2. 错误处理

Gecko 提倡显式的错误处理。
*   **Workflow**: 节点抛出的异常会中断流程。如果希望容错，请在节点内部 try-catch 并返回特定的 `Next` 指令，或者开启 `Workflow(enable_retry=True)`。
*   **Team**: `Team` 引擎默认捕获单个成员的异常，返回 `MemberResult(is_success=False)`，不会中断整个团队的执行。

## 3. 性能优化

*   **SQLite**: 默认启用了 WAL 模式，并发读写性能很好。但在极高并发下建议切换到 Redis。
*   **Token 计数**: `TokenMemory` 内部有 LRU 缓存。对于批量消息处理，确保开启缓存以减少 `tiktoken` 计算开销。

## 4. 调试与日志

Gecko 使用结构化日志。
*   推荐安装 `structlog` 以获得更好的 JSON 日志格式。
*   设置环境变量 `GECKO_LOG_LEVEL=DEBUG` 可以查看详细的 Prompt 和 Token 消耗。