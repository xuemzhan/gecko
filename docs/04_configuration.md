# 配置详解

Gecko 使用 `pydantic-settings` 管理全局配置。你可以通过环境变量（`GECKO_` 前缀）或 `.env` 文件进行覆盖。

## 核心配置

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_LOG_LEVEL` | `INFO` | 日志级别 (DEBUG, INFO, WARNING, ERROR) |
| `GECKO_LOG_FORMAT` | `text` | 日志格式，生产环境推荐 `json` |
| `GECKO_DEFAULT_MODEL` | `gpt-4o` | 默认 LLM 模型名称 |
| `GECKO_DEFAULT_STORAGE_URL`| `sqlite:///./gecko_data.db` | 默认 KV 存储地址 |

## 智能体与运行时

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_MAX_TURNS` | `10` | ReAct 循环最大轮数，防止死循环消耗 Token |
| `GECKO_MAX_CONTEXT_TOKENS` | `4000` | Memory 的最大 Token 窗口 |
| `GECKO_TOOL_EXECUTION_TIMEOUT`| `30.0` | 单个工具执行的超时时间（秒） |

## 记忆与摘要 (Memory)

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_MEMORY_SUMMARY_INTERVAL` | `30.0` | 摘要更新的最小间隔（秒），用于防抖 |
| `GECKO_MEMORY_CACHE_SIZE` | `2000` | Token 计数器的 LRU 缓存大小 |
| `GECKO_MEMORY_SUMMARY_RESERVE_TOKENS` | `500` | 为摘要生成预留的 Token 数 |

## 工作流 (Workflow)

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_WORKFLOW_CHECKPOINT_STRATEGY`| `final` | 持久化策略：`always` (每步), `final` (仅结束), `manual` |
| `GECKO_WORKFLOW_HISTORY_RETENTION` | `20` | 持久化时保留的历史记录步数，防止状态爆炸 |

## 可观测性 (Telemetry)

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_TELEMETRY_ENABLED` | `True` | 是否启用 OpenTelemetry |
| `GECKO_TELEMETRY_SERVICE_NAME` | `gecko-app` | 在 APM 系统中显示的服务名称 |