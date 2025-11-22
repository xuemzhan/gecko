# 配置详解

Gecko 使用 `pydantic-settings` 管理配置。你可以通过 `.env` 文件或系统环境变量来覆盖默认值。

## 基础配置

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_LOG_LEVEL` | `INFO` | 日志级别 (DEBUG, INFO, WARNING, ERROR) |
| `GECKO_LOG_FORMAT` | `text` | 日志格式，生产环境建议设为 `json` 以配合 ELK/Splunk |
| `GECKO_DEFAULT_MODEL` | `gpt-3.5-turbo` | `AgentBuilder` 未指定模型时的默认值 |
| `GECKO_DEFAULT_TIMEOUT` | `30.0` | 工具执行的默认超时时间（秒） |

## 安全与限制

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_MAX_TURNS` | `5` | ReAct 循环的最大思考轮数，防止死循环消耗 Token |
| `GECKO_MAX_CONTEXT_TOKENS` | `4000` | Memory 的最大 Token 窗口限制 |

## 存储连接

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `GECKO_DEFAULT_STORAGE_URL` | `sqlite://./gecko_data.db` | 默认的持久化存储地址 |

## Model Provider 凭证

Gecko 底层依赖 `LiteLLM`，支持其所有标准环境变量：

*   `OPENAI_API_KEY`
*   `ZHIPU_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `AZURE_API_KEY` / `AZURE_API_BASE`
*   ... (更多请参考 LiteLLM 文档)