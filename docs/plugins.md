# 插件系统

Gecko 采用插件化架构，所有外部依赖（模型、存储、工具）均通过 Protocol 接入。

## Models (模型)

支持通过 `LiteLLM` 接入 100+ 种模型。

### 初始化
```python
from gecko.plugins.models import ZhipuChat, OpenAIChat, OllamaChat

# 智谱 AI
zhipu = ZhipuChat(api_key="...", model="glm-4-air")

# 本地 Ollama
ollama = OllamaChat(model="llama3", base_url="http://localhost:11434")
```

## Storage (存储)

支持多种后端，通过 URL Scheme 自动加载。

| Scheme | Backend | 特性 |
| :--- | :--- | :--- |
| `sqlite://` | SQLite | WAL 模式，支持跨进程文件锁，无需额外服务。 |
| `redis://` | Redis | 高性能，支持 TTL 自动过期。 |
| `chroma://` | ChromaDB | 向量存储，支持 Metadata 过滤。 |
| `lancedb://` | LanceDB | 基于文件的向量库，自动建表。 |

## Tools (工具)

工具必须继承 `BaseTool` 并定义 `args_schema`。

```python
from pydantic import BaseModel, Field
from gecko.plugins.tools.base import BaseTool, ToolResult

class SearchArgs(BaseModel):
    query: str = Field(..., description="查询关键词")

class SearchTool(BaseTool):
    name = "search"
    description = "搜索引擎"
    args_schema = SearchArgs

    async def _run(self, args: SearchArgs) -> ToolResult:
        # 执行逻辑...
        return ToolResult(content="结果...")
```

## Knowledge (RAG)

目前处于预览阶段，提供基础的入库流水线。

*   `IngestionPipeline`: 负责 `Load -> Split -> Embed -> Store` 流程。
*   `RetrievalTool`: 封装了向量检索逻辑的标准工具。