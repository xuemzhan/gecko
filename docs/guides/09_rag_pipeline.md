# RAG 知识库流水线

Gecko v0.3.1 提供了模块化的 RAG 支持，涵盖了从文档入库到检索增强生成的全流程。

## 架构组件

*   **Embedder**: 将文本转换为向量 (支持 LiteLLM 适配的所有模型)。
*   **VectorStore**: 存储向量和元数据 (支持 ChromaDB, LanceDB)。
*   **IngestionPipeline**: 处理文档加载、切分和入库。
*   **RetrievalTool**: 供 Agent 调用的检索工具。

## 1. 文档入库 (Ingestion)

使用 `IngestionPipeline` 可以快速处理本地文件。它会自动处理文本切分和向量化。

```python
from gecko.plugins.knowledge import IngestionPipeline, RecursiveCharacterTextSplitter
from gecko.plugins.models.embedding import LiteLLMEmbedder, ModelConfig
from gecko.plugins.storage.factory import create_storage

# 1. 初始化组件
embedder = LiteLLMEmbedder(
    config=ModelConfig(model_name="text-embedding-3-small", api_key="..."),
    dimension=1536
)
# 使用 LanceDB (高性能本地文件存储)
vector_store = await create_storage("lancedb://./my_vectors") 

# 2. 自定义切分器 (可选)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 3. 执行管道
pipeline = IngestionPipeline(vector_store, embedder, splitter)
await pipeline.run(["./company_policy.pdf", "./product_manual.md"])
```

## 2. 向量检索与过滤

Gecko 支持在检索时使用元数据过滤 (`metadata filtering`)，提高检索准确性。

```python
# 搜索 "报销流程"，但限制只在 "hr" 分类的文档中查找
results = await vector_store.search(
    query_embedding=vec, 
    top_k=5, 
    filters={"category": "hr"} 
)
```

## 3. 集成到 Agent

通过 `RetrievalTool` 将检索能力暴露给 Agent。

```python
from gecko.plugins.knowledge.tool import RetrievalTool

# 封装工具
rag_tool = RetrievalTool(
    name="search_knowledge_base",
    description="查询公司内部文档和政策",
    vector_store=vector_store,
    embedder=embedder,
    top_k=3
)

# 构建 Agent
agent = AgentBuilder().with_tools([rag_tool])...
```