# gecko/plugins/knowledge/tool.py
from typing import Type
from pydantic import BaseModel, Field
from gecko.plugins.tools.base import BaseTool
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.core.utils import ensure_awaitable

class RetrievalTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "搜索内部知识库以获取相关信息。当问题涉及特定文档、报告或私有数据时使用。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "用于在知识库中检索的查询语句"
            }
        },
        "required": ["query"]
    }

    def __init__(self, vector_store: VectorInterface, embedder: EmbedderProtocol, top_k: int = 3):
        super().__init__()
        # Private attributes are not Pydantic fields
        object.__setattr__(self, "_vector_store", vector_store)
        object.__setattr__(self, "_embedder", embedder)
        object.__setattr__(self, "_top_k", top_k)

    async def execute(self, arguments: dict) -> str:
        query = arguments.get("query")
        if not query:
            return "错误：查询语句为空"

        # 1. Embed Query
        query_vec = await ensure_awaitable(self._embedder.embed_query, query)
        
        # 2. Vector Search
        results = await self._vector_store.search(query_vec, top_k=self._top_k)
        
        if not results:
            return "未在知识库中找到相关内容。"
            
        # 3. Format Results
        context = "找到以下相关内容：\n\n"
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('filename', 'unknown')
            score = f"{res['score']:.2f}" if 'score' in res else 'N/A'
            context += f"--- 文档 {i} (来源: {source}, 相关度: {score}) ---\n{res['text']}\n\n"
            
        return context