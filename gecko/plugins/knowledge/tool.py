# gecko/plugins/knowledge/tool.py

from typing import Any, Type
from pydantic import BaseModel, Field, PrivateAttr  # 修复: 添加 PrivateAttr 导入
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.core.utils import ensure_awaitable


class RetrievalTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "搜索内部知识库以获取相关信息。当问题涉及特定文档、报告或私有数据时使用。"
    parameters: dict = { # type: ignore
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "用于在知识库中检索的查询语句"
            }
        },
        "required": ["query"]
    }
    
    # 修复: 使用 Pydantic PrivateAttr
    _vector_store: Any = PrivateAttr()
    _embedder: Any = PrivateAttr()
    _top_k: int = PrivateAttr(default=3)

    def __init__(
        self, 
        vector_store: VectorInterface, 
        embedder: EmbedderProtocol, 
        top_k: int = 3,
        **data
    ):
        # 修复: 先调用 super().__init__，再设置私有属性
        super().__init__(**data)
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = top_k

    async def _run(self, args: BaseModel) -> ToolResult:  # type: ignore
        query = getattr(args, 'query', None) or args.model_dump().get("query")
        if not query:
            return ToolResult(content="错误：查询语句为空", is_error=True)

        # 1. Embed Query
        query_vec = await ensure_awaitable(self._embedder.embed_query, query)
        
        # 2. Vector Search
        results = await self._vector_store.search(query_vec, top_k=self._top_k)
        
        if not results:
            return ToolResult(content="未在知识库中找到相关内容。")
            
        # 3. Format Results
        context = "找到以下相关内容：\n\n"
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('filename', 'unknown')
            score = f"{res['score']:.2f}" if 'score' in res else 'N/A'
            context += f"--- 文档 {i} (来源: {source}, 相关度: {score}) ---\n{res['text']}\n\n"
            
        return ToolResult(content=context)