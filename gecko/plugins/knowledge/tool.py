# gecko/plugins/knowledge/tool.py

from typing import Any, Type
from pydantic import BaseModel, Field, PrivateAttr
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.core.utils import ensure_awaitable


class RetrievalArgs(BaseModel):
    """
    RAG 检索工具的参数定义

    使用 Pydantic 模型作为 args_schema：
    - 便于 BaseTool.execute 统一做参数校验
    - 避免直接在 Tool 上手写 parameters 字典
    """
    query: str = Field(
        ...,
        description="用于在知识库中检索的查询语句"
    )


class RetrievalTool(BaseTool):
    """
    基于向量检索的知识库查询工具

    角色：
    - 作为 BaseTool 的一个具体实现，用于 RAG 检索场景
    - 通过 args_schema + _run 实现参数校验和业务逻辑
    """

    # 工具的基础元信息
    name: str = "knowledge_search"
    description: str = (
        "搜索内部知识库以获取相关信息。当问题涉及特定文档、报告或私有数据时使用。"
    )

    # ✅ 关键修复点：
    # 使用 args_schema 指定参数模型，避免在子类中定义与父类同名的字段 `parameters`
    # BaseTool.parameters 属性会自动基于此生成 OpenAI Tool Schema
    args_schema: Type[BaseModel] = RetrievalArgs  # type: ignore[assignment]

    # 使用 Pydantic PrivateAttr 来承载运行时依赖，避免被当作模型字段
    _vector_store: Any = PrivateAttr()
    _embedder: Any = PrivateAttr()
    _top_k: int = PrivateAttr(default=3)

    def __init__(
        self,
        vector_store: VectorInterface,
        embedder: EmbedderProtocol,
        top_k: int = 3,
        **data: Any,
    ) -> None:
        """
        初始化检索工具

        参数:
        - vector_store: 向量存储后端，须实现 VectorInterface
        - embedder: 向量化模型适配器，须实现 EmbedderProtocol
        - top_k: 检索返回的候选数量
        - data: 传递给 BaseTool/BaseModel 的其他字段（目前通常为空）
        """
        # 先初始化 BaseModel / BaseTool 部分（包括 args_schema 等）
        super().__init__(**data)

        # 再设置运行时依赖
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = top_k

    async def _run(self, args: BaseModel) -> ToolResult:  # type: ignore[override]
        """
        具体业务逻辑实现

        输入:
        - args: 已通过 args_schema 校验过的参数对象 (RetrievalArgs)

        输出:
        - ToolResult: 统一结果封装

        修复点：
        - 对向量检索调用使用 ensure_awaitable 包装，兼容：
          * 同步函数 (return list)
          * 协程函数 (async def)
          * AsyncMock / MagicMock
        避免测试中使用 AsyncMock 时出现 "coroutine was never awaited"。
        """
        # 为了兼容潜在的 schema 变动，这里既支持属性访问也支持 model_dump
        query = getattr(args, "query", None) or args.model_dump().get("query")
        if not query:
            return ToolResult(content="错误：查询语句为空", is_error=True)

        # 1. 向量化查询
        #    embed_query 可能是同步函数、协程或 AsyncMock，所以统一通过 ensure_awaitable 调用
        query_vec = await ensure_awaitable(self._embedder.embed_query, query)

        # 2. 进行向量检索
        #    ✅ 修复点：search 同样使用 ensure_awaitable，而不是直接 `await self._vector_store.search(...)`
        results = await ensure_awaitable(
            self._vector_store.search,
            query_vec,
            top_k=self._top_k,
        )

        if not results:
            return ToolResult(content="未在知识库中找到相关内容。")

        # 3. 格式化结果
        context = "找到以下相关内容：\n\n"
        for i, res in enumerate(results, 1):
            metadata = res.get("metadata", {}) or {}
            source = metadata.get("filename", "unknown")
            snippet = res.get("text") or metadata.get("snippet", "")
            context += f"[{i}] 来源：{source}\n{snippet}\n\n"

        return ToolResult(content=context)
