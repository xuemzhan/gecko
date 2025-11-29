# gecko/plugins/knowledge/embedders.py
from __future__ import annotations
import os
from typing import List
from gecko.plugins.knowledge.interfaces import EmbedderProtocol


def _get_litellm():
    """
    惰性加载 litellm，避免导入 gecko.plugins.knowledge 时就强制依赖。

    在真正调用 embedding 时再检查依赖是否存在。
    """
    try:
        import litellm  # type: ignore
        return litellm
    except ImportError as e:
        raise ImportError(
            "OpenAIEmbedder / OllamaEmbedder requires 'litellm'. "
            "Install with: pip install litellm"
        ) from e

class OpenAIEmbedder(EmbedderProtocol):
    """
    基于 LiteLLM 的通用 Embedder
    支持 OpenAI, Azure, Ollama 等所有 LiteLLM 支持的 embedding 模型
    """
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536, **kwargs):
        self.model = model
        self._dimension = dimension
        self.kwargs = kwargs

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档嵌入（惰性加载 litellm）"""
        litellm = _get_litellm()

        # 替换换行符以提升某些模型的表现
        texts = [t.replace("\n", " ") for t in texts]
        response = await litellm.aembedding(
            model=self.model,
            input=texts,
            **self.kwargs
        )
        return [r["embedding"] for r in response.data]

    async def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入（惰性加载 litellm）"""
        litellm = _get_litellm()

        text = text.replace("\n", " ")
        response = await litellm.aembedding(
            model=self.model,
            input=[text],
            **self.kwargs
        )
        return response.data[0]["embedding"]

# 预设 Ollama 配置
class OllamaEmbedder(OpenAIEmbedder):
    """
    Ollama 本地嵌入模型适配器
    """
    def __init__(self, model: str = "ollama/nomic-embed-text", base_url: str = "http://localhost:11434", dimension: int = 768):
        super().__init__(
            model=model, 
            dimension=dimension, 
            api_base=base_url
        )