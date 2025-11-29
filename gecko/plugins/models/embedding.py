# gecko/plugins/models/embedding.py
from __future__ import annotations

from typing import List

def _get_litellm():
    """
    惰性加载 litellm，避免导入 gecko.plugins.models.embedding 时强制要求安装。

    在实际调用 embedding 接口时再检查依赖是否存在。
    """
    try:
        import litellm  # type: ignore
        return litellm
    except ImportError as e:
        raise ImportError(
            "LiteLLMEmbedder requires 'litellm'. "
            "Install with: pip install litellm"
        ) from e

from gecko.core.exceptions import ModelError
from gecko.plugins.models.adapter import safe_access
from gecko.plugins.models.base import BaseEmbedder

class LiteLLMEmbedder(BaseEmbedder):
    """
    基于 LiteLLM 的通用 Embedding 模型实现
    """

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            litellm = _get_litellm()

            # 预处理：移除换行符
            clean_texts = [t.replace("\n", " ") for t in texts]
            
            params = {
                "model": self.config.model_name,
                "input": clean_texts,
                "timeout": self.config.timeout,
                **self.config.extra_kwargs
            }
            
            if self.config.api_key:
                params["api_key"] = self.config.api_key
            if self.config.base_url:
                params["api_base"] = self.config.base_url

            resp = await litellm.aembedding(**params)
            
            # 使用 safe_access 进行健壮提取
            embeddings: List[List[float]] = []
            data_items = safe_access(resp, "data", [])
            
            if isinstance(data_items, list):
                for item in data_items:
                    emb = safe_access(item, "embedding")
                    if emb:
                        embeddings.append(emb)
            
            return embeddings

        except Exception as e:
            raise ModelError(f"Embedding failed: {str(e)}") from e

    async def embed_query(self, text: str) -> List[float]:
        res = await self.embed_documents([text])
        if not res:
            raise ModelError("Embedding returned empty result")
        return res[0]