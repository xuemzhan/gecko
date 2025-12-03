# gecko/core/memory/hybrid.py
"""
混合记忆系统 (Hybrid Memory)

结合短期 TokenMemory 和 长期 VectorStore 的记忆管理。
"""
from __future__ import annotations

import time
import asyncio
from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.memory.base import TokenMemory
from gecko.core.message import Message
from gecko.core.utils import ensure_awaitable
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.plugins.storage.interfaces import VectorInterface

logger = get_logger(__name__)


class HybridMemory(TokenMemory):
    """
    长短期混合记忆
    
    流程:
    1. Short-term: 保留最近对话 (基于 Token 限制)。
    2. Long-term:  基于 Query 进行向量检索，找回历史相关片段。
    """

    def __init__(
        self,
        session_id: str,
        vector_store: VectorInterface,
        embedder: EmbedderProtocol,
        short_term_limit: int = 2000,
        top_k_recall: int = 3,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """
        初始化
        
        Args:
            session_id: 会话ID
            vector_store: 向量数据库接口
            embedder: Embedding 模型接口
            short_term_limit: 短期记忆的 Token 上限
            top_k_recall: 长期记忆召回条数
            similarity_threshold: 相似度阈值 (0-1)
        """
        # 基类负责管理短期记忆 (Short-term Buffer)
        super().__init__(session_id, max_tokens=short_term_limit, **kwargs)
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k_recall = top_k_recall
        self.similarity_threshold = similarity_threshold

    async def get_history( # type: ignore
        self, 
        raw_messages: List[Dict[str, Any]], 
        query: Optional[str] = None, 
        preserve_system: bool = True
    ) -> List[Message]:
        """
        获取上下文 (Short + Long Term)
        
        [优化] 并行执行短期记忆处理和长期记忆检索
        """
        
        # 定义长期记忆检索任务
        async def _fetch_long_term() -> List[Message]:
            # 如果没有 query 或者没有配置向量库，直接跳过
            if not query or not self.vector_store:
                return []
                
            try:
                # 向量化 Query
                query_vec = await ensure_awaitable(self.embedder.embed_query, query)
                
                # 搜索 (增加 session_id 过滤，确保只搜到当前用户的记忆)
                docs = await self.vector_store.search(
                    query_vec, 
                    top_k=self.top_k_recall, 
                    filters={"session_id": self.session_id}
                )
                
                relevant_snippets = []
                for doc in docs:
                    score = doc.get("score", 0.0)
                    if score >= self.similarity_threshold:
                        text = doc.get("text", "")
                        timestamp = doc.get("metadata", {}).get("timestamp", "unknown")
                        relevant_snippets.append(f"[{timestamp}] {text}")
                
                if relevant_snippets:
                    context_block = "\n---\n".join(relevant_snippets)
                    # 将召回的内容封装为 System Message
                    return [Message.system(
                        f"Relevant Context from Memory (Historical Data):\n{context_block}"
                    )]
            except Exception as e:
                logger.error(f"HybridMemory recall failed: {e}")
                return []
            return []

        # [核心优化] 并行执行
        short_term_msgs, long_term_msgs = await asyncio.gather(
            super().get_history(raw_messages, preserve_system),
            _fetch_long_term()
        )

        # 3. 组装上下文
        # 顺序建议: [System Prompt] -> [Long Term Context] -> [Short Term History]
        final_msgs = []
        
        # 提取原有的 System Prompt (如果有)
        if short_term_msgs and short_term_msgs[0].role == "system":
            final_msgs.append(short_term_msgs.pop(0))
        
        # 插入长期记忆
        final_msgs.extend(long_term_msgs)
        
        # 追加短期记忆
        final_msgs.extend(short_term_msgs)
        
        return final_msgs

    async def archive_message(self, message: Message):
        """
        手动归档一条消息到长期记忆 (Write Path)
        
        通常在 Agent 每一轮对话结束后调用。
        """
        text = message.get_text_content()
        # 忽略过短内容或非文本内容
        if not text or len(text) < 10: 
            return

        try:
            # 生成向量
            vec = await ensure_awaitable(self.embedder.embed_query, text)
            
            # 构造文档
            doc_id = f"{self.session_id}_{int(time.time()*1000)}"
            document = {
                "id": doc_id,
                "text": text,
                "embedding": vec,
                "metadata": {
                    "session_id": self.session_id,
                    "role": message.role,
                    "timestamp": time.time()
                }
            }
            
            # 写入向量库
            await self.vector_store.upsert([document])
            logger.debug("Archived message to vector store", doc_id=doc_id)
            
        except Exception as e:
            logger.error(f"Failed to archive message: {e}")