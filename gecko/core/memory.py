# gecko/core/memory.py
"""
Token Memory（Phase 2 优化版）

优化：
1. 智能缓存（避免重复计算）
2. 批量计数（提升性能）
3. 自动清理（防止内存泄漏）
"""
from __future__ import annotations
from typing import List, Optional, Dict
import hashlib
import tiktoken
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.core.logging import get_logger

logger = get_logger(__name__)

class TokenMemory:
    """
    基于 Token 计数的记忆管理器（优化版）
    
    新增功能：
    1. LRU 缓存（自动淘汰旧条目）
    2. 批量计数优化
    3. 缓存统计
    """
    
    def __init__(
        self,
        session_id: str,
        storage: Optional[SessionInterface] = None,
        max_tokens: int = 4000,
        model_name: str = "gpt-3.5-turbo",
        cache_size: int = 1000,  # ✅ 新增：缓存大小
    ):
        self.session_id = session_id
        self.storage = storage
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.cache_size = cache_size
        
        # Tokenizer（懒加载）
        self._encoding = None
        
        # ✅ Token 计数缓存
        self._token_cache: Dict[str, int] = {}
        self._cache_access_order: List[str] = []  # LRU 顺序
        
        # ✅ 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def tokenizer(self):
        """懒加载 tokenizer"""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(
                    "Unknown model for tiktoken",
                    model=self.model_name,
                    fallback="cl100k_base"
                )
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    # ========== Token 计数（带缓存） ==========

    def count_message_tokens(self, message: Message) -> int:
        """
        计算单条消息的 Token 数（带缓存）
        
        优化：
        1. 基于内容哈希的缓存
        2. LRU 淘汰策略
        3. 自动清理过大缓存
        """
        # 1. 生成缓存 Key
        cache_key = self._make_cache_key(message)
        
        # 2. 检查缓存
        if cache_key in self._token_cache:
            self._cache_hits += 1
            self._update_lru(cache_key)
            return self._token_cache[cache_key]
        
        # 3. 计算 Token
        self._cache_misses += 1
        token_count = self._count_tokens_impl(message)
        
        # 4. 存入缓存
        self._cache_token_count(cache_key, token_count)
        
        return token_count

    def _count_tokens_impl(self, message: Message) -> int:
        """实际的 Token 计数实现"""
        content = str(message.content)
        
        # 基础 Token 数
        content_tokens = len(self.tokenizer.encode(content))
        
        # 消息结构开销（role, name等）
        overhead = 4
        
        # Tool Calls 的额外开销
        if message.tool_calls:
            import json
            tool_str = json.dumps(message.tool_calls)
            overhead += len(self.tokenizer.encode(tool_str))
        
        return content_tokens + overhead

    def _make_cache_key(self, message: Message) -> str:
        """
        生成缓存 Key
        
        策略：基于消息内容的哈希
        """
        # 构建特征字符串
        features = f"{message.role}:{message.content}"
        if message.tool_calls:
            import json
            features += f":{json.dumps(message.tool_calls, sort_keys=True)}"
        
        # 生成哈希
        return hashlib.md5(features.encode()).hexdigest()[:16]

    def _cache_token_count(self, key: str, count: int):
        """
        缓存 Token 计数（LRU 策略）
        """
        # 如果缓存已满，淘汰最旧的
        if len(self._token_cache) >= self.cache_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._token_cache[oldest_key]
        
        # 存入缓存
        self._token_cache[key] = count
        self._cache_access_order.append(key)

    def _update_lru(self, key: str):
        """更新 LRU 顺序"""
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
            self._cache_access_order.append(key)

    # ========== 批量计数（优化） ==========

    def count_messages_batch(self, messages: List[Message]) -> List[int]:
        """
        批量计算 Token 数
        
        优化：减少重复的 tokenizer 调用
        """
        return [self.count_message_tokens(msg) for msg in messages]

    def count_total_tokens(self, messages: List[Message]) -> int:
        """计算总 Token 数"""
        return sum(self.count_messages_batch(messages))

    # ========== 历史加载与修剪 ==========

    async def get_history(self, raw_messages: List[dict]) -> List[Message]:
        """
        加载并修剪历史消息
        
        策略：
        1. 始终保留 System Prompt
        2. 倒序选取最近的对话
        3. 确保不超过 Token 上限
        """
        if not raw_messages:
            return []
        
        # 1. 反序列化
        messages = [Message(**m) for m in raw_messages]
        if not messages:
            return []
        
        # 2. 分离 System 消息
        system_msg = None
        if messages[0].role == "system":
            system_msg = messages.pop(0)
        
        # 3. 计算 System Token
        current_tokens = 0
        if system_msg:
            current_tokens = self.count_message_tokens(system_msg)
        
        # 4. 倒序选取历史
        selected_msgs: List[Message] = []
        
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens(msg)
            
            if current_tokens + msg_tokens > self.max_tokens:
                logger.debug(
                    "Message trimmed",
                    current_tokens=current_tokens,
                    msg_tokens=msg_tokens,
                    max_tokens=self.max_tokens
                )
                break
            
            selected_msgs.insert(0, msg)
            current_tokens += msg_tokens
        
        # 5. 重新添加 System
        if system_msg:
            selected_msgs.insert(0, system_msg)
        
        logger.debug(
            "History loaded",
            total_messages=len(messages) + (1 if system_msg else 0),
            selected_messages=len(selected_msgs),
            total_tokens=current_tokens
        )
        
        return selected_msgs

    # ========== 缓存管理 ==========

    def clear_cache(self):
        """清空缓存"""
        self._token_cache.clear()
        self._cache_access_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Token cache cleared")

    def get_cache_stats(self) -> Dict[str, any]:
        """获取缓存统计"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._token_cache),
            "max_size": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def print_cache_stats(self):
        """打印缓存统计（调试用）"""
        stats = self.get_cache_stats()
        logger.info(
            "Token cache stats",
            **stats
        )