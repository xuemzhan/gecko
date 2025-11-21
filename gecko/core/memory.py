# gecko/core/memory.py  
"""  
Token Memory（优化版）  
1. 懒加载 tiktoken，减少不必要依赖  
2. 使用 OrderedDict 实现 LRU 缓存  
3. 支持多模态消息的 token 估算  
4. 批量计数时复用同一个 encoder，绕过缓存开销，提升性能  
"""  
  
from __future__ import annotations  
  
import hashlib  
from collections import OrderedDict  
from typing import Any, Dict, List, Optional  
  
from gecko.core.logging import get_logger  
from gecko.core.message import ContentBlock, Message  
from gecko.plugins.storage.interfaces import SessionInterface  
  
logger = get_logger(__name__)  
  
  
class TokenMemory:  
    def __init__(  
        self,  
        session_id: str,  
        storage: Optional[SessionInterface] = None,  
        max_tokens: int = 4000,  
        model_name: str = "gpt-3.5-turbo",  
        cache_size: int = 1000,  
    ):  
        self.session_id = session_id  
        self.storage = storage  
        self.max_tokens = max_tokens  
        self.model_name = model_name  
        self.cache_size = cache_size  
  
        self._encoding = None  
        self._token_cache: OrderedDict[str, int] = OrderedDict()  
        self._cache_hits = 0  
        self._cache_misses = 0  
  
    # ---------- Tokenizer ----------  
    @property  
    def tokenizer(self):  
        if self._encoding is None:  
            try:  
                import tiktoken  
                self._encoding = tiktoken.encoding_for_model(self.model_name)  
            except Exception:  
                import tiktoken  
                logger.warning("Unknown model for tiktoken, fallback to cl100k_base")  
                self._encoding = tiktoken.get_encoding("cl100k_base")  
        return self._encoding  
  
    # ---------- 单条计数（带缓存） ----------  
    def count_message_tokens(self, message: Message) -> int:  
        cache_key = self._make_cache_key(message)  
        if cache_key in self._token_cache:  
            self._cache_hits += 1  
            self._token_cache.move_to_end(cache_key)  
            return self._token_cache[cache_key]  
  
        self._cache_misses += 1  
        token_count = self._count_tokens_impl(message)  
        self._cache_token_count(cache_key, token_count)  
        return token_count  
  
    def _make_cache_key(self, message: Message) -> str:  
        if isinstance(message.content, list):  
            key = "".join(block.text or "[image]" for block in message.content)  
        else:  
            key = str(message.content)  
  
        raw = f"{message.role}:{key}"  
        if message.tool_calls:  
            import json  
            raw += json.dumps(message.tool_calls, sort_keys=True)  
        return hashlib.md5(raw.encode()).hexdigest()  
  
    def _cache_token_count(self, key: str, count: int):  
        self._token_cache[key] = count  
        self._token_cache.move_to_end(key)  
        if len(self._token_cache) > self.cache_size:  
            self._token_cache.popitem(last=False)  
  
    # ---------- 批量计数 ----------  
    def count_messages_batch(self, messages: List[Message]) -> List[int]:  
        """  
        批量计算 Token 数：  
        - 共享同一个 encoder  
        - 跳过缓存和哈希操作（比逐条更快）  
        """  
        encode = self.tokenizer.encode  
        return [self._count_tokens_impl(msg, encode=encode) for msg in messages]  
  
    def _count_tokens_impl(self, message: Message, encode=None) -> int:  
        encode = encode or self.tokenizer.encode  
        content_tokens = 0  
  
        if isinstance(message.content, str):  
            content_tokens = len(encode(message.content))  
        elif isinstance(message.content, list):  
            buffer = []  
            for block in message.content:  
                if block.type == "text" and block.text:  
                    buffer.append(block.text)  
                elif block.type == "image_url":  
                    buffer.append("[image]")  
            content_tokens = len(encode(" ".join(buffer)))  
  
        overhead = 4  
        if message.tool_calls:  
            import json  
            overhead += len(encode(json.dumps(message.tool_calls)))  
  
        return content_tokens + overhead  
  
    # ---------- 历史加载 ----------  
    async def get_history(self, raw_messages: List[dict]) -> List[Message]:  
        messages: List[Message] = []  
        for entry in raw_messages:  
            try:  
                msg = Message(**entry)  
            except Exception as e:  
                logger.warning("Invalid history message, skip", error=str(e))  
                continue  
  
            # 限制单条消息长度，防止极端情况  
            if isinstance(msg.content, str) and len(msg.content) > 2000:  
                msg.content = msg.content[:2000]  
            messages.append(msg)  
  
        if not messages:  
            return []  
  
        system_msg = None  
        if messages[0].role == "system":  
            system_msg = messages.pop(0)  
  
        selected: List[Message] = []  
        current_tokens = self.count_message_tokens(system_msg) if system_msg else 0  
  
        for msg in reversed(messages):  
            token = self.count_message_tokens(msg)  
            if current_tokens + token > self.max_tokens:  
                break  
            selected.insert(0, msg)  
            current_tokens += token  
  
        if system_msg:  
            selected.insert(0, system_msg)  
  
        logger.debug("History loaded", total_messages=len(messages), selected=len(selected))  
        return selected  
  
    # ---------- 缓存管理 ----------  
    def clear_cache(self):  
        self._token_cache.clear()  
        self._cache_hits = 0  
        self._cache_misses = 0  
  
    def get_cache_stats(self) -> Dict[str, Any]:  
        total = self._cache_hits + self._cache_misses  
        return {  
            "cache_size": len(self._token_cache),  
            "hits": self._cache_hits,  
            "misses": self._cache_misses,  
            "hit_rate": self._cache_hits / total if total else 0,  
        }  
