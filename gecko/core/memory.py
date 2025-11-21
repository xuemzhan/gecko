# gecko/core/memory.py
"""
Token Memory - 对话历史与上下文管理

核心功能：
1. Token 计数（基于 tiktoken）
2. LRU 缓存优化
3. 历史消息加载与裁剪
4. 多模态消息支持

优化点：
1. 改进缓存策略（批量计数也使用缓存）
2. 可配置的消息长度限制
3. 更安全的历史加载
4. 完善的缓存统计
5. 更好的错误处理
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.message import ContentBlock, Message
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)


class TokenMemory:
    """
    Token-aware 记忆管理器
    
    负责：
    - 计算消息的 token 数量
    - 管理对话历史（限制在 max_tokens 内）
    - 缓存 token 计数结果以提升性能
    
    示例:
        ```python
        memory = TokenMemory(
            session_id="user_123",
            storage=sqlite_storage,
            max_tokens=4000,
            model_name="gpt-4"
        )
        
        # 计算单条消息
        count = memory.count_message_tokens(Message.user("Hello"))
        print(f"Token count: {count}")
        
        # 批量计算
        messages = [Message.user("Hi"), Message.assistant("Hello")]
        counts = memory.count_messages_batch(messages)
        
        # 加载历史（自动裁剪）
        history = await memory.get_history(raw_messages)
        
        # 查看缓存统计
        stats = memory.get_cache_stats()
        print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        ```
    """

    def __init__(
        self,
        session_id: str,
        storage: Optional[SessionInterface] = None,
        max_tokens: int = 4000,
        model_name: str = "gpt-3.5-turbo",
        cache_size: int = 1000,
        max_message_length: int = 10000,
        enable_cache_for_batch: bool = True,
    ):
        """
        初始化 TokenMemory
        
        参数:
            session_id: 会话唯一标识
            storage: 可选的持久化存储
            max_tokens: 上下文窗口最大 token 数
            model_name: 模型名称（用于选择 tiktoken encoder）
            cache_size: LRU 缓存大小
            max_message_length: 单条消息最大字符长度（防止极端情况）
            enable_cache_for_batch: 批量计数时是否使用缓存
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数，收到: {max_tokens}")
        
        if cache_size <= 0:
            raise ValueError(f"cache_size 必须为正数，收到: {cache_size}")
        
        self.session_id = session_id
        self.storage = storage
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.cache_size = cache_size
        self.max_message_length = max_message_length
        self.enable_cache_for_batch = enable_cache_for_batch
        
        # 延迟初始化的 tokenizer
        self._encoding = None
        
        # LRU 缓存（OrderedDict 实现）
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        
        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    # ====================== Tokenizer（延迟加载）======================

    @property
    def tokenizer(self):
        """
        延迟加载 tiktoken encoder
        
        优势：
        1. 仅在首次使用时加载
        2. 避免不必要的依赖
        3. 支持模型名称降级
        """
        if self._encoding is None:
            try:
                import tiktoken
                
                try:
                    # 尝试按模型名称加载
                    self._encoding = tiktoken.encoding_for_model(self.model_name)
                    logger.debug("Tokenizer loaded", model=self.model_name)
                except KeyError:
                    # 模型未知，降级到 cl100k_base（GPT-4/3.5 的编码）
                    logger.warning(
                        "Unknown model for tiktoken, fallback to cl100k_base",
                        model=self.model_name
                    )
                    self._encoding = tiktoken.get_encoding("cl100k_base")
                    
            except ImportError as e:
                raise ImportError(
                    "TokenMemory 需要 tiktoken 库。请安装：pip install tiktoken"
                ) from e
            except Exception as e:
                logger.error("Failed to load tokenizer", error=str(e))
                raise RuntimeError(f"Tokenizer 加载失败: {e}") from e
        
        return self._encoding

    # ====================== 单条消息计数（带缓存）======================

    def count_message_tokens(self, message: Message) -> int:
        """
        计算单条消息的 token 数（带缓存）
        
        参数:
            message: Message 对象
        
        返回:
            token 数量
        
        缓存策略:
            - 使用 MD5 哈希作为缓存键
            - LRU 淘汰最久未使用的条目
        """
        cache_key = self._make_cache_key(message)
        
        # 检查缓存
        if cache_key in self._token_cache:
            self._cache_hits += 1
            # 移动到末尾（标记为最近使用）
            self._token_cache.move_to_end(cache_key)
            return self._token_cache[cache_key]
        
        # 缓存未命中，计算 token
        self._cache_misses += 1
        token_count = self._count_tokens_impl(message)
        
        # 存入缓存
        self._cache_token_count(cache_key, token_count)
        
        return token_count

    def _make_cache_key(self, message: Message) -> str:
        """
        生成消息的缓存键（使用 MD5 哈希）
        
        包含：
        - role
        - content（文本或多模态）
        - tool_calls（如果有）
        
        注意：使用 JSON 序列化确保一致性
        """
        # 构建键内容
        key_data = {
            "role": message.role,
            "content": self._serialize_content(message.content),
        }
        
        # 包含 tool_calls（如果存在）
        if message.tool_calls:
            # 排序确保一致性
            key_data["tool_calls"] = json.dumps(
                message.tool_calls,
                sort_keys=True,
                ensure_ascii=False
            )
        
        # 序列化并哈希
        raw = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def _serialize_content(self, content: str | List[ContentBlock]) -> str | List[str]:
        """
        序列化消息内容（用于缓存键生成）
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # 多模态消息：提取文本和类型
            parts = []
            for block in content:
                if block.type == "text" and block.text:
                    parts.append(f"text:{block.text}")
                elif block.type == "image_url":
                    # 图片只记录类型和 URL（不包含 base64 数据）
                    url = ""
                    if block.image_url:
                        url = block.image_url.url or "base64_image"
                    parts.append(f"image:{url}")
            return parts
        else:
            return str(content)

    def _cache_token_count(self, key: str, count: int):
        """
        将 token 计数存入缓存（LRU 策略）
        """
        self._token_cache[key] = count
        self._token_cache.move_to_end(key)
        
        # 检查缓存大小，必要时淘汰
        if len(self._token_cache) > self.cache_size:
            # 移除最早的条目
            self._token_cache.popitem(last=False)
            self._cache_evictions += 1

    # ====================== 批量计数 ======================

    def count_messages_batch(
        self,
        messages: List[Message],
        use_cache: Optional[bool] = None
    ) -> List[int]:
        """
        批量计算消息的 token 数
        
        参数:
            messages: 消息列表
            use_cache: 是否使用缓存（None 则使用配置的默认值）
        
        返回:
            token 数量列表（与输入顺序一致）
        
        优化：
        1. 可选地复用缓存
        2. 共享 encoder 避免重复创建
        """
        if not messages:
            return []
        
        # 确定是否使用缓存
        should_use_cache = (
            use_cache
            if use_cache is not None
            else self.enable_cache_for_batch
        )
        
        if should_use_cache:
            # 使用缓存：逐个检查/计算
            return [self.count_message_tokens(msg) for msg in messages]
        else:
            # 不使用缓存：直接计算（共享 encoder）
            encode = self.tokenizer.encode
            return [self._count_tokens_impl(msg, encode=encode) for msg in messages]

    def _count_tokens_impl(
        self,
        message: Message,
        encode=None
    ) -> int:
        """
        实际的 token 计数实现
        
        参数:
            message: Message 对象
            encode: 可选的 encoder 函数（复用以提升性能）
        
        返回:
            token 数量
        
        计算规则：
        - 内容 tokens
        - 角色开销（约 4 tokens）
        - tool_calls 开销
        """
        encode = encode or self.tokenizer.encode
        content_tokens = 0
        
        # 计算内容 tokens
        if isinstance(message.content, str):
            content_tokens = len(encode(message.content))
        elif isinstance(message.content, list):
            # 多模态消息：累加文本部分
            text_parts = []
            for block in message.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "image_url":
                    # 图片 token 估算（OpenAI 的图片 token 计算较复杂）
                    # 这里简化为固定值（实际应根据图片分辨率）
                    text_parts.append("[image]")
            
            combined_text = " ".join(text_parts)
            content_tokens = len(encode(combined_text))
        
        # 角色开销（每条消息约 4 tokens）
        overhead = 4
        
        # tool_calls 开销
        if message.tool_calls:
            try:
                tool_calls_str = json.dumps(message.tool_calls)
                overhead += len(encode(tool_calls_str))
            except Exception as e:
                logger.warning("Failed to encode tool_calls", error=str(e))
                overhead += 50  # 估算值
        
        return content_tokens + overhead

    # ====================== 历史加载（带裁剪）======================

    async def get_history(
        self,
        raw_messages: List[dict],
        preserve_system: bool = True
    ) -> List[Message]:
        """
        加载并裁剪历史消息，确保不超过 max_tokens
        
        参数:
            raw_messages: 原始消息字典列表
            preserve_system: 是否优先保留 system 消息
        
        返回:
            Message 列表（已裁剪）
        
        策略：
        1. 解析并验证消息
        2. 限制单条消息长度
        3. 优先保留 system 消息
        4. 从最新消息开始累加，直到达到 token 限制
        """
        if not raw_messages:
            logger.debug("No history messages to load")
            return []
        
        # 1. 解析消息
        messages: List[Message] = []
        for idx, entry in enumerate(raw_messages):
            try:
                msg = Message(**entry)
                
                # 限制单条消息长度（防止极端情况）
                if isinstance(msg.content, str):
                    if len(msg.content) > self.max_message_length:
                        logger.warning(
                            "Message content truncated",
                            original_length=len(msg.content),
                            max_length=self.max_message_length
                        )
                        msg.content = msg.content[:self.max_message_length]
                
                messages.append(msg)
                
            except Exception as e:
                logger.warning(
                    "Invalid history message, skipping",
                    index=idx,
                    error=str(e)
                )
                continue
        
        if not messages:
            logger.warning("All history messages are invalid")
            return []
        
        # 2. 分离 system 消息
        system_msg = None
        other_messages = messages
        
        if preserve_system and messages[0].role == "system":
            system_msg = messages[0]
            other_messages = messages[1:]
        
        # 3. 计算 system 消息的 tokens
        current_tokens = 0
        if system_msg:
            current_tokens = self.count_message_tokens(system_msg)
            
            # 检查 system 消息是否已超过限制
            if current_tokens >= self.max_tokens:
                logger.error(
                    "System message exceeds max_tokens",
                    system_tokens=current_tokens,
                    max_tokens=self.max_tokens
                )
                # 强制截断 system 消息
                if isinstance(system_msg.content, str):
                    truncate_length = int(len(system_msg.content) * self.max_tokens / current_tokens)
                    system_msg.content = system_msg.content[:truncate_length]
                    current_tokens = self.count_message_tokens(system_msg)
        
        # 4. 从最新消息开始累加（倒序）
        selected: List[Message] = []
        
        for msg in reversed(other_messages):
            msg_tokens = self.count_message_tokens(msg)
            
            # 检查是否会超过限制
            if current_tokens + msg_tokens > self.max_tokens:
                logger.debug(
                    "Reached token limit, stopping history loading",
                    current_tokens=current_tokens,
                    max_tokens=self.max_tokens,
                    total_messages=len(other_messages),
                    selected_messages=len(selected)
                )
                break
            
            selected.insert(0, msg)
            current_tokens += msg_tokens
        
        # 5. 重新加入 system 消息
        if system_msg:
            selected.insert(0, system_msg)
        
        logger.info(
            "History loaded",
            session_id=self.session_id,
            total_messages=len(messages),
            selected_messages=len(selected),
            total_tokens=current_tokens,
            max_tokens=self.max_tokens
        )
        
        return selected

    # ====================== 缓存管理 ======================

    def clear_cache(self):
        """清空 token 计数缓存"""
        cleared_size = len(self._token_cache)
        self._token_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        
        logger.info("Token cache cleared", cleared_entries=cleared_size)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        返回:
            包含缓存命中率、大小等信息的字典
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self._token_cache),
            "max_cache_size": self.cache_size,
            "cache_utilization": len(self._token_cache) / self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    def print_cache_stats(self):
        """打印缓存统计信息（格式化输出）"""
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 60)
        print("Token Cache Statistics".center(60))
        print("=" * 60)
        print(f"Cache Size:        {stats['cache_size']} / {stats['max_cache_size']}")
        print(f"Cache Utilization: {stats['cache_utilization']:.1%}")
        print(f"Total Requests:    {stats['total_requests']}")
        print(f"Cache Hits:        {stats['hits']}")
        print(f"Cache Misses:      {stats['misses']}")
        print(f"Cache Evictions:   {stats['evictions']}")
        print(f"Hit Rate:          {stats['hit_rate']:.1%}")
        print("=" * 60 + "\n")

    # ====================== 工具方法 ======================

    def estimate_tokens(self, text: str) -> int:
        """
        快速估算文本的 token 数（不使用缓存）
        
        用于：
        - 快速预估
        - 不需要精确计数的场景
        """
        return len(self.tokenizer.encode(text))

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"TokenMemory("
            f"session_id='{self.session_id}', "
            f"max_tokens={self.max_tokens}, "
            f"model='{self.model_name}', "
            f"cache_size={len(self._token_cache)}/{self.cache_size}"
            f")"
        )