# gecko/core/memory.py
"""
Token Memory - 上下文记忆管理器

负责管理对话历史的 Token 计数、裁剪与缓存。

核心功能：
1. 精确计数：基于 tiktoken 的模型特定 Token 计算
2. 智能裁剪：基于滑动窗口 (Sliding Window) 的历史记录加载
3. 性能缓存：LRU 缓存 Token 计算结果，减少重复计算开销
4. 多模态支持：估算图片/文件的 Token 占用
5. 完备的工具链：批量计算、统计打印、快速估算

优化日志：
- [Perf] get_history 采用 O(N) 算法 (append + reverse)
- [Perf] 缓存键生成使用 model_dump_json 加速
- [Fix] 补全所有原始方法 (print_cache_stats, batch optimizations)
- [Fix] 修正统计键名以通过单元测试
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.prompt import PromptTemplate
from gecko.core.protocols import ModelProtocol
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)


class TokenMemory:
    """
    Token 感知的记忆管理器
    
    负责在有限的 Context Window 内最大化保留有效对话历史。
    """

    def __init__(
        self,
        session_id: str,
        storage: Optional[SessionInterface] = None,
        max_tokens: int = 4000,
        model_name: str = "gpt-3.5-turbo",
        cache_size: int = 2000,
        max_message_length: int = 20000,
        enable_cache_for_batch: bool = True,
    ):
        """
        初始化 Memory
        
        参数:
            session_id: 会话唯一标识
            storage: 持久化存储后端
            max_tokens: 最大上下文 Token 限制
            model_name: 模型名称 (用于加载 tokenizer)
            cache_size: Token 计数缓存大小 (LRU)
            max_message_length: 单条消息最大字符数 (防御性截断)
            enable_cache_for_batch: 批量计算时是否启用缓存
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")

        self.session_id = session_id
        self.storage = storage
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.cache_size = cache_size  # 公开属性
        self.max_message_length = max_message_length
        self.enable_cache_for_batch = enable_cache_for_batch
        
        # LRU 缓存: Hash(Content) -> TokenCount
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        
        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        
        # 延迟加载的 tokenizer
        self._encoding = None
        self._tokenizer_failed = False

    # ====================== Tokenizer ======================

    @property
    def tokenizer(self):
        """延迟加载 tiktoken encoder"""
        if self._encoding:
            return self._encoding
        
        if self._tokenizer_failed:
            return None

        try:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(f"Model {self.model_name} not found in tiktoken, using cl100k_base")
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed. Token counting will be estimated by char length.")
            self._tokenizer_failed = True
            return None
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self._tokenizer_failed = True
            return None
            
        return self._encoding

    # ====================== 单条计数 ======================

    def count_message_tokens(self, message: Message) -> int:
        """
        计算单条消息的 Token 数（带缓存）
        """
        # 1. 生成缓存键
        cache_key = self._make_cache_key(message)
        
        # 2. 检查缓存
        if cache_key in self._token_cache:
            self._token_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._token_cache[cache_key]
        
        # 3. 计算
        self._cache_misses += 1
        count = self._count_tokens_impl(message)
        
        # 4. 更新缓存 (LRU)
        self._token_cache[cache_key] = count
        self._token_cache.move_to_end(cache_key)
        
        if len(self._token_cache) > self.cache_size:
            self._token_cache.popitem(last=False)
            self._cache_evictions += 1
            
        return count

    def _make_cache_key(self, message: Message) -> str:
        """
        生成消息的缓存键
        
        优化: 使用 Pydantic model_dump_json (Rust) 加速序列化
        """
        # 快速路径: 普通文本消息直接哈希
        if isinstance(message.content, str) and not message.tool_calls:
            raw = f"{message.role}:{message.name}:{message.content}"
            return hashlib.md5(raw.encode("utf-8")).hexdigest()
            
        # 慢速路径: 多模态或工具调用
        # exclude_none=True 减少数据量，sort_keys=True (默认False) 在 dump_json 中不支持，
        # 但 Pydantic 字段顺序通常是固定的。为了绝对安全，可以用 json.dumps(model_dump)
        # 不过对于缓存键，model_dump_json 通常足够稳定且快。
        try:
            raw_json = message.model_dump_json(
                include={"role", "content", "tool_calls", "name"},
                exclude_none=True
            )
            return hashlib.md5(raw_json.encode("utf-8")).hexdigest()
        except Exception:
            # 降级方案
            data = message.model_dump(include={"role", "content", "tool_calls", "name"})
            return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    # ====================== 批量计数 ======================

    def count_messages_batch(
        self,
        messages: List[Message],
        use_cache: Optional[bool] = None
    ) -> List[int]:
        """
        批量计算消息的 Token 数
        
        优化: 复用 encoder 对象，减少属性查找开销
        """
        if not messages:
            return []
        
        should_use_cache = (
            use_cache if use_cache is not None else self.enable_cache_for_batch
        )
        
        if should_use_cache:
            return [self.count_message_tokens(msg) for msg in messages]
        else:
            # 性能优化: 提取 encode 方法，避免在循环中重复查找 self.tokenizer
            encode_fn = self.tokenizer.encode if self.tokenizer else None
            return [self._count_tokens_impl(msg, encode=encode_fn) for msg in messages]

    def _count_tokens_impl(
        self,
        message: Message,
        encode: Optional[Callable[[str], List[int]]] = None
    ) -> int:
        """
        实际 Token 计算逻辑
        
        参数:
            message: 消息对象
            encode: 可选的编码函数（性能优化用）
        """
        if not encode:
            if self.tokenizer:
                encode = self.tokenizer.encode
            else:
                # 降级: 字符估算
                return len(message.get_text_content()) // 4 + 2

        num_tokens = 4  # Per-message overhead
        
        # 1. Content Tokens
        if isinstance(message.content, str):
            num_tokens += len(encode(message.content))
        elif isinstance(message.content, list):
            for block in message.content:
                if block.type == "text" and block.text:
                    num_tokens += len(encode(block.text))
                elif block.type == "image_url":
                    num_tokens += self._estimate_image_tokens(block.image_url)

        # 2. Tool Calls Overhead
        if message.tool_calls:
            try:
                # 使用快速序列化
                dump = self._fast_json_dumps(message.tool_calls)
                num_tokens += len(encode(dump))
            except Exception:
                num_tokens += 100

        # 3. Name Overhead
        if message.name:
            num_tokens += 1

        return num_tokens

    # ====================== 历史加载 ======================

    async def get_history(
        self,
        raw_messages: List[Dict[str, Any]],
        preserve_system: bool = True
    ) -> List[Message]:
        """
        加载并裁剪历史消息 (O(N) 复杂度优化版)
        """
        if not raw_messages:
            logger.debug("No history messages to load")
            return []

        # 1. 解析消息 (单次遍历)
        parsed_messages: List[Message] = []
        for i, raw in enumerate(raw_messages):
            try:
                msg = Message(**raw)
                self._truncate_message_safety(msg)
                parsed_messages.append(msg)
            except Exception as e:
                logger.warning("Skipping invalid message history", index=i, error=str(e))

        if not parsed_messages:
            return []

        # 2. 分离 System Message
        system_msg: Optional[Message] = None
        candidates = parsed_messages
        
        if preserve_system and parsed_messages[0].role == "system":
            system_msg = parsed_messages[0]
            candidates = parsed_messages[1:]

        # 3. 计算 System Token 开销
        current_tokens = 0
        if system_msg:
            sys_tokens = self.count_message_tokens(system_msg)
            if sys_tokens > self.max_tokens:
                logger.warning("System prompt exceeds max_tokens, force truncating")
                self._truncate_to_fit(system_msg, self.max_tokens)
                sys_tokens = self.count_message_tokens(system_msg)
            current_tokens += sys_tokens

        # 4. 反向回填 (Reverse Accumulation - O(N))
        selected_reverse: List[Message] = []
        
        for msg in reversed(candidates):
            tokens = self.count_message_tokens(msg)
            
            if current_tokens + tokens > self.max_tokens:
                logger.debug(
                    "Context limit reached", 
                    current=current_tokens, 
                    limit=self.max_tokens
                )
                break
            
            selected_reverse.append(msg)
            current_tokens += tokens

        # 5. 重组列表
        result = []
        if system_msg:
            result.append(system_msg)
        
        # 再次反转回时间正序
        result.extend(reversed(selected_reverse))
        
        return result

    # ====================== 辅助方法 ======================

    def clear_cache(self):
        """清空计数缓存"""
        cleared_size = len(self._token_cache)
        self._token_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        logger.info("Token cache cleared", cleared_entries=cleared_size)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        [Fix] 键名修正为 'cache_size' 以符合测试预期
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0.0
        
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

    def estimate_tokens(self, text: str) -> int:
        """快速估算文本的 token 数（不使用缓存）"""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4

    # ====================== 内部工具 ======================

    def _fast_json_dumps(self, obj: Any) -> str:
        """快速 JSON 序列化"""
        try:
            import orjson
            return orjson.dumps(obj).decode("utf-8")
        except ImportError:
            import json
            return json.dumps(obj)

    def _estimate_image_tokens(self, image_resource: Any) -> int:
        """估算图片 Token"""
        if not image_resource:
            return 0
        detail = getattr(image_resource, "detail", "auto")
        if detail == "low":
            return 85
        return 1000 

    def _truncate_message_safety(self, message: Message):
        """防御性截断"""
        if isinstance(message.content, str):
            if len(message.content) > self.max_message_length:
                message.content = message.content[:self.max_message_length]

    def _truncate_to_fit(self, message: Message, limit_tokens: int):
        """强行截断文本"""
        if not isinstance(message.content, str):
            return
        char_limit = int(limit_tokens * 3.5)
        if len(message.content) > char_limit:
            message.content = message.content[:char_limit] + "...(truncated)"

    def __repr__(self) -> str:
        return (
            f"TokenMemory("
            f"session_id='{self.session_id}', "
            f"max_tokens={self.max_tokens}, "
            f"model='{self.model_name}', "
            f"cache_size={len(self._token_cache)}/{self.cache_size}"
            f")"
        )
    
class SummaryTokenMemory(TokenMemory):
    """
    支持自动摘要的记忆管理器
    
    当历史记录超出 max_tokens 时，不是简单丢弃，而是调用 LLM 对早期历史进行摘要。
    """
    
    def __init__(
        self,
        session_id: str,
        model: ModelProtocol,  # 需要传入模型实例用于摘要
        summary_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(session_id, **kwargs)
        self.model = model
        # 默认摘要模板
        self.summary_template = PromptTemplate(
            template=summary_prompt or (
                "Please condense the following conversation history into a concise summary, "
                "preserving key information and context:\n\n{{ history }}\n\nSummary:"
            ),
            input_variables=["history"]
        )
        # 内存中维护当前的摘要
        self.current_summary: str = ""

    async def get_history(
        self,
        raw_messages: List[Dict[str, Any]],
        preserve_system: bool = True
    ) -> List[Message]:
        """
        重写历史加载逻辑：
        1. 加载原始消息
        2. 如果 Token 超限，触发摘要流程
        3. 返回 [System, Summary, ...Recent Messages]
        """
        # 1. 先用父类逻辑加载和清洗基础消息
        messages = await super().get_history(raw_messages, preserve_system)
        
        # 计算总 Token (此时 messages 已经被父类截断过一次，但那是硬截断)
        # 我们需要基于未截断的完整列表来做决策，或者在此处再次检查
        # 这里简化逻辑：如果父类已经截断了，我们可能丢失了信息。
        # 更理想的是完全重写 get_history，但为了复用，我们这里主要处理 "摘要注入"。
        
        # 实际上，TokenMemory.get_history 的逻辑是反向回填直到满。
        # 这意味着早期的消息已经被丢弃了。
        # 为了实现摘要，我们需要改变策略：
        # 不直接丢弃，而是把丢弃的部分拿来做摘要。
        
        # === 重写核心逻辑 ===
        if not raw_messages:
            return []

        parsed_messages = []
        for raw in raw_messages:
            try:
                msg = Message(**raw)
                self._truncate_message_safety(msg)
                parsed_messages.append(msg)
            except Exception:
                pass

        if not parsed_messages:
            return []

        # 分离 System
        system_msg = None
        candidates = parsed_messages
        if preserve_system and parsed_messages[0].role == "system":
            system_msg = parsed_messages[0]
            candidates = parsed_messages[1:]

        # 预留 System 和 Summary 的 Token 空间 (估算 500 tokens)
        reserved_tokens = 500
        if system_msg:
            reserved_tokens += self.count_message_tokens(system_msg)
        
        current_tokens = 0
        recent_messages = []
        to_summarize = []

        # 反向选取最近的消息
        for msg in reversed(candidates):
            tokens = self.count_message_tokens(msg)
            if current_tokens + tokens > (self.max_tokens - reserved_tokens):
                # 超出窗口，归入待摘要队列
                to_summarize.append(msg)
            else:
                recent_messages.append(msg)
                current_tokens += tokens
        
        recent_messages.reverse() # 恢复时间正序
        to_summarize.reverse()    # 恢复时间正序

        # 如果有需要摘要的消息，生成或更新摘要
        if to_summarize:
            await self._update_summary(to_summarize)

        # 组装最终历史
        final_history = []
        if system_msg:
            final_history.append(system_msg)
        
        # 注入摘要 (作为 System 消息或 User 提示)
        if self.current_summary:
            summary_msg = Message.system(f"Previous conversation summary: {self.current_summary}")
            final_history.append(summary_msg)
            
        final_history.extend(recent_messages)
        
        return final_history

    async def _update_summary(self, messages: List[Message]):
        """调用 LLM 更新摘要"""
        if not messages:
            return

        # 将消息转为文本
        history_text = "\n".join([f"{m.role}: {m.get_text_content()}" for m in messages])
        
        # 如果已有摘要，将其合并进去
        if self.current_summary:
            history_text = f"Previous Summary: {self.current_summary}\n\nNew Conversation:\n{history_text}"

        # 构造 Prompt
        prompt = self.summary_template.format(history=history_text)
        
        try:
            # 调用模型生成摘要
            # 注意：这里是一个简单的阻塞调用，生产环境可能需要后台异步更新
            response = await self.model.acompletion([{"role": "user", "content": prompt}])
            new_summary = response.choices[0].message["content"]
            if new_summary:
                self.current_summary = new_summary
                logger.info("Conversation summary updated", summary_len=len(new_summary))
        except Exception as e:
            logger.error("Failed to update summary", error=str(e))
            # 失败时保持原有摘要，或者不做处理