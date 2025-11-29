# gecko/core/memory/base.py
"""
TokenMemory 基础实现模块

职责：
- 提供 Token 感知的上下文管理能力：
  * Token 计数（tiktoken / model_driver / 字符估算）
  * LRU 缓存 Token 结果
  * 同步 / 异步批量计数
  * 按 max_tokens 对历史消息进行裁剪
  * 多模态（文本 + 图片） Token 估算

本文件对应原 memory.py 中的 TokenMemory 相关逻辑的完整拆分版。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface

from gecko.core.memory._executor import get_token_executor

if TYPE_CHECKING:
    from gecko.core.protocols import ModelProtocol

logger = get_logger(__name__)


class TokenMemory:
    """
    Token 感知的记忆管理器（基础版）

    负责：
    - 在有限的 Context Window 内最大化保留有效对话历史
    - 为上层提供 Token 计数、历史裁剪等能力
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
        model_driver: Optional["ModelProtocol"] = None,
        enable_async_counting: bool = True,
    ):
        """
        初始化 TokenMemory

        参数：
            session_id:
                会话 ID（一般用于区分不同用户/会话）
            storage:
                会话存储接口（当前未使用，预留给后续“自动存取记忆”能力）
            max_tokens:
                上下文最大 Token 限制（系统 + 历史 + 当前输入）
            model_name:
                用于 tiktoken 加载 encoder 的模型名（如：gpt-3.5-turbo, gpt-4 等）
            cache_size:
                LRU 缓存最大条目数（缓存 Key -> Token 数）
            max_message_length:
                单条消息最大字符长度（字符级防御性截断）
            enable_cache_for_batch:
                批量计数时是否默认启用缓存
            model_driver:
                可选的模型驱动，实现 count_tokens 接口，用于精确计数
            enable_async_counting:
                是否启用线程池进行异步批量 Token 计算
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")

        self.session_id = session_id
        self.storage = storage
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.cache_size = cache_size
        self.max_message_length = max_message_length
        self.enable_cache_for_batch = enable_cache_for_batch
        self.model_driver = model_driver
        self.enable_async_counting = enable_async_counting

        # LRU 缓存：key -> token_count
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        # 读写都会发生，因此使用可重入锁保证线程安全
        self._cache_lock = threading.RLock()

        # 缓存统计信息（便于监控与调优）
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

        # 延迟加载的 tokenizer 编码器
        self._encoding: Any = None
        self._tokenizer_failed = False
        self._encode_func: Optional[Callable[[str], List[int]]] = None

    # ====================== Tokenizer ======================

    @property
    def tokenizer(self) -> Any:
        """
        延迟加载 tiktoken encoder

        行为：
        - 优先：tiktoken.encoding_for_model(self.model_name)
        - 找不到模型：使用 cl100k_base 编码器
        - tiktoken 未安装或加载失败：标记 _tokenizer_failed=True，后续走字符估算路径
        """
        if self._encoding is not None:
            return self._encoding

        if self._tokenizer_failed:
            return None

        try:
            import tiktoken

            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(
                    f"Model {self.model_name} not found in tiktoken, using cl100k_base"
                )
                self._encoding = tiktoken.get_encoding("cl100k_base")

            self._encode_func = self._encoding.encode

        except ImportError:
            logger.warning(
                "tiktoken not installed. Token counting will use char estimation."
            )
            self._tokenizer_failed = True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self._tokenizer_failed = True

        return self._encoding

    def _get_encode_func(self) -> Optional[Callable[[str], List[int]]]:
        """
        获取 encode(text) 函数（确保 tokenizer 已加载）

        返回：
            encode 函数，若加载失败则返回 None。
        """
        if self._encode_func is None:
            _ = self.tokenizer  # 触发加载
        return self._encode_func

    # ====================== 缓存操作 ======================

    def _cache_get(self, key: str) -> Optional[int]:
        """
        线程安全的缓存读取（LRU）

        行为：
        - 命中：将 key 移至尾部（表示最近被访问），并增加 _cache_hits。
        - 未命中：返回 None。
        """
        with self._cache_lock:
            if key in self._token_cache:
                self._token_cache.move_to_end(key)
                self._cache_hits += 1
                return self._token_cache[key]
            return None

    def _cache_set(self, key: str, value: int) -> None:
        """
        线程安全的缓存写入（LRU）

        行为：
        - 写入或更新 key 后，将其移到尾部。
        - 认为发生了一次“需要计算”的请求，增加 _cache_misses。
        - 若缓存条目数超过 cache_size，则弹出最老的一条，并增加 _cache_evictions。
        """
        with self._cache_lock:
            self._token_cache[key] = value
            self._token_cache.move_to_end(key)
            self._cache_misses += 1

            while len(self._token_cache) > self.cache_size:
                self._token_cache.popitem(last=False)
                self._cache_evictions += 1

    def _make_cache_key(self, message: Message) -> str:
        """
        生成消息的缓存键（稳定且简短）

        策略：
        - 简单文本消息（无 tool_calls）：
            key = md5(f"{role}:{name}:{content}")
        - 含 tool_calls / 多模态：
            序列化为 JSON（model_dump_json / model_dump），再取 md5。
        """
        if isinstance(message.content, str) and not message.tool_calls:
            raw = f"{message.role}:{message.name or ''}:{message.content}"
            return hashlib.md5(raw.encode("utf-8")).hexdigest()

        try:
            raw_json = message.model_dump_json(
                include={"role", "content", "tool_calls", "name"},
                exclude_none=True,
            )
            return hashlib.md5(raw_json.encode("utf-8")).hexdigest()
        except Exception:
            data = message.model_dump(
                include={"role", "content", "tool_calls", "name"}
            )
            return hashlib.md5(
                json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

    # ====================== 文本截断工具 ======================

    def _truncate_text_to_tokens(self, text: str, limit_tokens: int) -> str:
        """
        将纯文本按 Token 限制进行截断，并返回截断后的文本。

        策略：
        - 无 encode：
            退化为字符估算，按 1 token ≈ 3.5 字符，截断后加后缀。
        - 有 encode：
            使用“二分 + encode 实测”精确控制 Token 数，
            并在截断后追加后缀 "...(truncated)"。
        """
        if not text or limit_tokens <= 0:
            return ""

        encode = self._get_encode_func()
        suffix = "...(truncated)"

        if encode is None:
            # 无 tokenizer：粗略按字符数估算
            char_limit = int(limit_tokens * 3.5)
            if len(text) <= char_limit:
                return text
            return text[:char_limit] + suffix

        # 若完整文本已经在预算内，直接返回
        if len(encode(text)) <= limit_tokens:
            return text

        left, right = 0, len(text)
        best_idx = 0

        # 二分搜索最长前缀
        while left <= right:
            mid = (left + right) // 2
            candidate = text[:mid] + suffix
            token_count = len(encode(candidate))
            if token_count <= limit_tokens:
                best_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        if best_idx == 0:
            # 极端情况：连后缀本身都可能超限，保守处理
            if len(encode(suffix)) <= limit_tokens:
                return suffix
            return ""

        return text[:best_idx] + suffix

    # ====================== 单条计数 ======================

    def count_message_tokens(self, message: Message) -> int:
        """
        计算单条消息的 Token 数（带 LRU 缓存）

        流程：
        1. 生成缓存 key
        2. 若命中缓存，直接返回
        3. 未命中则调用 _count_tokens_impl 计算，并写入缓存
        """
        cache_key = self._make_cache_key(message)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        count = self._count_tokens_impl(message)
        self._cache_set(cache_key, count)
        return count

    # ====================== 批量计数（同步） ======================

    def count_messages_batch(
        self,
        messages: List[Message],
        use_cache: Optional[bool] = None,
    ) -> List[int]:
        """
        批量计算消息的 Token 数（同步版本）

        参数：
            messages:
                消息列表
            use_cache:
                - None：跟随实例级配置 self.enable_cache_for_batch
                - True：强制使用缓存
                - False：强制禁用缓存
        """
        if not messages:
            return []

        should_use_cache = (
            use_cache if use_cache is not None else self.enable_cache_for_batch
        )

        if should_use_cache:
            return [self.count_message_tokens(m) for m in messages]
        else:
            encode_fn = self._get_encode_func()
            return [self._count_tokens_impl(m, encode=encode_fn) for m in messages]

    # ====================== 批量计数（异步） ======================

    async def count_messages_batch_async(
        self,
        messages: List[Message],
        use_cache: Optional[bool] = None,
    ) -> List[int]:
        """
        异步批量计算 Token 数（CPU 密集型计算卸载到线程池）

        参数：
            messages:
                消息列表
            use_cache:
                - None：跟随实例级配置 self.enable_cache_for_batch
                - True：强制使用缓存
                - False：强制禁用缓存
        """
        if not messages:
            return []

        should_use_cache = (
            use_cache if use_cache is not None else self.enable_cache_for_batch
        )

        cache_keys: List[str] = []
        cache_results: List[Optional[int]] = []

        # 第一阶段：批量缓存查询
        for msg in messages:
            key = self._make_cache_key(msg)
            cache_keys.append(key)

            if should_use_cache:
                cached = self._cache_get(key)
                cache_results.append(cached)
            else:
                cache_results.append(None)

        # 若全部命中缓存，直接返回
        if should_use_cache and all(r is not None for r in cache_results):
            return cache_results  # type: ignore

        # 第二阶段：收集需要计算的消息
        compute_indices: List[int] = []
        compute_messages: List[Message] = []

        for i, cached in enumerate(cache_results):
            if cached is None:
                compute_indices.append(i)
                compute_messages.append(messages[i])

        # 第三阶段：线程池中计算
        if compute_messages:
            computed = await self._compute_in_thread(compute_messages)

            # 第四阶段：结果回填 + 更新缓存
            for i, idx in enumerate(compute_indices):
                count = computed[i]
                cache_results[idx] = count
                if should_use_cache:
                    self._cache_set(cache_keys[idx], count)

        return cache_results  # type: ignore

    async def _compute_in_thread(self, messages: List[Message]) -> List[int]:
        """
        在线程池中执行 Token 计算。

        关键点：
        - 显式传入 encode_fn，确保在子线程中不会走 model_driver.count_tokens，
          避免在子线程里做网络调用或非线程安全操作。
        - 子线程中只进行本地 encode / 字符估算，属于纯 CPU 计算。
        """
        encode_fn = self._get_encode_func()

        def _compute() -> List[int]:
            return [self._count_tokens_impl(m, encode=encode_fn) for m in messages]

        if self.enable_async_counting:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(get_token_executor(), _compute)
        else:
            return _compute()

    # ====================== Token 计算实现 ======================

    def _count_tokens_impl(
        self,
        message: Message,
        encode: Optional[Callable[[str], List[int]]] = None,
    ) -> int:
        """
        内部 Token 计算核心逻辑。

        优先级：
        1. 若 encode 为 None 且存在 model_driver：
           尝试调用 model_driver.count_tokens 进行精确计数。
        2. 若有 encode，则用 encode 精确计算。
        3. 否则退化为基于字符长度的大致估算。
        """
        # 1) 模型驱动计数（仅在未传入 encode 时使用）
        if self.model_driver is not None and encode is None:
            try:
                return self.model_driver.count_tokens([message.to_openai_format()])
            except Exception:
                # 若模型驱动计数失败，则退回本地估算
                pass

        # 2) 获取 encode 函数
        if encode is None:
            encode = self._get_encode_func()

        if encode is None:
            # 3) 字符估算路径
            # 粗略估计：1 token ≈ 4 个字符，并额外加上一些固定开销
            return len(message.get_text_content()) // 4 + 4

        num_tokens = 4  # per-message overhead（参考 OpenAI 官方估算）

        content = message.content

        if isinstance(content, str):
            num_tokens += len(encode(content))
        elif isinstance(content, list):
            # 多模态 / 结构化内容
            for block in content:
                # 支持对象与 dict 两种形式
                b_type = getattr(block, "type", None)
                b_text = getattr(block, "text", None)
                b_image = getattr(block, "image_url", None)

                if isinstance(block, dict):
                    b_type = block.get("type")
                    b_text = block.get("text")
                    b_image = block.get("image_url")

                if b_type == "text" and b_text:
                    num_tokens += len(encode(b_text))
                elif b_type == "image_url":
                    num_tokens += self._estimate_image_tokens(b_image)

        # tool_calls 计数
        if message.tool_calls:
            try:
                dump = json.dumps(message.tool_calls, ensure_ascii=False)
                num_tokens += len(encode(dump))
            except Exception:
                # 若序列化失败，给一个保守估值
                num_tokens += 100

        # name 开销
        if message.name:
            num_tokens += 1

        return num_tokens

    def _estimate_image_tokens(self, image_resource: Any) -> int:
        """
        粗略估算图片 Token 占用。

        说明：
        - detail == "low" 时使用较小估值（如缩略图、低分辨率）
        - 其他情况使用较大估值（偏保守，以避免低估）
        """
        if not image_resource:
            return 0
        detail = getattr(image_resource, "detail", "auto")
        if detail == "low":
            return 85
        return 1000

    # ====================== 历史加载 ======================

    async def get_history(
        self,
        raw_messages: List[Dict[str, Any]],
        preserve_system: bool = True,
    ) -> List[Message]:
        """
        加载并裁剪历史消息（基础版本）

        算法：
        1. 将 raw_messages（dict 列表）解析为 Message 对象，并做防御性截断。
        2. 若首条为 system 且 preserve_system=True，则单独保留为 system_msg。
        3. 对 system_msg 进行 Token 计数，若超 max_tokens，则强制截断。
        4. 从尾部开始累加剩余消息的 Token 数，直到即将超过 max_tokens 为止。
        5. 返回结果为：[system_msg?] + 最近若干条消息。
        """
        if not raw_messages:
            return []

        parsed: List[Message] = []
        for i, raw in enumerate(raw_messages):
            try:
                msg = Message(**raw)
                self._truncate_if_needed(msg)
                parsed.append(msg)
            except Exception as e:
                logger.warning(f"Skipping invalid message at index {i}: {e}")

        if not parsed:
            return []

        system_msg: Optional[Message] = None
        candidates = parsed

        if preserve_system and parsed[0].role == "system":
            system_msg = parsed[0]
            candidates = parsed[1:]

        used_tokens = 0
        if system_msg:
            sys_tokens = self.count_message_tokens(system_msg)
            if sys_tokens > self.max_tokens:
                self._force_truncate(system_msg, self.max_tokens)
                sys_tokens = self.count_message_tokens(system_msg)
            used_tokens = sys_tokens

        selected: List[Message] = []

        for msg in reversed(candidates):
            tokens = self.count_message_tokens(msg)
            if used_tokens + tokens > self.max_tokens:
                break
            selected.append(msg)
            used_tokens += tokens

        selected.reverse()

        result: List[Message] = []
        if system_msg:
            result.append(system_msg)
        result.extend(selected)
        return result

    def _truncate_if_needed(self, message: Message) -> None:
        """
        防御性截断（按字符数限制单条消息）

        说明：
        - 只对纯文本 content 生效
        - 用于避免异常超长输入导致内存/性能问题
        - 精确控制 Token 数的工作由 `_force_truncate` 负责
        """
        if isinstance(message.content, str):
            if len(message.content) > self.max_message_length:
                message.content = message.content[:self.max_message_length]

    def _force_truncate(self, message: Message, limit_tokens: int) -> None:
        """
        强制将消息内容截断到指定 Token 数以内（仅对纯文本 content 生效）
        """
        if not isinstance(message.content, str):
            return
        message.content = self._truncate_text_to_tokens(message.content, limit_tokens)

    # ====================== 工具方法 ======================

    def count_total_tokens(self, messages: List[Message]) -> int:
        """
        计算消息列表总 Token 数（使用批量计数，默认启用缓存）
        """
        return sum(self.count_messages_batch(messages, use_cache=True))

    def estimate_tokens(self, text: str) -> int:
        """
        快速估算一段文本的 Token 数。

        策略：
        - 有 encode：使用 encode 精算
        - 无 encode：退化为 len(text) // 4
        """
        if not text:
            return 0
        encode = self._get_encode_func()
        if encode:
            return len(encode(text))
        return len(text) // 4

    def clear_cache(self) -> None:
        """
        清空 Token 缓存及其统计数据。
        """
        with self._cache_lock:
            cleared = len(self._token_cache)
            self._token_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_evictions = 0
        logger.info(f"Token cache cleared, {cleared} entries removed")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息（可用于监控与调优）
        """
        with self._cache_lock:
            size = len(self._token_cache)
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0

            return {
                "cache_size": size,
                "max_cache_size": self.cache_size,
                "cache_utilization": size / self.cache_size if self.cache_size > 0 else 0,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
                "total_requests": total,
                "hit_rate": hit_rate,
            }

    def print_cache_stats(self) -> None:
        """
        打印缓存统计信息到标准输出（用于临时调试）
        """
        stats = self.get_cache_stats()
        print("\n" + "=" * 50)
        print("Token Cache Statistics".center(50))
        print("=" * 50)
        print(f"  Size:        {stats['cache_size']} / {stats['max_cache_size']}")
        print(f"  Utilization: {stats['cache_utilization']:.1%}")
        print(f"  Hits:        {stats['hits']}")
        print(f"  Misses:      {stats['misses']}")
        print(f"  Evictions:   {stats['evictions']}")
        print(f"  Hit Rate:    {stats['hit_rate']:.1%}")
        print("=" * 50 + "\n")

    def __repr__(self) -> str:
        return (
            f"TokenMemory(session_id='{self.session_id}', "
            f"max_tokens={self.max_tokens}, "
            f"cache={len(self._token_cache)}/{self.cache_size})"
        )
