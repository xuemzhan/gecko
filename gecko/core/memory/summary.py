# gecko/core/memory/summary.py
"""
SummaryTokenMemory 摘要记忆实现模块 (Production Optimized)

优化日志：
- [Perf] 引入 min_update_interval (防抖)，避免频繁调用 LLM。
- [Perf] 支持 background_update (后台更新)，防止摘要生成阻塞主对话流程。
- [Safety] 保持 asyncio.Lock 确保同一时刻只有一个摘要任务在运行。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.memory.base import TokenMemory

if TYPE_CHECKING:
    from gecko.core.protocols import ModelProtocol
    from gecko.core.prompt import PromptTemplate

logger = get_logger(__name__)


class SummaryTokenMemory(TokenMemory):
    """
    支持自动摘要的记忆管理器 (高性能版)。
    """

    def __init__(
        self,
        session_id: str,
        model: "ModelProtocol",
        summary_prompt: Optional[str] = None,
        summary_reserve_tokens: int = 500,
        # [新增] 生产级配置
        min_update_interval: float = 30.0,  # 两次摘要更新的最小间隔(秒)
        background_update: bool = True,     # 是否在后台异步更新摘要(推荐True)
        **kwargs,
    ):
        """
        初始化 SummaryTokenMemory

        参数：
            min_update_interval: 防抖窗口，默认 30 秒内只会触发一次摘要更新。
            background_update: 若为 True，get_history 不会等待摘要生成，直接返回旧摘要+新历史。
        """
        # 将模型驱动作为 model_driver 传给基类
        kwargs.setdefault("model_driver", model)
        super().__init__(session_id, **kwargs)

        self.model = model
        self.summary_reserve_tokens = summary_reserve_tokens
        self.current_summary: str = ""
        
        # [新增] 控制参数
        self.min_update_interval = min_update_interval
        self.background_update = background_update
        self._last_update_time: float = 0.0

        # 延迟导入 PromptTemplate
        from gecko.core.prompt import PromptTemplate  # type: ignore

        self.summary_template: "PromptTemplate" = PromptTemplate(
            template=summary_prompt
            or (
                "Condense the following conversation into a brief summary, "
                "preserving key information:\n\n{{ history }}\n\nSummary:"
            ),
            input_variables=["history"],
        )

        self._summary_lock: Optional[asyncio.Lock] = None

        # [新增] 保存后台任务句柄，便于：
        # 1) 避免重复创建任务
        # 2) 支持取消/关闭（工业级生命周期治理）
        self._update_task: Optional[asyncio.Task] = None

        # [新增] pending：当锁占用时不丢更新意图，只保留最后一份（last-write-wins）
        self._pending_messages: Optional[List[Message]] = None

    def _get_summary_lock(self) -> asyncio.Lock:
        if self._summary_lock is None:
            self._summary_lock = asyncio.Lock()
        return self._summary_lock

    async def get_history(
        self,
        raw_messages: List[Dict[str, Any]],
        preserve_system: bool = True,
    ) -> List[Message]:
        """
        获取历史消息 (优化版)
        """
        if not raw_messages:
            return []

        # 1. 解析消息
        parsed: List[Message] = []
        for i, raw in enumerate(raw_messages):
            try:
                msg = Message(**raw)
                self._truncate_if_needed(msg)
                parsed.append(msg)
            except Exception as e:
                logger.warning(f"[SummaryTokenMemory] Skipping invalid message at index {i}: {e}")

        if not parsed:
            return []

        # 2. 分离 system 消息
        system_msg: Optional[Message] = None
        candidates = parsed

        if preserve_system and parsed[0].role == "system": # type: ignore
            system_msg = parsed[0]
            candidates = parsed[1:]

        # 3. 计算预算
        sys_tokens = 0
        if system_msg:
            sys_tokens = self.count_message_tokens(system_msg)
            if sys_tokens > self.max_tokens:
                self._force_truncate(system_msg, self.max_tokens)
                sys_tokens = self.count_message_tokens(system_msg)

        reserved = self.summary_reserve_tokens + sys_tokens
        available = self.max_tokens - reserved
        if available < 0:
            available = 0

        # 4. 拆分 Recent / ToSummarize
        used = 0
        recent: List[Message] = []
        to_summarize: List[Message] = []

        for msg in reversed(candidates):
            tokens = self.count_message_tokens(msg)
            if used + tokens <= available:
                recent.append(msg)
                used += tokens
            else:
                to_summarize.append(msg)

        recent.reverse()
        to_summarize.reverse()

        # 5. [核心] 触发摘要更新（to_summarize 是本次需要摘要的内容）
        if to_summarize:
            await self._trigger_summary_update(to_summarize)

        # [二次加固] 如果之前因为 lock/task 占用产生了 pending，
        # 在本次 get_history 中“顺手尝试”消费。
        # 注意：必须仍经过 debounce，避免破坏现有单元测试语义。
        if self._pending_messages:
            now = time.time()
            lock = self._get_summary_lock()
            task_running = self._update_task is not None and not self._update_task.done()
            if self._should_update_summary(now) and (not lock.locked()) and (not task_running):
                pending = self._pending_messages
                self._pending_messages = None
                await self._trigger_summary_update(pending)

        # 6. 组装结果
        result: List[Message] = []
        if system_msg:
            result.append(system_msg)

        # 注入摘要 (使用当前内存中已有的摘要)
        if self.current_summary:
            summary_text = f"Previous context: {self.current_summary}"
            # 动态计算剩余给摘要的预算
            summary_budget = max(self.max_tokens - sys_tokens - used, 0)

            if summary_budget > 0:
                truncated_summary = self._truncate_text_to_tokens(summary_text, summary_budget)
                if truncated_summary:
                    result.append(Message.system(truncated_summary))

        result.extend(recent)
        return result

    async def _trigger_summary_update(self, messages: List[Message]) -> None:
        """
        触发摘要更新（含防抖 + 后台执行 + pending 合并）

        改进点：
        1) debounce：未到时间直接跳过（保持原有语义）
        2) 若 lock/task 正在运行：不丢弃，而是把 messages 记为 pending（保留最新）
        3) background_update：保存 Task 句柄，避免重复创建且支持取消
        4) 二次加固：在“调度/启动更新”时写入 _last_update_time，防失败时反复重试
        """
        now = time.time()

        # 1) 防抖检查
        if not self._should_update_summary(now):
            logger.debug("Summary update skipped (debounce)")
            return

        lock = self._get_summary_lock()

        # 2) 如果锁占用或已有后台任务在跑：合并 pending（last-write-wins）
        task_running = self._update_task is not None and not self._update_task.done()
        if lock.locked() or task_running:
            logger.debug("Summary update deferred (locked/task running)")
            self._pending_messages = messages
            return

        # 3) 准备执行：在调度时更新 last_update_time（避免失败时被频繁重试）
        self._last_update_time = now

        if self.background_update:
            # 4) 后台执行：保存任务句柄，便于关闭时 cancel/drain
            self._update_task = asyncio.create_task(self._update_summary_task(messages))
        else:
            # 5) 同步执行：保持旧行为
            await self._update_summary_task(messages)


    async def _update_summary_task(self, messages: List[Message]) -> None:
        """
        实际执行摘要更新。

        重要约束（与单元测试语义一致）：
        - 不在任务结束时自动触发 pending（否则会绕过 debounce，导致额外 LLM 调用）
        - pending 仅在后续 get_history() 中、且 debounce 通过时才会被消费
        """
        lock = self._get_summary_lock()

        async with lock:
            try:
                logger.debug(f"Updating summary for {len(messages)} messages...")

                history_text = "\n".join(
                    f"{m.role}: {m.get_text_content()}" for m in messages
                )

                if self.current_summary:
                    history_text = f"Previous: {self.current_summary}\n\nNew:\n{history_text}"

                prompt = self.summary_template.format(history=history_text)

                response = await self.model.acompletion(
                    [{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.get("content", "")

                if content:
                    self.current_summary = content
                    logger.info(f"Summary updated successfully, length: {len(content)}")

            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to update summary: {e}")

            finally:
                # 清理 task 句柄：避免引用悬挂（并为下一次调度做准备）
                # 注意：这里不触发 pending，保持 debounce + 触发式更新模型
                if self._update_task is not None and self._update_task.done():
                    self._update_task = None


    def clear_summary(self) -> None:
        self.current_summary = ""
        self._last_update_time = 0.0

    def _should_update_summary(self, now: Optional[float] = None) -> bool:
        """
        判断是否允许触发摘要更新（防抖）。

        设计说明：
        - 使用 min_update_interval 控制更新频率，防止高频触发导致成本/延迟暴涨
        - now 参数用于测试或减少重复 time.time() 调用
        """
        if now is None:
            now = time.time()
        return (now - self._last_update_time) >= self.min_update_interval
    
    async def aclose(self) -> None:
        """
        关闭 Memory（工业级生命周期治理）。

        场景：
        - 服务退出 / 会话结束时，取消后台摘要任务，避免 pending task 泄漏
        """
        task = self._update_task
        if task is None:
            return

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # 关闭阶段不应再抛异常影响主流程
                pass
        self._update_task = None