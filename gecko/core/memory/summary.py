# gecko/core/memory/summary.py
"""
SummaryTokenMemory 摘要记忆实现模块

在 TokenMemory 的基础上扩展：
- 当历史超过 Token 限制时，不是简单丢弃旧消息，
  而是把较早的消息用 LLM 汇总成摘要，以 system 消息的形式注入上下文。
"""

from __future__ import annotations

import asyncio
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
    支持自动摘要的记忆管理器。

    设计目标：
    - 在 Token 限制内，尽可能保留“语义信息”而不仅仅是“最近 N 条对话”；
    - 对较久远的历史进行“语义压缩”，并持续滚动更新摘要；
    - 在 get_history 返回的上下文中，以 system 消息形式携带摘要。
    """

    def __init__(
        self,
        session_id: str,
        model: "ModelProtocol",
        summary_prompt: Optional[str] = None,
        summary_reserve_tokens: int = 500,
        **kwargs,
    ):
        """
        初始化 SummaryTokenMemory

        参数：
            session_id:
                会话 ID
            model:
                实现 acompletion / count_tokens 的模型驱动
            summary_prompt:
                自定义的摘要 Prompt 模板（可选）
            summary_reserve_tokens:
                为“摘要 + recent 消息”预留的 Token 预算（软预算，不是硬限制）

        说明：
        - 这里会将 model 作为 model_driver 传递给基类，
          以复用模型自身的 count_tokens 能力（单条计数时使用）。
        """
        # 将模型驱动作为 model_driver 传给基类
        kwargs.setdefault("model_driver", model)
        super().__init__(session_id, **kwargs)

        self.model = model
        self.summary_reserve_tokens = summary_reserve_tokens
        self.current_summary: str = ""

        # 延迟导入 PromptTemplate，避免模块级循环依赖
        from gecko.core.prompt import PromptTemplate  # type: ignore

        self.summary_template: "PromptTemplate" = PromptTemplate(
            template=summary_prompt
            or (
                "Condense the following conversation into a brief summary, "
                "preserving key information:\n\n{{ history }}\n\nSummary:"
            ),
            input_variables=["history"],
        )

        # 懒加载的 asyncio.Lock，用于确保并发更新摘要时串行执行
        self._summary_lock: Optional[asyncio.Lock] = None

    def _get_summary_lock(self) -> asyncio.Lock:
        """
        获取摘要更新锁（懒加载）

        说明：
        - 在多协程环境下，可能多个请求同时触发摘要更新；
        - 使用 Lock 确保同时只有一个协程在更新 current_summary，
          避免摘要刷新顺序错乱或覆盖问题。
        """
        if self._summary_lock is None:
            self._summary_lock = asyncio.Lock()
        return self._summary_lock

    async def get_history(
        self,
        raw_messages: List[Dict[str, Any]],
        preserve_system: bool = True,
    ) -> List[Message]:
        """
        重写历史加载逻辑，加入摘要机制。

        流程：
        1. 解析 raw_messages → Message，并做防御性截断。
        2. 若首条为 system 且 preserve_system=True，则单独保留 system_msg。
        3. 计算系统消息 Token，必要时强制截断。
        4. 按 max_tokens 和 summary_reserve_tokens 计算“可用于 recent 的预算”。
        5. 从后往前选取 recent 消息，直至用完预算；
           超出的旧消息归为 to_summarize，用于生成摘要。
        6. 若 to_summarize 非空，调用 LLM 更新 current_summary。
        7. 计算实际可用于“摘要 system 消息”的 Token 预算：
           max_tokens - sys_tokens - recent_tokens；
           若预算 > 0，则对摘要文本进行 Token 级截断并注入到上下文中。
        8. 返回结果：[system?] + [summary?] + recent。
        """
        if not raw_messages:
            return []

        # 1. 解析消息（防御性 + 日志）
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

        if preserve_system and parsed[0].role == "system":
            system_msg = parsed[0]
            candidates = parsed[1:]

        # 3. 计算系统消息 Token，并在必要时强制截断
        sys_tokens = 0
        if system_msg:
            sys_tokens = self.count_message_tokens(system_msg)
            if sys_tokens > self.max_tokens:
                # 若系统消息本身就超过上限，则进行强制截断
                self._force_truncate(system_msg, self.max_tokens)
                sys_tokens = self.count_message_tokens(system_msg)

        # 4. 计算“预留 + 可用” Token
        reserved = self.summary_reserve_tokens + sys_tokens
        available = self.max_tokens - reserved
        if available < 0:
            logger.warning(
                f"[SummaryTokenMemory] reserved tokens ({reserved}) exceed max_tokens "
                f"({self.max_tokens}), no tokens left for recent messages."
            )
            available = 0

        used = 0
        recent: List[Message] = []
        to_summarize: List[Message] = []

        # 从后往前拆分出 recent 与 to_summarize
        for msg in reversed(candidates):
            tokens = self.count_message_tokens(msg)
            if used + tokens <= available:
                recent.append(msg)
                used += tokens
            else:
                to_summarize.append(msg)

        recent.reverse()
        to_summarize.reverse()

        # 5. 更新摘要（若有需要摘要的旧消息）
        if to_summarize:
            await self._update_summary(to_summarize)

        # 6. 组装最终结果
        result: List[Message] = []
        if system_msg:
            result.append(system_msg)

        # 将摘要以 system 消息形式注入，但保证不超过整体 Token 上限
        if self.current_summary:
            summary_text = f"Previous context: {self.current_summary}"
            summary_budget = max(self.max_tokens - sys_tokens - used, 0)

            if summary_budget > 0:
                truncated_summary = self._truncate_text_to_tokens(summary_text, summary_budget)
                if truncated_summary:
                    result.append(Message.system(truncated_summary))
            else:
                logger.warning(
                    "[SummaryTokenMemory] No token budget for summary, "
                    "skipping summary message in history."
                )

        result.extend(recent)
        return result

    async def _update_summary(self, messages: List[Message]) -> None:
        """
        内部方法：基于待摘要消息列表更新 current_summary。

        处理逻辑：
        - 将 messages 转成 "role: content" 文本（只取文本部分）。
        - 若已有旧摘要，则按：
              Previous: {old}
              
              New:
              {history_text}
          的方式合并，再交给模型进行“二次压缩”。
        - 使用 summary_template 生成最终 prompt。
        - 调用 model.acompletion 获取摘要内容，并覆盖 current_summary。
        - 使用 asyncio.Lock 保证并发情况下摘要更新是串行的。
        """
        if not messages:
            return

        lock = self._get_summary_lock()
        async with lock:
            history_text = "\n".join(
                f"{m.role}: {m.get_text_content()}" for m in messages
            )

            if self.current_summary:
                history_text = f"Previous: {self.current_summary}\n\nNew:\n{history_text}"

            prompt = self.summary_template.format(history=history_text)

            try:
                # 假设返回结构与 OpenAI 对齐：choices[0].message.content
                response = await self.model.acompletion(
                    [{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.get("content", "")
                if content:
                    self.current_summary = content
                    logger.info(f"Summary updated, length: {len(content)}")
            except Exception as e:
                logger.error(f"Failed to update summary: {e}")

    def clear_summary(self) -> None:
        """
        清除当前摘要内容。

        使用场景：
        - 会话被显式“重置”时
        - 测试场景希望从零开始观测摘要行为时
        """
        self.current_summary = ""
