# gecko/compose/workflow/persistence.py
"""
持久化管理器

职责：
1. 负责 WorkflowContext 的序列化与反序列化
2. 调用 safe_serialize_context 进行数据清洗 (CPU Bound -> ThreadPool)
3. 执行异步存储操作 (IO Bound)
"""
from __future__ import annotations

import time
from typing import Optional, Any

from anyio.to_thread import run_sync

from gecko.core.logging import get_logger
from gecko.core.utils import safe_serialize_context
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.compose.workflow.models import WorkflowContext, CheckpointStrategy

logger = get_logger(__name__)


class PersistenceManager:
    def __init__(
        self, 
        storage: Optional[SessionInterface],
        strategy: CheckpointStrategy = CheckpointStrategy.ALWAYS,
        history_retention: int = 20
    ):
        self.storage = storage
        self.strategy = strategy
        self.history_retention = history_retention

    async def save_checkpoint(
        self,
        session_id: str,
        steps: int,
        current_node: Optional[str],
        context: WorkflowContext,
        force: bool = False
    ):
        """
        保存检查点 (优化版：瘦身 + 异步清洗)
        """
        if not self.storage or not session_id:
            return

        # 策略检查
        if not force:
            if self.strategy == CheckpointStrategy.MANUAL:
                return
            if self.strategy == CheckpointStrategy.FINAL:
                return

        try:
            # 1. 上下文瘦身 (Context Slimming)
            # 获取纯 Python 字典，移除冗余轨迹和过早历史
            raw_data = context.to_storage_payload(max_history_steps=self.history_retention)
            
            # 2. 深度清洗 (CPU 密集型 -> 卸载到线程池)
            # 清理不可序列化对象 (Lock, Socket 等)
            def _clean_task():
                return safe_serialize_context(raw_data)
            
            clean_context_data = await run_sync(_clean_task)

            # 3. 写入存储 (IO 密集型)
            payload = {
                "step": steps,
                "last_node": current_node,
                "context": clean_context_data,
                "updated_at": time.time(),
            }
            
            await self.storage.set(f"workflow:{session_id}", payload)
            
        except Exception as e:
            logger.warning("Failed to persist workflow state", session_id=session_id, error=str(e))

    async def load_checkpoint(self, session_id: str) -> Optional[dict]:
        """加载检查点数据"""
        if not self.storage:
            return None
        return await self.storage.get(f"workflow:{session_id}")