# gecko/compose/workflow/models.py
"""
Workflow 数据模型定义

职责：
1. 定义节点状态枚举 (NodeStatus)
2. 定义持久化策略枚举 (CheckpointStrategy)
3. 定义执行上下文 (WorkflowContext)，并实现核心的“瘦身”逻辑
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

T = TypeVar("T")


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointStrategy(str, Enum):
    """
    持久化策略
    """
    ALWAYS = "always"  # 每步保存 (开发环境推荐)
    FINAL = "final"    # 仅结束时保存 (生产环境高性能推荐)
    MANUAL = "manual"  # 手动控制


class NodeExecution(BaseModel):
    """节点执行轨迹记录 (Trace)"""
    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return max(0.0, self.end_time - self.start_time)


class WorkflowContext(BaseModel):
    """
    工作流执行上下文
    
    优化：
    实现了 `to_storage_payload` 方法，用于在持久化时剥离监控数据和冗余历史，
    解决生产环境下的 "State Bloat" (状态爆炸) 问题。
    """
    execution_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="单次运行的唯一 ID"
    )
    input: Any = Field(..., description="工作流初始输入")
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="共享状态存储（业务核心数据，全量保存）"
    )
    history: Dict[str, Any] = Field(
        default_factory=dict, 
        description="节点历史输出记录（可能很大）"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据（如 session_id, trace_id）"
    )
    executions: List[NodeExecution] = Field(
        default_factory=list,
        description="完整执行轨迹（监控数据，持久化时可剥离）"
    )
    next_pointer: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Next 指令产生的动态跳转目标"
    )

    def add_execution(self, execution: NodeExecution):
        self.executions.append(execution)

    def get_last_output(self) -> Any:
        return self.history.get("last_output", self.input)
    
    def clear_next_pointer(self):
        self.next_pointer = None

    # ================= 核心优化：上下文瘦身 =================

    def to_storage_payload(self, max_history_steps: int = 10) -> Dict[str, Any]:
        """
        [High Priority Fix] 生成用于持久化的轻量级数据包
        
        策略：
        1. 移除 `executions`：轨迹属于监控数据，不存入 Redis/DB 这种运行态存储。
        2. 裁剪 `history`：仅保留最近 N 步，防止 O(N) 的 IO 膨胀。
        """
        # 1. 基础导出，排除 executions
        payload = self.model_dump(mode='python', exclude={'executions'})
        
        # 2. 裁剪 history
        if max_history_steps > 0 and len(self.history) > max_history_steps:
            all_keys = list(self.history.keys())
            keep_keys = all_keys[-max_history_steps:]
            
            # 必须保留 last_output，它是 Next 节点的默认输入
            if "last_output" in self.history and "last_output" not in keep_keys:
                keep_keys.append("last_output")
                
            payload['history'] = {k: self.history[k] for k in keep_keys}
            
        return payload

    @classmethod
    def from_storage_payload(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """从存储数据重建 Context，自动补全缺失字段"""
        if "executions" not in data:
            data["executions"] = []
        return cls.model_validate(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        total_time = sum(e.duration for e in self.executions)
        is_failed = any(e.status == NodeStatus.FAILED for e in self.executions)
        return {
            "execution_id": self.execution_id,
            "total_nodes": len(self.executions),
            "total_time": total_time,
            "last_node": self.executions[-1].node_name if self.executions else None,
            "status": "failed" if is_failed else "completed"
        }

    def get_last_output_as(self, type_: Type[T]) -> T:
        """类型安全地获取上一步输出"""
        val = self.get_last_output()
        
        # 1. 直接类型匹配
        if isinstance(val, type_):
            return val
            
        # 2. Pydantic 转换
        if isinstance(val, dict) and hasattr(type_, "model_validate"):
            try:
                return type_.model_validate(val) # type: ignore
            except Exception:
                pass
                
        # 3. 简单类型转换
        try:
            return type_(val) # type: ignore
        except Exception as e:
            raise TypeError(f"Cannot convert last output {type(val)} to {type_}") from e