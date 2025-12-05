# gecko/compose/workflow/executor.py
"""
节点执行器 (Node Executor) - v0.4 Enhanced

职责：
1. 负责单个节点（Function / Agent / Team）的调度与执行。
2. 实现“智能参数绑定” (Smart Binding)：根据函数签名或类型提示自动注入 Context 或 Input。
3. 处理重试逻辑 (Retries) 与 异常捕获。
4. [v0.4 关键修复] 识别并透传 Next 控制流指令，防止被错误序列化。
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel

from gecko.core.exceptions import WorkflowError
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.utils import ensure_awaitable
from gecko.compose.nodes import Next
from gecko.compose.workflow.models import WorkflowContext, NodeExecution, NodeStatus

logger = get_logger(__name__)


class NodeExecutor:
    """
    节点执行引擎
    独立于 Workflow 主逻辑，专注于“如何执行一个节点”。
    它是无状态的（Stateless），可以安全地在并发环境中使用。
    """

    def __init__(self, enable_retry: bool = False, max_retries: int = 3):
        self.enable_retry = enable_retry
        self.max_retries = max_retries

    async def execute_node(
        self, 
        node_name: str, 
        node_func: Callable, 
        context: WorkflowContext
    ) -> Any:
        """
        执行节点并返回结果（包含状态记录与异常处理）
        
        Args:
            node_name: 节点名称
            node_func: 节点可调用对象
            context: 工作流上下文 (在并行执行中，这是主 Context 的深拷贝)
            
        Returns:
            执行结果（如果是 Next 指令则原样返回，否则返回标准化后的数据）
        """
        # 1. 记录执行开始状态 (Trace)
        # 注意：持久化层会利用 RUNNING 状态做 Pre-Commit
        execution = NodeExecution(node_name=node_name, status=NodeStatus.RUNNING)
        context.add_execution(execution)
        
        try:
            # 2. 执行核心逻辑 (根据配置决定是否重试)
            if self.enable_retry:
                result = await self._execute_with_retry(node_func, context)
            else:
                result = await self._dispatch_call(node_func, context)
            
            # 3. [v0.4 关键修复] 处理控制流指令 (Next)
            # 如果返回值是 Next 对象，必须原样返回给 Engine，
            # 绝不能进行 _normalize_result，否则 Engine 无法识别跳转意图。
            if isinstance(result, Next):
                execution.output_data = f"<Next -> {result.node}>"
                execution.status = NodeStatus.SUCCESS
                return result
            
            # 4. 结果标准化 (Normalization)
            # 将 Pydantic 对象、Message 对象转为 Dict/Str，便于存储和下游使用
            normalized = self._normalize_result(result)
            
            # 更新执行记录 (仅记录预览，防止大对象爆内存)
            execution.output_data = self._safe_preview(normalized)
            execution.status = NodeStatus.SUCCESS
            
            return normalized

        except Exception as e:
            # 5. 异常捕获
            execution.status = NodeStatus.FAILED
            execution.error = str(e)
            # 记录详细日志，包含节点信息和简要上下文预览
            logger.exception(
                f"Node execution failed: {node_name}",
                node=node_name,
                status=execution.status,
                preview=self._safe_preview(execution.output_data if execution.output_data else execution.error)
            )
            # 将异常包装为 WorkflowError 以便上层识别并保留原始 traceback
            raise WorkflowError(f"Node '{node_name}' execution failed: {e}") from e
        finally:
            execution.end_time = time.time()

    def _normalize_result(self, result: Any) -> Any:
        """
        标准化结果 (Pydantic Friendly)
        
        将复杂对象转换为可 JSON 序列化的结构，防止存储层报错。
        """
        if isinstance(result, BaseModel):
            return result.model_dump()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"): # 兼容 Pydantic v1
            return result.dict()
        if isinstance(result, Message):
            return result.to_openai_format()
        return result

    def _safe_preview(self, data: Any, limit: int = 200) -> str:
        """生成数据的简略预览（用于日志/Trace）"""
        try:
            s = str(data)
            return s[:limit] + "..." if len(s) > limit else s
        except Exception:
            return "<Unprintable>"

    async def _execute_with_retry(self, func: Callable, context: WorkflowContext) -> Any:
        """执行带有指数退避的重试循环"""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._dispatch_call(func, context)
            except Exception as e:
                last_error = e
                # 如果不是最后一次尝试，则记录日志并等待
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Node execution failed, retrying ({attempt+1}/{self.max_retries})", 
                        error=str(e),
                        wait_s=wait_time
                    )
                    await asyncio.sleep(wait_time)
        
        # 重试耗尽，抛出最后一次异常
        raise last_error # type: ignore

    async def _dispatch_call(self, func: Callable, context: WorkflowContext) -> Any:
        """
        智能调度器 (Smart Dispatcher)
        
        根据 func 的类型（Agent 对象 vs 普通函数）和签名，
        自动决定如何注入参数。
        """
        # 1. 智能体对象 (Agent/Team 实现了 run 方法)
        if hasattr(func, "run"):
            return await self._run_intelligent_object(func, context)
            
        # 2. 普通函数 (Function/Lambda)
        return await self._run_function(func, context)

    async def _run_intelligent_object(self, obj: Any, context: WorkflowContext) -> Any:
        """
        运行 Agent/Team 对象
        
        策略：
        - 优先检查 state 中是否有 `_next_input` (由上一个节点的 Next 指令显式传递)。
        - 否则使用 `last_output`。
        - 否则使用 原始 `input`。
        """
        # pop 确保 _next_input 是一次性的
        # 注意：在并行执行中，context 是 deepcopy 的，pop 不会影响其他并发分支，是安全的
        if "_next_input" in context.state:
            inp = context.state.pop("_next_input")
        else:
            inp = context.get_last_output()
            
        return await obj.run(inp)

    async def _run_function(self, func: Callable, context: WorkflowContext) -> Any:
        """
        运行普通函数 (参数注入核心逻辑)
        
        策略：
        1. 分析函数签名。
        2. [v0.4 新增] 优先检查 Type Hint：如果参数类型是 WorkflowContext，注入 Context。
        3. 其次检查参数名：如果包含 `context` 或 `workflow_context`，注入 Context。
        4. 如果还有其他位置参数未被填充，注入当前 Input 数据。
        """
        sig = inspect.signature(func)
        kwargs = {}
        args = []
        
        # 1. 确定当前输入数据
        if "_next_input" in context.state:
            current_input = context.state.pop("_next_input")
        else:
            current_input = context.get_last_output()

        # 遍历参数寻找 Context 注入点
        params_to_skip = set(["self"])
        
        for name, param in sig.parameters.items():
            if name in params_to_skip:
                continue
            
            # 策略 A: 基于类型注解 (Type Hint)
            # 注意: 需要处理字符串注解的情况 (from __future__ import annotations)
            annotation = param.annotation
            is_context_type = False
            
            if annotation is not inspect.Parameter.empty:
                if annotation is WorkflowContext:
                    is_context_type = True
                elif isinstance(annotation, str) and "WorkflowContext" in annotation:
                    is_context_type = True
            
            # 策略 B: 基于参数名
            is_context_name = name in ("context", "workflow_context")
            
            if is_context_type or is_context_name:
                kwargs[name] = context
                params_to_skip.add(name)

        # 重新扫描以注入 Input (排除已注入 Context 的参数)
        # 简化逻辑：排除掉 kwargs 中的参数后，剩下的第一个参数给 Input
        remaining_params = [
            p for name, p in sig.parameters.items()
            if name not in kwargs and name != "self"
        ]
        
        if remaining_params:
            # 将 current_input 作为第一个剩余的位置参数传入
            args.append(current_input)
            
        # 4. 执行 (ensure_awaitable 兼容同步/异步函数)
        return await ensure_awaitable(func, *args, **kwargs)