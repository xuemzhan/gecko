# gecko/compose/workflow/executor.py
"""
节点执行器 (Node Executor)

职责：
1. 负责单个节点（Function / Agent / Team）的调度与执行。
2. 实现“智能参数绑定” (Smart Binding)：根据函数签名自动注入 Context 或 Input。
3. 处理重试逻辑 (Retries) 与 异常捕获。
4. 结果标准化 (Normalization)：将各种返回值统一为可序列化的格式。
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
        
        参数:
            node_name: 节点名称
            node_func: 节点可调用对象
            context: 工作流上下文
            
        返回:
            执行结果（如果是 Next 指令则原样返回，否则返回标准化后的数据）
        """
        # 1. 记录执行开始状态 (Trace)
        # 注意：此时 status=RUNNING，持久化层会利用此状态做 Pre-Commit
        execution = NodeExecution(node_name=node_name, status=NodeStatus.RUNNING)
        context.add_execution(execution)
        
        try:
            # 2. 执行核心逻辑 (带重试)
            if self.enable_retry:
                result = await self._execute_with_retry(node_func, context)
            else:
                result = await self._dispatch_call(node_func, context)
            
            # 3. 处理控制流指令 (Next)
            if isinstance(result, Next):
                execution.output_data = f"<Next -> {result.node}>"
                execution.status = NodeStatus.SUCCESS
                # Next 指令不进行标准化，直接交给 Engine 处理跳转
                return result
            
            # 4. 结果标准化 (Normalization)
            # 将 Pydantic 对象、Message 对象转为 Dict/Str，便于存储和下游使用
            normalized = self._normalize_result(result)
            
            # 更新执行记录
            execution.output_data = self._safe_preview(normalized)
            execution.status = NodeStatus.SUCCESS
            
            return normalized

        except Exception as e:
            # 5. 异常捕获
            execution.status = NodeStatus.FAILED
            execution.error = str(e)
            # 这里抛出异常，由上层 Engine 决定是中断还是处理
            raise e
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
        for attempt in range(self.max_retries):
            try:
                return await self._dispatch_call(func, context)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Node execution failed, retrying ({attempt+1}/{self.max_retries})", 
                    error=str(e)
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
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
        """
        # pop 确保 _next_input 是一次性的
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
        2. 如果参数名包含 `context` 或 `workflow_context`，注入 WorkflowContext 对象。
        3. 如果还有其他位置参数未被填充，注入当前 Input 数据。
        """
        sig = inspect.signature(func)
        kwargs = {}
        args = []
        
        # 1. 确定当前输入数据
        if "_next_input" in context.state:
            current_input = context.state.pop("_next_input")
        else:
            current_input = context.get_last_output()

        # 2. 注入 Context (支持多种命名习惯)
        if "context" in sig.parameters:
            kwargs["context"] = context
        elif "workflow_context" in sig.parameters:
            # [Fix] 兼容旧版本测试用例中使用的参数名
            kwargs["workflow_context"] = context
        
        # 3. 注入 Input (Input Injection)
        # 排除掉 self, context, workflow_context 之后，如果还有参数，则认为第一个参数是 Input
        remaining = [
            p for p in sig.parameters 
            if p not in kwargs and p != "self"
        ]
        
        if remaining:
            # 将 current_input 作为第一个位置参数传入
            args.append(current_input)
            
        # 4. 执行 (ensure_awaitable 兼容同步/异步函数)
        return await ensure_awaitable(func, *args, **kwargs)