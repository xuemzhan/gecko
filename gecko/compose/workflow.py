# gecko/compose/workflow.py
"""
Workflow 引擎

提供基于 DAG（有向无环图）的任务编排能力，支持复杂的控制流和状态管理。

核心功能：
1. 节点编排：支持普通函数、Agent、Team 等多种节点类型混编
2. 状态管理：基于 Pydantic 的强类型上下文，支持完整的序列化与持久化
3. 控制流：支持条件分支、循环（通过 Next 指令）
4. 智能绑定：自动根据函数签名注入 Context 或 Input
5. 可观测性：详细的节点执行轨迹与统计

优化日志：
- [Fix] 修复参数注入逻辑，支持同时接收 Input 和 Context 的函数签名
- [Fix] 将 WorkflowContext 升级为 BaseModel，彻底解决持久化数据截断问题
- [Fix] 增加分支歧义检测，确保逻辑确定性
- [Fix] 优化 Agent 间的数据流转 (Data Handover)，避免 Prompt 污染

优化日志：
- [Feat] 支持 allow_cycles 配置，允许定义有环图
- [Feat] 支持处理 Next.update_state
- [Feat] 新增 CheckpointStrategy 控制持久化频率
- [Feat] 重构 _execute_loop 支持自定义起始点
- [Feat] 新增 resume 方法实现断点恢复
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
# 明确导入 run_sync，避免模块/函数混淆
from anyio.to_thread import run_sync
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, Set

from pydantic import BaseModel, Field, PrivateAttr

from gecko.compose.nodes import Next
from gecko.core.events import BaseEvent, EventBus
from gecko.core.exceptions import WorkflowCycleError, WorkflowError
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.utils import ensure_awaitable, safe_serialize_context
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)

T = TypeVar("T")


# ========================= 事件定义 =========================

class WorkflowEvent(BaseEvent):
    """Workflow 专用事件对象"""
    pass


# ========================= 状态模型 =========================

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeExecution(BaseModel):
    """
    节点执行记录（轨迹追踪）
    """
    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """计算执行耗时"""
        if self.end_time == 0.0:
            return 0.0
        return max(0.0, self.end_time - self.start_time)


class WorkflowContext(BaseModel):
    """
    工作流执行上下文
    
    使用 Pydantic 模型确保类型安全和原生序列化支持。
    """
    execution_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="单次运行的唯一 ID"
    )
    input: Any = Field(..., description="工作流初始输入")
    state: Dict[str, Any] = Field(
        default_factory=dict, 
        description="共享状态存储（用户自定义）"
    )
    history: Dict[str, Any] = Field(
        default_factory=dict, 
        description="节点历史输出记录"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据（如 session_id, trace_id）"
    )
    executions: List[NodeExecution] = Field(
        default_factory=list,
        description="完整执行轨迹"
    )
    # [新增] 动态跳转指针，用于断点恢复
    # 格式: {"target_node": "NodeB", "input": ...}
    next_pointer: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Next 指令产生的动态跳转目标"
    )

    def clear_next_pointer(self):
        """消费完指针后清除"""
        self.next_pointer = None

    def add_execution(self, execution: NodeExecution):
        """添加执行记录"""
        self.executions.append(execution)

    def get_last_output(self) -> Any:
        """获取上一个节点的输出（如果无历史则返回初始输入）"""
        return self.history.get("last_output", self.input)
    
    def get_last_output_as(self, type_: Type[T]) -> T:
        """
        类型安全地获取上一步输出
        尝试将输出转换为指定类型，或进行断言。
        """
        val = self.get_last_output()
        
        # 1. 如果已经是该类型，直接返回
        if isinstance(val, type_):
            return val
            
        # 2. 尝试 Pydantic 转换 (如果是 dict -> Model)
        if isinstance(val, dict) and hasattr(type_, "model_validate"):
            try:
                return type_.model_validate(val) # type: ignore
            except Exception:
                pass
                
        # 3. 简单的类型转换 (如 int, str)
        try:
            return type_(val) # type: ignore
        except Exception as e:
            raise TypeError(f"Cannot convert last output {type(val)} to {type_}") from e

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
    
    # 辅助方法，用于持久化时计算当前步数
    def get_step_count(self) -> int:
        """获取当前执行步数（基于执行记录数量）"""
        return len(self.executions)

# [New] 持久化策略枚举
class CheckpointStrategy(str, Enum):
    ALWAYS = "always"  # 每步保存 (默认, 最安全)
    FINAL = "final"    # 仅在结束时保存 (性能最好)
    MANUAL = "manual"  # 不自动保存 (需用户手动处理)

# ========================= 工作流引擎 =========================

class Workflow:
    """
    DAG 工作流引擎
    """

    def __init__(
        self,
        name: str = "Workflow",
        event_bus: Optional[EventBus] = None,
        storage: Optional[SessionInterface] = None,
        max_steps: int = 100,
        enable_retry: bool = False,
        max_retries: int = 3,
        allow_cycles: bool = False, # [New] 允许循环
        checkpoint_strategy: Union[str, CheckpointStrategy] = CheckpointStrategy.ALWAYS, # [New]
    ):
        self.name = name
        self.event_bus = event_bus or EventBus()
        self.storage = storage
        self.max_steps = max_steps
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.allow_cycles = allow_cycles 
        self.checkpoint_strategy = CheckpointStrategy(checkpoint_strategy)

        # 内部存储
        self._nodes: Dict[str, Callable] = {}
        # edges: source -> [(target, condition_func), ...]
        self._edges: Dict[str, List[Tuple[str, Optional[Callable]]]] = {}
        self._entry_point: Optional[str] = None

        # 验证状态
        self._validated = False
        self._validation_errors: List[str] = []

    # ========================= 构建 API =========================

    def add_node(self, name: str, func: Callable) -> "Workflow":
        """
        添加节点
        
        参数:
            name: 节点唯一名称
            func: 可调用对象（函数、Agent、Team）
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        self._nodes[name] = func
        self._validated = False
        logger.debug("Node added", workflow=self.name, node=name)
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[WorkflowContext], bool]] = None,
    ) -> "Workflow":
        """
        添加边（支持条件分支）
        
        参数:
            source: 源节点名称
            target: 目标节点名称
            condition: 转移条件函数（接收 Context 返回 bool）
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        self._edges.setdefault(source, []).append((target, condition))
        self._validated = False
        logger.debug("Edge added", source=source, target=target, conditional=bool(condition))
        return self

    def set_entry_point(self, name: str) -> "Workflow":
        """设置入口节点"""
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        self._entry_point = name
        self._validated = False
        return self

    # ========================= 验证逻辑 =========================

    def validate(self) -> bool:
        """验证工作流结构的合法性"""
        if self._validated:
            return len(self._validation_errors) == 0

        self._validation_errors.clear()

        # 1. 检查入口
        if not self._entry_point:
            self._validation_errors.append("No entry point defined")
        elif self._entry_point not in self._nodes:
            self._validation_errors.append(f"Entry point '{self._entry_point}' not in nodes")

        # 2. 检查歧义分支 (同一节点存在多个无条件出边)
        for node, edges in self._edges.items():
            unconditional_edges = [t for t, c in edges if c is None]
            if len(unconditional_edges) > 1:
                self._validation_errors.append(
                    f"Node '{node}' has ambiguous edges: multiple unconditional targets {unconditional_edges}"
                )

        # 3. 检查死循环 (静态检测)
        # [Updated] 仅在不允许循环时检测环
        if not self.allow_cycles:
            try:
                self._detect_cycles()
            except WorkflowCycleError as e:
                self._validation_errors.append(str(e))

        # 4. 连通性警告
        self._check_connectivity()

        self._validated = True
        if self._validation_errors:
            logger.error("Workflow validation failed", errors=self._validation_errors)
            return False

        logger.info("Workflow validation passed", name=self.name)
        return True

    def _detect_cycles(self):
        """DFS 检测环"""
        visited = set()
        recursion_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            for neighbor, _ in self._edges.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in recursion_stack:
                    cycle_start = path.index(neighbor)
                    cycle = " -> ".join(path[cycle_start:] + [neighbor])
                    raise WorkflowCycleError(f"Cycle detected: {cycle}")

            recursion_stack.remove(node)
            path.pop()

        for node in self._nodes:
            if node not in visited:
                dfs(node, [])

    def _check_connectivity(self):
        """检查不可达节点（仅警告）"""
        if not self._entry_point:
            return

        reachable = set()
        queue = [self._entry_point]
        while queue:
            curr = queue.pop(0)
            if curr in reachable:
                continue
            reachable.add(curr)
            for target, _ in self._edges.get(curr, []):
                queue.append(target)
        
        unreachable = set(self._nodes.keys()) - reachable
        if unreachable:
            logger.warning("Unreachable nodes detected", nodes=list(unreachable))

    # ========================= 执行引擎 =========================

    async def execute(self, input_data: Any, session_id: Optional[str] = None) -> Any:
        """
        执行工作流
        
        参数:
            input_data: 初始输入数据
            session_id: 会话 ID（用于持久化和状态恢复）
            
        返回:
            最终输出（Context 中的 last_output）
        """
        if not self.validate():
            raise WorkflowError(f"Workflow validation failed:\n" + "\n".join(self._validation_errors))

        # 初始化上下文
        context = WorkflowContext(input=input_data)
        if session_id:
            context.metadata["session_id"] = session_id

        await self.event_bus.publish(
            WorkflowEvent(
                type="workflow_started", 
                data={"name": self.name, "execution_id": context.execution_id}
            )
        )

        try:
            # [Update] 默认从入口点开始，步数为 0
            await self._execute_loop(context, session_id, start_node=self._entry_point, start_step=0)

            # 最终保存 (如果策略是 FINAL)
            if self.storage and session_id and self.checkpoint_strategy == CheckpointStrategy.FINAL:
                await self._persist_state(session_id, 9999, None, context, force=True)

            # await self._execute_loop(context, session_id)
            
            result = context.get_last_output()
            await self.event_bus.publish(
                WorkflowEvent(
                    type="workflow_completed",
                    data={"name": self.name, "summary": context.get_summary()},
                )
            )
            return result
            
        except Exception as e:
            logger.exception("Workflow execution failed")
            await self.event_bus.publish(
                WorkflowEvent(type="workflow_error", error=str(e), data={"name": self.name})
            )
            raise

    async def resume(self, session_id: str) -> Any:
        """
        从存储中恢复执行
        
        修复: 延迟清除 next_pointer，确保首步执行成功后再清除
        """
        if not self.storage:
            raise ValueError("Cannot resume: Storage not configured")
        
        # 1. 加载状态
        saved_data = await self.storage.get(f"workflow:{session_id}")
        if not saved_data:
            raise ValueError(f"Session '{session_id}' not found")
        
        logger.info("Resuming workflow", session_id=session_id, last_node=saved_data.get("last_node"))
        
        # 2. 重建 Context
        try:
            context = WorkflowContext(**saved_data["context"])
        except Exception as e:
            raise WorkflowError(f"Failed to reconstruct context: {e}") from e
            
        last_node = saved_data.get("last_node")
        current_step = saved_data.get("step", 0)
        
        # 3. 确定下一步
        next_node = None
        # ✅ 修复: 使用标记位追踪是否从 next_pointer 恢复，而不是立即清除
        resumed_from_pointer = False

        if context.next_pointer:
            logger.info("Resuming from dynamic Next pointer", target=context.next_pointer.get("target_node"))
            next_node = context.next_pointer.get("target_node")
            
            # 恢复可能存在的输入传递
            if context.next_pointer.get("input"):
                context.state["_next_input"] = context.next_pointer["input"]
            
            # ✅ 修复: 标记而不是立即清除，让 _execute_loop 在成功执行后清除
            resumed_from_pointer = True

        elif last_node:
            next_node = await self._find_next_node(last_node, context)
            if not next_node:
                logger.info("Workflow already completed (no next node)", session_id=session_id)
                return context.get_last_output()
        else:
            next_node = self._entry_point
        
        # 4. 继续执行循环
        try:
            await self._execute_loop(
                context, 
                session_id, 
                start_node=next_node, 
                start_step=current_step,
                clear_pointer_after_first_step=resumed_from_pointer  # ✅ 新增参数
            )
            
            # 最终保存
            if self.checkpoint_strategy == CheckpointStrategy.FINAL:
                await self._persist_state(session_id, current_step, None, context, force=True)
                
            return context.get_last_output()
            
        except Exception as e:
            logger.exception("Resume execution failed")
            raise

    async def _execute_loop(
        self, 
        context: WorkflowContext,  
        session_id: Optional[str], 
        start_node: Optional[str],
        start_step: int,
        clear_pointer_after_first_step: bool = False  # ✅ 新增参数
    ):
        """核心执行循环"""
        current_node = start_node
        steps = start_step
        is_first_step = True  # ✅ 追踪是否为首步

        while current_node and steps < self.max_steps:
            steps += 1

            # 执行节点
            logger.debug("Executing step", step=steps, node=current_node)
            result = await self._execute_node_safe(current_node, context)

            # ✅ 修复: 首步成功后清除 pointer
            if is_first_step and clear_pointer_after_first_step:
                context.clear_next_pointer()
                is_first_step = False

            # 准备持久化所需的临时变量
            persist_node = current_node 
            next_target = None

            # 处理流转逻辑
            if isinstance(result, Next):
                next_target = result.node
                
                if result.input is not None:
                    normalized = self._normalize_result(result.input)
                    context.history["last_output"] = normalized
                    context.state["_next_input"] = normalized
                
                if result.update_state:
                    context.state.update(result.update_state)
                    
                context.next_pointer = {
                    "target_node": next_target,
                    "input": context.state.get("_next_input")
                }
                
                context.history[current_node] = f"<Next -> {next_target}>"

            else:
                normalized = self._normalize_result(result)
                context.history[current_node] = normalized
                context.history["last_output"] = normalized
                
                context.clear_next_pointer()
                
                next_target = await self._find_next_node(current_node, context)

            # 持久化
            if self.storage and session_id:
                await self._persist_state(session_id, steps, persist_node, context)

            current_node = next_target

        if steps >= self.max_steps:
            raise WorkflowError(f"Exceeded max steps: {self.max_steps}", context={"last": current_node})

    async def _execute_node_safe(self, node_name: str, context: WorkflowContext) -> Any:
        """
        节点执行包装器：负责状态记录、重试和错误处理
        
        [优化] 引入两阶段检查点 (Two-Phase Checkpoint):
        1. Pre-Commit: 标记节点为 RUNNING 并持久化。防止 Crash 后状态丢失。
        2. Execution: 执行业务逻辑。
        3. Commit: 标记节点为 SUCCESS/FAILED 并持久化。
        """
        # 1. 状态初始化：标记为运行中
        execution = NodeExecution(node_name=node_name, status=NodeStatus.RUNNING)
        context.add_execution(execution)
        
        await self.event_bus.publish(WorkflowEvent(type="node_started", data={"node": node_name}))

        # [新增] Phase 1: Pre-Commit 持久化
        # 如果配置了存储且有 session_id，强制保存当前 RUNNING 状态
        # 这样如果随后的 _run_any_node 导致进程崩溃，Resume 时我们会看到一个 RUNNING 状态的节点
        session_id = context.metadata.get("session_id")
        
        # [新增] Phase 1: Pre-Commit (预写日志)
        # 强制持久化当前 "RUNNING" 状态。即便策略是 FINAL，为了数据安全也必须记录中间态。
        if self.storage and session_id:
            try:
                # force=True 确保即使策略是 FINAL 也能记录中间态（为了数据安全）
                # 注意：这会增加 IO 开销，但在分布式系统中是必要的
                # 使用 context.get_step_count() 获取当前步数
                await self._persist_state(
                    session_id, 
                    context.get_step_count(), 
                    node_name, 
                    context, 
                    force=True # 强制写入
                )
            except Exception as e:
                logger.warning(f"Failed to pre-persist node state: {e}")

        try:
            node_func = self._nodes[node_name]
            
            # 2. 执行业务逻辑
            if self.enable_retry:
                result = await self._execute_with_retry(node_func, context)
            else:
                result = await self._run_any_node(node_func, context)
            
            # 处理控制流指令
            if isinstance(result, Next):
                execution.output_data = f"Next(node={result.node})"
                execution.status = NodeStatus.SUCCESS # 控制流本身算执行成功
            else:
                normalized = self._normalize_result(result)
                execution.output_data = normalized
                result = normalized # 更新 result 为标准化后的数据
                execution.status = NodeStatus.SUCCESS
            
            await self.event_bus.publish(
                WorkflowEvent(
                    type="node_completed", 
                    data={"node": node_name, "duration": execution.duration}
                )
            )
            
            # (Phase 2 Commit 持久化由 _execute_loop 中的循环末尾处理)
            return result

        except Exception as e:
            # 3. 异常处理
            execution.status = NodeStatus.FAILED
            execution.error = str(e)
            
            # [新增] 发生错误时立即持久化 (Commit Failed State)
            # 确保错误现场被保留，方便调试或人工干预
            if self.storage and session_id:
                await self._persist_state(
                    session_id, 
                    context.get_step_count(), 
                    node_name, 
                    context, 
                    force=True
                )
            
            await self.event_bus.publish(
                WorkflowEvent(type="node_error", error=str(e), data={"node": node_name})
            )
            raise WorkflowError(f"Node '{node_name}' failed: {e}") from e
            
        finally:
            execution.end_time = time.time()
            # 注意：context.add_execution 已在开头调用，引用已存在，无需再次添加

    # ========================= 节点调度逻辑 (核心) =========================

    async def _run_any_node(self, node_callable: Callable, context: WorkflowContext) -> Any:
        """
        通用节点执行器 (Duck Typing & Smart Binding)
        
        支持:
        1. Agent/Team (具备 run 方法的对象)
        2. 普通函数 (自动注入 context/input/none)
        
        修复：智能处理参数绑定，支持同时需要 Input 和 Context 的场景
        """
        # 1. 智能体对象 (Agent or Team)
        if hasattr(node_callable, "run") and callable(node_callable.run): # type: ignore
            return await self._run_intelligent_object(node_callable, context)
        
        # 2. 普通可调用对象
        if callable(node_callable):
            sig = inspect.signature(node_callable)
            params = sig.parameters
            kwargs = {}
            args = []
            
            # 获取当前输入数据
            current_input = context.state.pop("_next_input", None) or context.get_last_output()
            
            # A. 注入 Context
            if "context" in params:
                kwargs["context"] = context
            elif "workflow_context" in params:
                kwargs["workflow_context"] = context
                
            # B. 注入 Input (Input Injection)
            # 找到除 context, self 之外的参数，通常是第一个位置参数或特定的参数名
            # 这里采取简单策略：如果有剩余参数位置，则将 Input 作为第一个位置参数传入
            
            # 过滤掉已处理的 context 参数
            remaining_params = [
                name for name, p in params.items() 
                if name not in ("context", "workflow_context", "self")
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            
            if remaining_params:
                # 如果有剩余参数，假设第一个是用来接收 input 的
                args.append(current_input)
            elif not kwargs:
                # 如果既不需要 Context 也没有其他参数（无参函数），直接调用
                pass
                
            # 执行
            return await ensure_awaitable(node_callable, *args, **kwargs)
            
        raise WorkflowError(f"Node '{node_callable}' is not callable")

    async def _run_intelligent_object(self, obj: Any, context: WorkflowContext) -> Any:
        """
        执行 Agent/Team 对象
        
        优化: 智能处理数据流转 (Data Handover)
        优化: [Refactored Phase 1] 移除数据流转的魔法清洗逻辑
        """
        # 1. 获取输入 (优先使用 Next 传递的，否则用上一步输出)
        raw_input = context.state.pop("_next_input", None) or context.get_last_output()
        
        # 2. 数据清洗 (Data Handover Fix)
        ## 以前这里会尝试从 dict 中提取 content。
        ## 现在我们假设 Agent.run 能够处理 raw_input (因为 Agent.run 支持 dict 输入)
        ## 或者上一个节点的输出就是 Agent 期望的格式。
        # 如果上一个节点返回的是 AgentOutput (dict)，且当前 Agent 需要文本输入，
        # 我们尝试提取 content，避免将整个 JSON 结构扔给 LLM。
        # agent_input = raw_input
        # if isinstance(raw_input, dict):
        #     # 如果有 content 且没有 role (说明不是 Message 对象，而是 Output 字典)
        #     if "content" in raw_input and "role" not in raw_input:
        #         agent_input = raw_input["content"]
        
        # 3. 执行
        output = await obj.run(raw_input)
        
        # 4. 结果处理
        if hasattr(output, "model_dump"):
            return output.model_dump()
        return output

    async def _execute_with_retry(self, func: Callable, context: WorkflowContext) -> Any:
        """重试逻辑"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self._run_any_node(func, context)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Node retry triggered",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_error # type: ignore

    # ========================= 辅助方法 =========================

    async def _find_next_node(self, current: str, context: WorkflowContext) -> Optional[str]:
        """
        寻找下一个节点
        
        包含歧义检测：不允许同时满足多个无条件路径
        """
        edges = self._edges.get(current, [])
        candidates = []
        
        for target, condition in edges:
            should_go = False
            if condition is None:
                should_go = True
            else:
                try:
                    if inspect.iscoroutinefunction(condition):
                        should_go = await condition(context)
                    else:
                        should_go = condition(context)
                except Exception as e:
                    logger.error("Condition evaluation failed", source=current, target=target, error=str(e))
            
            if should_go:
                candidates.append(target)
        
        if len(candidates) == 0:
            return None
        
        # 歧义检测
        if len(candidates) > 1:
            raise WorkflowError(
                f"Ambiguous branching from '{current}': multiple conditions met for targets {candidates}. "
                "Workflow logic must be deterministic."
            )
            
        return candidates[0]

    def _normalize_result(self, result: Any) -> Any:
        """
        标准化结果 (Pydantic Friendly)
        """
        if isinstance(result, BaseModel):
            return result.model_dump()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, Message):
            return result.to_openai_format()
        return result

    async def _persist_state(
        self,
        session_id: str,
        steps: int,
        current_node: Optional[str],
        context: WorkflowContext,
        force: bool = False
    ):
        """
        状态持久化
        
        使用 Pydantic 的 mode='json' 确保完整序列化，不进行截断。
        参数 force: 是否强制保存（忽略策略）

        [优化] 异步非阻塞状态持久化
        1. 获取原始数据 (mode='python') 避免 Pydantic 报错。
        2. 在线程池中执行深层清洗 (CPU 密集型)。
        3. 异步写入存储。
        """
        # 如果策略是 MANUAL 且非强制，跳过
        if not force and self.checkpoint_strategy == CheckpointStrategy.MANUAL:
            return
            
        # 如果策略是 FINAL 且非强制，也跳过 (FINAL 只在 execute 结束或 resume 结束时 force=True 调用)
        if not force and self.checkpoint_strategy == CheckpointStrategy.FINAL:
            return

        try:
            # 1. 快速获取 Python 原生字典 (包含未序列化的 Lock 等对象)
            # 这一步很快，因为不涉及 JSON 转换
            raw_data = context.model_dump(mode='python')
            
            # 2. [核心优化] 将耗时的清洗工作卸载到线程池
            # 避免 Context 很大时阻塞 Event Loop
            def _heavy_clean_task():
                return safe_serialize_context(raw_data)
            
            clean_context_data = await run_sync(_heavy_clean_task)

            # 3. 写入存储 (clean_context_data 已经是纯净的 dict)
            await self.storage.set( # type: ignore
                f"workflow:{session_id}",
                {
                    "step": steps,
                    "last_node": current_node,
                    "context": clean_context_data,
                    "updated_at": time.time(),
                },
            )
        except Exception as e:
            logger.warning("Failed to persist workflow state", session_id=session_id, error=str(e))

    # ========================= 可视化 =========================

    def to_mermaid(self) -> str:
        """生成 Mermaid 流程图代码"""
        lines = ["graph TD"]
        for node in self._nodes:
            # 入口节点使用双圆圈
            shape_start = "((" if node == self._entry_point else "("
            shape_end = "))" if node == self._entry_point else ")"
            lines.append(f"    {node}{shape_start}{node}{shape_end}")
            
        for source, targets in self._edges.items():
            for target, condition in targets:
                label = "|condition|" if condition else ""
                lines.append(f"    {source} --{label}--> {target}")
        return "\n".join(lines)

    def print_structure(self):
        """打印工作流结构"""
        print(f"\n=== Workflow: {self.name} ===")
        print(f"Entry Point: {self._entry_point}")
        print(f"\nNodes ({len(self._nodes)}):")
        for node in self._nodes:
            print(f"  - {node}")

        print(f"\nEdges ({sum(len(v) for v in self._edges.values())}):")
        for source, targets in self._edges.items():
            for target, condition in targets:
                cond_str = " [conditional]" if condition else ""
                print(f"  - {source} -> {target}{cond_str}")
        print()