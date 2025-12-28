# gecko/compose/workflow/engine.py
"""
Workflow 引擎主入口 (v0.5 Stable / 2025-12-28 Optimized)

核心职责：
1. **并行执行调度 (Phase 2 Core)**: 将 DAG 解析为执行层级 (Layers)，利用 TaskGroup 并行执行。
2. **状态管理 (State Management)**:
   - 隔离: 使用 Copy-On-Write (COW) 机制防止并行节点污染主上下文。
   - 合并: 基于 Diff 的状态合并策略 (Last Write Wins)，支持增量更新与删除。
3. **动态流控制 (Dynamic Flow)**: 支持 Next 指令动态跳转，可中断静态 DAG 计划。
4. **断点恢复 (Resume Support)**: 支持从存储加载状态并从指定节点继续执行。
5. **持久化 (Persistence)**: 细粒度的 Step 级状态保存 (Pre-Commit / Post-Commit)。

优化日志：
- [Refactor] 将 _COWDict 移至 state.py，实现关注点分离
- [Fix P0-1] 修复状态删除操作在合并时失效的问题 (Tombstone Support)
- [Fix P0-6] 修复 Team 嵌套结构导致 Next 控制流指令被忽略的问题
- [Fix P1-5] 移除运行时历史强制清理，防止长距离依赖断裂
"""
from __future__ import annotations

import inspect
from collections import deque
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Set, Union, List

import anyio

from gecko.config import get_settings
from gecko.core.events import EventBus
from gecko.core.logging import get_logger
from gecko.core.exceptions import WorkflowError
from gecko.plugins.storage.interfaces import SessionInterface

# 导入子模块
from gecko.compose.workflow.models import WorkflowContext, CheckpointStrategy, NodeStatus
from gecko.compose.workflow.graph import WorkflowGraph
from gecko.compose.workflow.executor import NodeExecutor
from gecko.compose.workflow.persistence import PersistenceManager
from gecko.compose.workflow.state import COWDict  # [Refactor] 引入独立的状态管理
from gecko.compose.nodes import Next

logger = get_logger(__name__)


class Workflow:
    """
    DAG 工作流引擎
    
    特性:
    - 支持并行节点执行
    - 支持条件分支
    - 支持循环与动态跳转
    - 支持持久化与断点恢复
    """

    def __init__(
        self,
        name: str = "Workflow",
        event_bus: Optional[EventBus] = None,
        storage: Optional[SessionInterface] = None,
        max_steps: int = 100,
        checkpoint_strategy: Optional[str] = None, 
        max_history_retention: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = 3,
        allow_cycles: bool = False,
        enable_parallel: bool = False,
    ):
        self.name = name
        self.event_bus = event_bus or EventBus()
        self.max_steps = max_steps

        # 动态获取最新配置
        current_settings = get_settings()

        strategy_val = checkpoint_strategy or current_settings.workflow_checkpoint_strategy
        retention_val = (
            max_history_retention 
            if max_history_retention is not None 
            else current_settings.workflow_history_retention
        )
        
        self._allow_cycles = allow_cycles
        self._enable_parallel = enable_parallel
        
        self._validation_errors = []

        # === 子组件组装 ===
        self.graph = WorkflowGraph()
        
        self.executor = NodeExecutor(
            enable_retry=enable_retry,
            max_retries=max_retries
        )
        
        self.persistence = PersistenceManager(
            storage=storage,
            strategy=CheckpointStrategy(strategy_val),
            history_retention=retention_val
        )

    # ================= 属性代理 (保持兼容性) =================
    
    @property
    def storage(self) -> Optional[SessionInterface]:
        return self.persistence.storage

    @storage.setter
    def storage(self, value: Optional[SessionInterface]):
        self.persistence.storage = value

    @property
    def allow_cycles(self) -> bool:
        return self._allow_cycles

    @allow_cycles.setter
    def allow_cycles(self, value: bool):
        self._allow_cycles = value
        # 修改配置后重置图验证状态
        self.graph._validated = False

    @property
    def checkpoint_strategy(self) -> CheckpointStrategy:
        return self.persistence.strategy

    @checkpoint_strategy.setter
    def checkpoint_strategy(self, value: Union[str, CheckpointStrategy]):
        self.persistence.strategy = CheckpointStrategy(value)

    # ================= 构建 API (Proxy to Graph) =================
    
    def add_node(self, name: str, func: Callable) -> "Workflow":
        """添加节点"""
        self.graph.add_node(name, func)
        return self

    def add_edge(self, source: str, target: str, condition: Optional[Callable] = None) -> "Workflow":
        """添加边 (支持条件)"""
        self.graph.add_edge(source, target, condition)
        return self

    def set_entry_point(self, name: str) -> "Workflow":
        """设置入口节点"""
        self.graph.set_entry_point(name)
        return self

    def to_mermaid(self) -> str:
        """生成 Mermaid 图表"""
        return self.graph.to_mermaid()
        
    def print_structure(self):
        print(self.to_mermaid())

    def validate(self) -> bool:
        """验证工作流结构"""
        valid, errors = self.graph.validate(
            allow_cycles=self.allow_cycles,
            enable_parallel=self._enable_parallel
        )
        self._validation_errors = errors
        return valid

    def add_parallel_group(self, *node_names: str) -> "Workflow":
        """
        [Deprecated] 定义并行组
        v0.4 引擎会自动通过图拓扑推导并行层级，此方法保留仅作兼容。
        """
        for name in node_names:
            if name not in self.graph.nodes:
                raise ValueError(f"Node '{name}' not found")
        return self

    def set_dependency(self, node: str, depends_on: Union[str, List[str]]) -> "Workflow":
        """显式设置节点依赖 (辅助构建图)"""
        deps = [depends_on] if isinstance(depends_on, str) else depends_on
        self.graph.node_dependencies.setdefault(node, set()).update(deps)
        self.graph._validated = False
        return self
        
    # 暴露内部方法供测试调用
    def _build_execution_layers(self, start_node: str):
        return self.graph.build_execution_layers(start_node)
    
    def _normalize_result(self, result: Any) -> Any:
        return self.executor._normalize_result(result)

    # ================= 执行入口 (Execution Core) =================

    async def execute_parallel(self, input_data: Any, session_id: Optional[str] = None) -> Any:
        """
        并行执行入口 (Phase 2 新增别名)
        """
        return await self.execute(input_data, session_id)

    async def execute(
        self, 
        input_data: Any, 
        session_id: Optional[str] = None, 
        start_node: Optional[str] = None,
        _resume_context: Optional[WorkflowContext] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        执行工作流 (主入口)
        
        Args:
            input_data: 初始输入数据
            session_id: 会话 ID (可选，用于持久化)
            start_node: 指定起始节点 (可选，用于 Resume 或特定入口)
            _resume_context: 注入已恢复的上下文 (用于 Resume)
            timeout: 超时时间 (秒)，None 表示无限制
        
        执行流程:
        1. 验证图结构 (强制启用并行支持)。
        2. 初始化或恢复上下文。
        3. 拓扑排序：构建静态分层执行计划 (Execution Plan)。
        4. 执行循环：逐层并行调度节点。
        5. 状态管理：处理状态合并、持久化与动态跳转。
        """
        # 1. 强制启用并行验证，允许分支结构
        self._enable_parallel = True 
        if not self.validate():
            raise WorkflowError(f"Workflow validation failed: {self._validation_errors}")

        # 2. 上下文准备
        if _resume_context:
            # 恢复模式：使用传入的上下文
            context = _resume_context
            # 注意：恢复时通常保留原有 input，除非 input_data 显式提供新值且逻辑允许覆盖
        else:
            # 全新模式：创建新上下文
            context = WorkflowContext(input=input_data)
            
        if session_id:
            context.metadata["session_id"] = session_id

        # 3. 规划执行路径
        # 优先使用传入的 start_node (Resume 场景)，否则使用图的 Entry Point
        entry = start_node or self.graph.entry_point
        if not entry:
            raise WorkflowError("No entry point defined")

        # 使用 Kahn 算法构建并行层级 (Static Plan)
        layers = self.graph.build_execution_layers(entry)
        
        # 将层级放入双端队列，作为初始执行计划
        execution_queue = deque(layers)
        
        # 记录计划到元数据 (用于调试/UI展示)
        context.metadata["execution_plan"] = [list(l) for l in layers]

        current_step = 0
        
        # 创建超时管理器 (实时超时保护)
        timeout_cm = anyio.move_on_after(timeout) if timeout is not None else nullcontext()
        
        try:
            # 4. 主执行循环 (带实时超时保护)
            with timeout_cm:
                while execution_queue:
                    # 安全检查：防止无限循环
                    if current_step >= self.max_steps:
                        raise WorkflowError(f"Exceeded max steps: {self.max_steps}")

                    # 取出当前层 (Set[node_name])
                    layer = execution_queue.popleft()

                    # 4.1 Pre-Commit (持久化 RUNNING 状态)
                    # 记录即将执行的层，以便 Crash 后知道是在哪一步挂的
                    if session_id:
                        await self.persistence.save_checkpoint(
                            session_id, current_step, list(layer), context, force=True # type: ignore
                        )

                    # 4.2 [核心] 并行执行当前层
                    # 返回: {node_name: {"output": ..., "updates": ..., "deletions": ..., "status": ...}}
                    layer_results = await self._execute_layer_parallel(layer, context)
                    
                    # 4.3 合并结果与状态 (Merge)
                    # 将并行节点的输出和状态变更合并回主 Context
                    # [Fix P0-1] 处理状态删除
                    self._merge_layer_results(context, layer_results)
                    
                    # [Fix P1-5] 移除运行时 _cleanup_history
                    # 原有的清理策略会导致长流程中依赖断裂 (KeyError)。
                    # 现在仅在持久化层 (PersistenceManager) 进行历史裁剪，运行时保留完整历史。
                    
                    # 4.4 处理动态跳转 (Next)
                    # 策略: 如果层中任何节点返回了 Next，则中断静态计划，优先处理跳转
                    # [Fix P0-6] 支持解包 Team 内部的 Next 指令
                    jump_instruction = self._handle_dynamic_jump(layer_results, context)
                    
                    if jump_instruction:
                        target_node = jump_instruction.node
                        logger.info(f"Dynamic jump to '{target_node}', static plan interrupted.")
                        
                        # 清空当前静态队列
                        execution_queue.clear()
                        # 将目标节点作为新的一层加入，转为动态执行模式
                        execution_queue.append({target_node})
                        # 清除上下文中的指针，防止污染
                        context.clear_next_pointer()

                    # 4.5 Post-Commit (持久化 SUCCESS 状态)
                    # 记录本层已完成
                    if session_id:
                        await self.persistence.save_checkpoint(
                            session_id, current_step, list(layer), context # type: ignore
                        )
                    
                    current_step += 1

            # 5. 超时检查
            if timeout is not None and getattr(timeout_cm, "cancel_called", False):  # type: ignore[attr-defined]
                raise WorkflowError(
                    f"Workflow execution exceeded timeout: {timeout}s after {current_step} steps"
                )
            
            # 6. 最终保存 (Strategy=FINAL)
            if self.persistence.strategy == CheckpointStrategy.FINAL:
                await self.persistence.save_checkpoint(
                    session_id, 9999, None, context, force=True # type: ignore
                )
                
            return context.get_last_output()
            
        except TimeoutError as e:
            # anyio 超时异常
            logger.exception(f"Workflow execution timeout after {current_step} steps")
            raise WorkflowError(
                f"Workflow execution timed out after {current_step} steps (timeout={timeout}s)"
            ) from e
        except Exception as e:
            logger.exception(f"Workflow execution failed at step {current_step}")
            if not isinstance(e, WorkflowError):
                raise WorkflowError(f"Workflow execution failed: {e}") from e
            raise

    async def _execute_layer_parallel(self, layer: Set[str], context: WorkflowContext) -> Dict[str, Any]:
        """
        并行执行单层节点
        
        Args:
            layer: 当前层的节点集合
            context: 主上下文
            
        Returns:
            Dict[node_name, exec_result_wrapper]
        """
        results: Dict[str, Any] = {}
        
        # 使用 anyio.create_task_group 实现结构化并发
        # 如果任一任务抛出异常，整个 Group 会被取消
        async with anyio.create_task_group() as tg:
            for node_name in layer:
                node_func = self.graph.nodes[node_name]
                
                # [核心机制] Copy-On-Write (COW) with Copy-on-Read Support
                # [Fix P0-5] 使用独立的 COWDict 类，支持“读取即深拷贝”策略，
                # 防止并行节点直接修改共享的可变对象（如 list/dict），造成数据污染。
                node_context = context.model_copy(deep=False)
                # [Fix] 2. 强制赋值绕过 Pydantic 验证
                # Pydantic 默认会将 COWDict 降级为 dict，导致 get_diff 方法丢失
                # 使用 object.__setattr__ 直接操作实例字典，保留 COWDict 类型
                object.__setattr__(node_context, "state", COWDict(context.state))
                # history 共享引用（只读），避免深拷贝开销
                node_context.history = context.history
                
                tg.start_soon(
                    self._run_node_wrapper, 
                    node_name, 
                    node_func, 
                    node_context, 
                    results
                )
        
        return results

    async def _run_node_wrapper(
        self, 
        name: str, 
        func: Callable, 
        ctx: WorkflowContext, 
        results: Dict[str, Any]
    ):
        """
        节点执行包装器 (Worker)
        
        职责:
        1. 运行时条件检查 (Condition Check)
        2. 调用 Executor 执行节点
        3. 捕获结果与状态变更 (Diff Calculation with Deletions)
        """
        # 1. 运行时条件检查
        # 逻辑: 只要有一条指向本节点的边满足条件 (OR 逻辑)，则执行。
        incoming_edges = []
        for src, edges in self.graph.edges.items():
            for target, cond in edges:
                if target == name:
                    incoming_edges.append((src, cond))
        
        if incoming_edges:
            should_run = False
            for src, cond in incoming_edges:
                if src == self.graph.entry_point or src in ctx.history:
                    if cond is None:
                        should_run = True
                        break
                    try:
                        res = cond(ctx)
                        if inspect.isawaitable(res):
                            res = await res
                        if res:
                            should_run = True
                            break
                    except Exception as e:
                        logger.error(f"Condition check failed for {src}->{name}: {e}")
            
            if not should_run:
                logger.info(f"Node {name} skipped due to conditions")
                results[name] = {
                    "output": None,
                    "updates": {},
                    "deletions": set(),
                    "status": NodeStatus.SKIPPED
                }
                return

        # 2. 执行节点
        try:
            output = await self.executor.execute_node(name, func, ctx)
            
            # 3. 计算 State Diff
            # [Fix P0-1] 获取更新列表 (updates) 和删除列表 (deletions)
            # 确保并行节点发出的删除指令能被主流程捕获
            updates, deletions = ctx.state.get_diff() # type: ignore
            
            results[name] = {
                "output": output,
                "updates": updates,
                "deletions": deletions,
                "status": NodeStatus.SUCCESS
            }
        except Exception as e:
            logger.exception(
                f"Node worker failed: {name}",
                node=name,
                session=getattr(ctx, 'metadata', {}).get('session_id')
            )
            # 标记失败，供上层处理或终止
            results[name] = {
                "status": NodeStatus.FAILED,
                "error": str(e)
            }
            raise WorkflowError(f"Node '{name}' execution error: {e}") from e

    def _merge_layer_results(self, context: WorkflowContext, results: Dict[str, Any]):
        """
        合并并行结果回主上下文 (Synchronization Point)
        
        [Fix P0-1] 支持删除操作：优先执行 deletes，再执行 updates
        [Fix P0-3] Next.input=None 时，保留上一步输出
        [Fix P0-4] 处理 SKIPPED 节点
        """
        layer_outputs = {}
        
        for node_name, res in results.items():
            if res.get("status") != NodeStatus.SUCCESS:
                continue
                
            output = res["output"]
            updates = res.get("updates", {})
            deletions = res.get("deletions", set())
            
            # [Fix P0-3] 处理 Next 对象：仅当 input 被显式提供时才覆盖
            actual_data = output
            if isinstance(output, Next):
                if output.input is not None:
                    actual_data = output.input
                else:
                    # 保留原有输出，不用 None 覆盖
                    actual_data = context.get_last_output()
            
            elif isinstance(output, dict) and output.get("node") and "<Next" in str(output):
                 actual_data = output.get("input")

            # 更新 History
            context.history[node_name] = actual_data
            layer_outputs[node_name] = actual_data
            
            # [Fix P0-1] 合并 State (Last Write Wins)
            # 1. 先执行删除
            for k in deletions:
                context.state.pop(k, None)
            
            # 2. 再执行更新
            context.state.update(updates)
        
        # 更新 Last Output (供下一层使用)
        if not layer_outputs:
            return

        if len(layer_outputs) == 1:
            # 单节点：直接作为值
            context.history["last_output"] = list(layer_outputs.values())[0]
        else:
            # 多节点：聚合为字典 {node_name: output}
            context.history["last_output"] = layer_outputs

    def _handle_dynamic_jump(self, results: Dict[str, Any], context: WorkflowContext) -> Optional[Next]:
        """
        检查并处理动态跳转 (Next)
        
        [Fix P0-6] 支持递归解包，从 Team (List[MemberResult]) 中提取隐藏的 Next 指令
        
        Args:
            results: 当前层的执行结果
            
        Returns:
            找到的第一个 Next 指令 (如果有)
        """
        for res in results.values():
            val = res.get("output")
            
            # Case 1: 直接返回 Next (普通 Function 节点)
            if isinstance(val, Next):
                self._apply_next_instruction(val, context)
                return val
            
            # Case 2: Team 结果 (List[MemberResult])
            # 需要识别 Team 内部某个 Member 是否发出了 Next 指令
            if isinstance(val, list):
                for item in val:
                    # Duck typing: 检查是否有 result 属性且为 Next
                    if hasattr(item, "result") and isinstance(item.result, Next):
                        # 发现跳转指令，应用并返回
                        # 注意：如果多个成员都返回 Next，这里默认采用第一个发现的 (Race 模式下通常只有一个 Winner)
                        logger.info(f"Unwrapped Next instruction from Team member")
                        self._apply_next_instruction(item.result, context)
                        return item.result
                        
        return None

    def _apply_next_instruction(self, next_obj: Next, context: WorkflowContext):
        """应用 Next 指令的副作用 (State Update / Next Input)"""
        if next_obj.input is not None:
            context.state["_next_input"] = next_obj.input
        
        if next_obj.update_state:
            context.state.update(next_obj.update_state)

    async def resume(self, session_id: str) -> Any:
        """
        从存储恢复执行
        
        逻辑：
        1. 加载 Checkpoint 数据。
        2. 重建 WorkflowContext。
        3. 确定断点位置 (Next Pointer 优先，其次是 Last Node 的下游)。
        4. 调用 execute 从断点继续运行。
        """
        data = await self.persistence.load_checkpoint(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
            
        try:
            context = WorkflowContext.from_storage_payload(data["context"])
        except Exception as e:
            raise WorkflowError(f"Failed to reconstruct context: {e}") from e

        last_node = data.get("last_node")
        
        # 1. 动态跳转优先 (Next Pointer)
        if context.next_pointer:
            next_node = context.next_pointer.get("target_node")
            # 恢复 Next 携带的输入
            if context.next_pointer.get("input") is not None:
                context.state["_next_input"] = context.next_pointer["input"]
            
            logger.info(f"Resuming workflow from dynamic jump: {next_node}")
            return await self.execute(
                input_data=context.input,
                session_id=session_id,
                start_node=next_node,
                _resume_context=context # 注入恢复后的上下文
            )

        # 2. 静态流程恢复 (Last Node Successor)
        next_node = None
        if last_node:
             edges = self.graph.edges.get(last_node, [])
             if edges:
                 # 简单取第一条出边 (复杂分叉恢复需更完整状态记录)
                 next_node = edges[0][0] 
        
        if next_node:
            logger.info(f"Resuming workflow from static flow: {next_node}")
            return await self.execute(
                input_data=context.input,
                session_id=session_id,
                start_node=next_node,
                _resume_context=context
            )
            
        # 3. 无法确定下一跳，返回历史结果
        logger.warning(f"Could not determine next step from checkpoint {session_id}. Returning last output.")
        return context.get_last_output()