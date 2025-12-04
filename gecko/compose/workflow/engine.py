# gecko/compose/workflow/engine.py
"""
Workflow 引擎主入口 (v0.4 Stable / 2025-12-03 Optimized)

核心职责：
1. **并行执行调度 (Phase 2 Core)**: 将 DAG 解析为执行层级 (Layers)，利用 TaskGroup 并行执行。
2. **状态管理 (State Management)**:
   - 隔离: 使用 Copy-On-Write (COW) 机制防止并行节点污染主上下文。
   - 合并: 基于 Diff 的状态合并策略 (Last Write Wins)。
3. **动态流控制 (Dynamic Flow)**: 支持 Next 指令动态跳转，可中断静态 DAG 计划。
4. **断点恢复 (Resume Support)**: 支持从存储加载状态并从指定节点继续执行。
5. **持久化 (Persistence)**: 细粒度的 Step 级状态保存 (Pre-Commit / Post-Commit)。
"""
from __future__ import annotations

import asyncio
import inspect
from collections import deque
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
        _resume_context: Optional[WorkflowContext] = None
    ) -> Any:
        """
        执行工作流 (主入口)
        
        Args:
            input_data: 初始输入数据
            session_id: 会话 ID (可选，用于持久化)
            start_node: [v0.4] 指定起始节点 (可选，用于 Resume 或特定入口)
            _resume_context: [v0.4] 注入已恢复的上下文 (用于 Resume)
        
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
        
        try:
            # 4. 主执行循环
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
                # 返回: {node_name: {"output": ..., "state_diff": ...}}
                layer_results = await self._execute_layer_parallel(layer, context)
                
                # 4.3 合并结果与状态 (Merge)
                # 将并行节点的输出和状态变更合并回主 Context
                self._merge_layer_results(context, layer_results)
                
                # [P1-2 修复] 定期清理 history，防止无界增长
                self._cleanup_history(context, max_steps=self.persistence.history_retention)
                
                # 4.4 处理动态跳转 (Next)
                # 策略: 如果层中任何节点返回了 Next，则中断静态计划，优先处理跳转
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

            # 5. 最终保存 (Strategy=FINAL)
            if self.persistence.strategy == CheckpointStrategy.FINAL:
                await self.persistence.save_checkpoint(
                    session_id, 9999, None, context, force=True # type: ignore
                )
                
            return context.get_last_output()
            
        except Exception as e:
            logger.exception("Workflow execution failed")
            if not isinstance(e, WorkflowError):
                raise WorkflowError(f"Workflow execution failed: {e}") from e
            raise

    def _cleanup_history(self, context: WorkflowContext, max_steps: int = 20):
        """
        [P1-2 修复] 定期清理 history，防止无界增长
        
        策略:
        1. 保留最后 N 步的历史记录
        2. 必须保留 last_output，它是下一步的默认输入
        3. 删除最老的记录以限制内存使用
        """
        if len(context.history) <= max_steps:
            return
        
        # 必须保留的关键字段
        must_keep = {"last_output"}
        
        # 获取所有可删除的键
        all_keys = set(context.history.keys()) - must_keep
        old_keys = sorted(all_keys)[:-max_steps]
        
        # 删除最老的键
        for key in old_keys:
            logger.debug(f"Cleaning up history key: {key}")
            del context.history[key]
        
        if old_keys:
            logger.info(
                "History cleanup",
                before=len(context.history) + len(old_keys),
                after=len(context.history),
                removed=len(old_keys)
            )

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
                
                # [核心机制] Copy-On-Write (COW)
                # 为每个节点创建 Context 的浅拷贝 (避免对大 history 做深拷贝)，
                # 并对 `state` 做一次浅复制以实现写时复制（Copy-On-Write）。
                # history 保持共享引用以节省内存和拷贝成本（只读语义）。
                node_context = context.model_copy(deep=False)
                # 将 state 设置为主 state 的浅拷贝，写操作不会影响主 context
                node_context.state = dict(context.state)
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
        3. 捕获结果与状态变更 (Diff Calculation)
        """
        # 1. 运行时条件检查
        # 逻辑: 只要有一条指向本节点的边满足条件 (OR 逻辑)，则执行。
        # 如果所有入边的条件都不满足，则跳过。
        incoming_edges = []
        for src, edges in self.graph.edges.items():
            for target, cond in edges:
                if target == name:
                    incoming_edges.append((src, cond))
        
        if incoming_edges:
            should_run = False
            for src, cond in incoming_edges:
                # 只有当上游已执行 (在 history 中) 或者是 Start 节点时，条件才有意义
                # 如果上游节点未执行（被跳过），则该路径视为不通
                if src == self.graph.entry_point or src in ctx.history:
                    # 无条件边 -> 视为 True
                    if cond is None:
                        should_run = True
                        break
                    try:
                        # 评估条件 (支持 sync/async)
                        res = cond(ctx)
                        if inspect.isawaitable(res):
                            res = await res
                        if res:
                            should_run = True
                            break
                    except Exception as e:
                        # 条件执行报错视为不通过 (Fail Safe)
                        logger.error(f"Condition check failed for {src}->{name}: {e}")
            
            if not should_run:
                logger.info(f"Node {name} skipped due to conditions")
                # [P0-4 修复] 返回 SKIPPED 状态，而不是 None
                # 这样 _merge_layer_results 可以正确处理被跳过的节点
                results[name] = {
                    "output": None,
                    "state_diff": {},
                    "status": NodeStatus.SKIPPED  # [新增] 标记为跳过
                }
                return

        # 2. 执行节点
        try:
            output = await self.executor.execute_node(name, func, ctx)
            
            # 3. 计算 State Diff
            # 简单策略：直接返回当前节点的所有 state，由 merge 负责更新 (Last Write Wins)
            # 进阶策略：可以对比 ctx.state 和初始快照，只返回变更部分
            state_diff = ctx.state 
            
            # 写入结果容器 (线程安全，因为是在协程回调中写入 dict)
            results[name] = {
                "output": output,
                "state_diff": state_diff
            }
        except Exception as e:
            # 异常冒泡，TaskGroup 会捕获并取消其他任务
            raise e

    def _merge_layer_results(self, context: WorkflowContext, results: Dict[str, Any]):
        """
        合并并行结果回主上下文 (Synchronization Point)
        
        [P0-3 修复] Next.input=None 时，保留上一步输出而不是用 None 覆盖
        [P0-4 修复] 处理 SKIPPED 节点，不更新历史
        """
        layer_outputs = {}
        
        for node_name, res in results.items():
            # [P0-4 修复] 跳过的节点不更新 history
            if res.get("status") == NodeStatus.SKIPPED:
                logger.debug(f"Node {node_name} was skipped, not updating history")
                continue
                
            output = res["output"]
            state_diff = res["state_diff"]
            
            # [P0-3 修复] 处理 Next 对象：仅当 input 被显式提供时才覆盖
            actual_data = output
            if isinstance(output, Next):
                # 重点：如果 input 为 None，保留上一步的输出
                if output.input is not None:
                    actual_data = output.input
                else:
                    # 保留原有输出，不用 None 覆盖
                    actual_data = context.get_last_output()
            
            # 兼容性处理：如果 Executor 返回的是 dict 形式的 Next (被意外 normalize 了)
            elif isinstance(output, dict) and output.get("node") and "<Next" in str(output):
                 actual_data = output.get("input")

            # 更新 History
            context.history[node_name] = actual_data
            layer_outputs[node_name] = actual_data
            
            # 合并 State (Last Write Wins)
            # 并行节点若修改同一 Key，后处理者覆盖前者
            if state_diff:
                context.state.update(state_diff)
        
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
        
        Args:
            results: 当前层的执行结果
            
        Returns:
            找到的第一个 Next 指令 (如果有)
        """
        for res in results.values():
            val = res["output"]
            
            if isinstance(val, Next):
                # 将 Next.input 注入 state 的 _next_input，供下一个节点作为 input 使用
                if val.input is not None:
                    context.state["_next_input"] = val.input
                
                # 处理附带的 state 更新
                if val.update_state:
                    context.state.update(val.update_state)
                    
                return val
        return None

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