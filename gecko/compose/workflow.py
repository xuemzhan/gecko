# gecko/compose/workflow.py
"""
Workflow 引擎（Phase 2 优化版）

改进：
1. 完整的 DAG 验证（环检测、孤立节点）
2. 执行追踪与可视化
3. 条件路由优化
4. 错误恢复机制
"""
from __future__ import annotations
import asyncio
import inspect
from typing import Any, Dict, Callable, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from gecko.core.events import EventBus, BaseEvent
from gecko.core.message import Message
from gecko.core.agent import Agent
from gecko.compose.nodes import Next, ensure_awaitable
from gecko.core.logging import get_logger
from gecko.core.exceptions import WorkflowError, WorkflowCycleError
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)

# ========== 事件定义 ==========

class WorkflowEvent(BaseEvent):
    """Workflow 专用事件"""
    pass

# ========== 节点状态 ==========

class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class NodeExecution:
    """节点执行记录"""
    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """执行耗时"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0

# ========== 工作流上下文 ==========

@dataclass
class WorkflowContext:
    """工作流执行上下文"""
    input: Any
    state: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ✅ 新增：执行追踪
    executions: List[NodeExecution] = field(default_factory=list)
    
    def add_execution(self, execution: NodeExecution):
        """记录节点执行"""
        self.executions.append(execution)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        total_time = sum(e.duration for e in self.executions)
        status_counts = {}
        for status in NodeStatus:
            status_counts[status.value] = sum(
                1 for e in self.executions if e.status == status
            )
        
        return {
            "total_nodes": len(self.executions),
            "total_time": total_time,
            "status_counts": status_counts,
            "node_details": [
                {
                    "name": e.node_name,
                    "status": e.status.value,
                    "duration": e.duration,
                    "error": e.error
                }
                for e in self.executions
            ]
        }
        
    # [新增] 序列化支持
    def to_dict(self) -> Dict[str, Any]:
        """简单的序列化，实际生产中可能需要更复杂的处理来排除不可序列化对象"""
        return {
            "input": str(self.input), # 简化处理
            "state": self.state,
            "history": {k: str(v) for k, v in self.history.items()}, # 简化处理
            "metadata": self.metadata
        }

# ========== Workflow 引擎 ==========

class Workflow:
    """
    工作流引擎（Phase 2 增强版）
    
    核心改进：
    1. 完整的 DAG 验证
    2. 执行追踪
    3. 错误处理
    4. 可视化支持
    """
    
    def __init__(
        self,
        name: str = "Workflow",
        event_bus: Optional[EventBus] = None,
        storage: Optional[SessionInterface] = None, # [新增] 持久化存储
        max_steps: int = 100,
        enable_retry: bool = False,  # ✅ 新增：重试开关
        max_retries: int = 3,
    ):
        self.name = name
        self.event_bus = event_bus or EventBus()
        self.storage = storage
        self.max_steps = max_steps
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        
        # 节点和边
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, List[Tuple[str, Optional[Callable]]]] = {}
        self.entry_point: Optional[str] = None
        
        # ✅ 验证状态
        self._validated = False
        self._validation_errors: List[str] = []

    # ========== 构建 API ==========

    def add_node(self, name: str, func: Callable) -> 'Workflow':
        """添加节点"""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        
        self.nodes[name] = func
        self._validated = False
        
        logger.debug("Node added", node=name)
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[WorkflowContext], bool]] = None
    ) -> 'Workflow':
        """添加边"""
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        if source not in self.edges:
            self.edges[source] = []
        
        self.edges[source].append((target, condition))
        self._validated = False
        
        logger.debug("Edge added", source=source, target=target)
        return self

    def set_entry_point(self, name: str) -> 'Workflow':
        """设置入口节点"""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")
        
        self.entry_point = name
        self._validated = False
        return self

    # ========== DAG 验证 ==========

    def validate(self) -> bool:
        """
        验证工作流的有效性
        
        检查：
        1. 是否有入口点
        2. 是否存在环
        3. 是否有孤立节点
        4. 边的条件函数是否有效
        
        返回：是否有效
        """
        if self._validated:
            return len(self._validation_errors) == 0
        
        self._validation_errors.clear()
        
        # 1. 检查入口点
        if not self.entry_point:
            self._validation_errors.append("No entry point defined")
        elif self.entry_point not in self.nodes:
            self._validation_errors.append(f"Entry point '{self.entry_point}' not in nodes")
        
        # 2. 检测环
        try:
            self._detect_cycles()
        except WorkflowCycleError as e:
            self._validation_errors.append(str(e))
        
        # 3. 检查可达性
        unreachable = self._find_unreachable_nodes()
        if unreachable:
            logger.warning(
                "Workflow has unreachable nodes",
                nodes=list(unreachable)
            )
            # 这只是警告，不算验证失败
        
        # 4. 检查死节点（没有出边且不是终止节点）
        dead_nodes = self._find_dead_nodes()
        if dead_nodes:
            logger.warning(
                "Workflow has dead-end nodes",
                nodes=list(dead_nodes)
            )
        
        self._validated = True
        
        if self._validation_errors:
            logger.error(
                "Workflow validation failed",
                errors=self._validation_errors
            )
            return False
        
        logger.info("Workflow validation passed", name=self.name)
        return True

    def _detect_cycles(self):
        """
        检测环（使用 DFS）
        
        如果发现环，抛出 WorkflowCycleError
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            """DFS 遍历，返回 True 表示发现环"""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # 遍历所有邻居
            for neighbor, _ in self.edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # 发现环
                    cycle_start = path.index(neighbor)
                    cycle = " → ".join(path[cycle_start:] + [neighbor])
                    raise WorkflowCycleError(f"Cycle detected: {cycle}")
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        # 从所有节点开始 DFS（处理不连通图）
        for node in self.nodes:
            if node not in visited:
                dfs(node, [])

    def _find_unreachable_nodes(self) -> Set[str]:
        """查找从入口点无法到达的节点"""
        if not self.entry_point:
            return set(self.nodes.keys())
        
        reachable: Set[str] = set()
        queue = [self.entry_point]
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            
            reachable.add(current)
            
            # 添加所有邻居
            for neighbor, _ in self.edges.get(current, []):
                if neighbor not in reachable:
                    queue.append(neighbor)
        
        return set(self.nodes.keys()) - reachable

    def _find_dead_nodes(self) -> Set[str]:
        """查找死节点（没有出边）"""
        dead = set()
        for node in self.nodes:
            if node not in self.edges or len(self.edges[node]) == 0:
                dead.add(node)
        return dead

    def get_validation_errors(self) -> List[str]:
        """获取验证错误列表"""
        return self._validation_errors.copy()

    # ========== 执行引擎 ==========

    async def execute(self, input_data: Any, session_id: Optional[str] = None) -> Any:
        """
        执行工作流
        
        流程：
        1. 验证
        2. 初始化上下文
        3. 执行节点
        4. 返回结果
        """
        # 1. 验证
        if not self.validate():
            errors = "\n".join(self._validation_errors)
            raise WorkflowError(f"Workflow validation failed:\n{errors}")
        
        # 2. 初始化上下文
        # 尝试从存储加载上下文 (这里简化为重新开始，但在真实场景可实现断点续传)
        context = WorkflowContext(input=input_data)
        if session_id:
            context.metadata["session_id"] = session_id
        
        # 3. 发布开始事件
        await self.event_bus.publish(
            WorkflowEvent(
                type="workflow_started",
                data={"name": self.name, "input": str(input_data)[:100]}
            )
        )
        
        try:
            # 4. 执行
            result = await self._execute_loop(context, session_id)
            
            # 5. 发布完成事件
            await self.event_bus.publish(
                WorkflowEvent(
                    type="workflow_completed",
                    data={
                        "name": self.name,
                        "summary": context.get_execution_summary()
                    }
                )
            )
            
            return result
        
        except Exception as e:
            # 发布错误事件
            await self.event_bus.publish(
                WorkflowEvent(
                    type="workflow_error",
                    error=str(e),
                    data={"name": self.name}
                )
            )
            raise

    async def _execute_loop(self, context: WorkflowContext, session_id: Optional[str]) -> Any:
        """执行循环"""
        current_node = self.entry_point
        steps = 0
        
        while current_node and steps < self.max_steps:
            steps += 1
            
            logger.debug(
                "Executing node",
                node=current_node,
                step=steps
            )
            
            # 执行节点
            result = await self._execute_node(current_node, context)
            
            # 保存结果
            context.history[current_node] = result
            context.history["last_output"] = result
            
            # [新增] 状态持久化
            if self.storage and session_id:
                try:
                    # 保存 Workflow 状态快照
                    # 注意：实际生产中需要确保 context 可序列化，或者只保存关键数据
                    await self.storage.set(
                        f"workflow:{session_id}", 
                        {"step": steps, "last_node": current_node, "context": context.to_dict()}
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist workflow state: {e}")
            
            # 路由决策
            if isinstance(result, Next):
                # 显式跳转
                current_node = result.node
                if result.input is not None:
                    context.history["last_output"] = result.input
            else:
                # 查找下一个节点
                # [修改] 使用 await 调用新的异步查找方法
                current_node = await self._find_next_node(current_node, context)
        
        if steps >= self.max_steps:
            raise WorkflowError(
                f"Workflow exceeded max steps: {self.max_steps}",
                context={"steps": steps}
            )
        
        return context.history.get("last_output")

    async def _execute_node(
        self,
        node_name: str,
        context: WorkflowContext
    ) -> Any:
        """
        执行单个节点
        
        支持：
        1. 普通函数
        2. Agent 实例
        3. Team 实例
        4. 重试机制（如果启用）
        """
        import time
        
        node_func = self.nodes[node_name]
        
        # 创建执行记录
        execution = NodeExecution(node_name=node_name)
        execution.start_time = time.time()
        execution.status = NodeStatus.RUNNING
        
        # 发布节点开始事件
        await self.event_bus.publish(
            WorkflowEvent(type="node_started", data={"node": node_name})
        )
        
        try:
            # 执行节点（带重试）
            if self.enable_retry:
                result = await self._execute_with_retry(node_func, context)
            else:
                result = await self._execute_once(node_func, context)
            
            # 更新执行记录
            execution.status = NodeStatus.SUCCESS
            execution.output_data = result
            execution.end_time = time.time()
            
            # 发布节点完成事件
            await self.event_bus.publish(
                WorkflowEvent(
                    type="node_completed",
                    data={
                        "node": node_name,
                        "duration": execution.duration,
                        "result": str(result)[:100]
                    }
                )
            )
            
            return result
        
        except Exception as e:
            # 更新执行记录
            execution.status = NodeStatus.FAILED
            execution.error = str(e)
            execution.end_time = time.time()
            
            # 发布节点错误事件
            await self.event_bus.publish(
                WorkflowEvent(
                    type="node_error",
                    error=str(e),
                    data={"node": node_name}
                )
            )
            
            raise WorkflowError(
                f"Node '{node_name}' execution failed: {e}",
                context={"node": node_name}
            ) from e
        
        finally:
            # 记录执行
            context.add_execution(execution)

    async def _execute_once(self, func: Callable, context: WorkflowContext) -> Any:
        """执行节点一次"""
        # 处理 Agent 实例
        if isinstance(func, Agent):
            user_input = context.history.get("last_output", context.input)
            if isinstance(user_input, str):
                user_input = Message.user(user_input)
            elif not isinstance(user_input, list):
                user_input = [Message.user(str(user_input))]
            
            output = await func.run(user_input)
            return output.content
        
        # 处理普通函数
        return await ensure_awaitable(func, context)

    async def _execute_with_retry(
        self,
        func: Callable,
        context: WorkflowContext
    ) -> Any:
        """带重试的执行"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._execute_once(func, context)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Node execution failed, retrying",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )
                
                if attempt < self.max_retries - 1:
                    # 等待后重试
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        # 所有重试都失败
        raise last_error

    async def _find_next_node(
        self,
        current: str,
        context: WorkflowContext
    ) -> Optional[str]:
        """
        查找下一个节点
        
        支持条件路由
        """
        edges = self.edges.get(current, [])
        
        for target, condition in edges:
            if condition is None:
                # 无条件边
                return target
            
            # 评估条件
            try:
                result = False
                if inspect.iscoroutinefunction(condition):
                    # [修改] 异步条件，必须 await
                    result = await condition(context)
                else:
                    # 同步条件
                    result = condition(context)
                
                if result:
                    return target

            except Exception as e:
                logger.error(
                    "Condition evaluation failed",
                    source=current,
                    target=target,
                    error=str(e)
                )
        
        return None  # 没有匹配的边，结束

    # ========== 可视化支持 ==========

    def to_mermaid(self) -> str:
        """
        生成 Mermaid 流程图
        
        可用于文档或调试
        """
        lines = ["graph TD"]
        
        # 添加节点
        for node in self.nodes:
            label = f"[{node}]" if node == self.entry_point else f"({node})"
            lines.append(f"    {node}{label}")
        
        # 添加边
        for source, targets in self.edges.items():
            for target, condition in targets:
                label = f"|condition|" if condition else ""
                lines.append(f"    {source} --{label}--> {target}")
        
        return "\n".join(lines)

    def print_structure(self):
        """打印工作流结构（调试用）"""
        print(f"\n=== Workflow: {self.name} ===")
        print(f"Entry Point: {self.entry_point}")
        print(f"\nNodes ({len(self.nodes)}):")
        for node in self.nodes:
            print(f"  - {node}")
        
        print(f"\nEdges ({sum(len(v) for v in self.edges.values())}):")
        for source, targets in self.edges.items():
            for target, condition in targets:
                cond_str = " [conditional]" if condition else ""
                print(f"  - {source} → {target}{cond_str}")
        
        print()