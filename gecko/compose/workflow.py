# gecko/compose/workflow.py  
"""  
Workflow 引擎（优化版）  
  
核心改进：  
1. 节点执行结果统一标准化，避免下游/持久化因类型不一致出错  
2. `Next` 控制流可携带自定义输入，自动注入到下一个节点  
3. 工作流上下文持久化使用安全序列化方法（pydantic_encoder）  
4. Agent / Team / 普通函数节点统一接收 WorkflowContext，行为更一致  
5. 状态持久化抽象成独立方法，捕获异常以免影响主流程  
"""  
  
from __future__ import annotations  
  
import asyncio  
import inspect  
import time  
from dataclasses import dataclass, field  
from enum import Enum  
from typing import Any, Callable, Dict, List, Optional, Set, Tuple  
  
from pydantic import BaseModel  
from pydantic.json import pydantic_encoder  
  
from gecko.compose.nodes import Next  
from gecko.core.agent import Agent  
from gecko.core.events import BaseEvent, EventBus  
from gecko.core.exceptions import WorkflowCycleError, WorkflowError  
from gecko.core.logging import get_logger  
from gecko.core.message import Message  
from gecko.core.utils import ensure_awaitable  
from gecko.plugins.storage.interfaces import SessionInterface  
  
logger = get_logger(__name__)  
  
  
# ========================= 事件定义 =========================  
class WorkflowEvent(BaseEvent):  
    """Workflow 专用事件对象，可被外部订阅"""  
    pass  
  
  
# ========================= 节点执行记录 =========================  
class NodeStatus(Enum):  
    PENDING = "pending"  
    RUNNING = "running"  
    SUCCESS = "success"  
    FAILED = "failed"  
    SKIPPED = "skipped"  
  
  
@dataclass  
class NodeExecution:  
    """单个节点的执行记录（方便追踪和可视化）"""  
    node_name: str  
    status: NodeStatus = NodeStatus.PENDING  
    input_data: Any = None  
    output_data: Any = None  
    error: Optional[str] = None  
    start_time: float = 0.0  
    end_time: float = 0.0  
  
    @property  
    def duration(self) -> float:  
        return max(0.0, self.end_time - self.start_time)  
  
  
# ========================= 工作流上下文 =========================  
@dataclass  
class WorkflowContext:  
    """  
    工作流执行过程中共享的上下文对象  
    - input: 初始输入  
    - state: 节点之间共享的状态（开发者可自由使用）  
    - history: 每个节点的输出以及 last_output  
    - metadata: 附加信息（如 session_id、external trace id 等）  
    - executions: 节点执行详情列表  
    """  
    input: Any  
    state: Dict[str, Any] = field(default_factory=dict)  
    history: Dict[str, Any] = field(default_factory=dict)  
    metadata: Dict[str, Any] = field(default_factory=dict)  
    executions: List[NodeExecution] = field(default_factory=list)  
  
    def add_execution(self, execution: NodeExecution):  
        self.executions.append(execution)  
  
    def get_execution_summary(self) -> Dict[str, Any]:  
        total_time = sum(e.duration for e in self.executions)  
        status_counts = {  
            status.value: sum(1 for e in self.executions if e.status == status)  
            for status in NodeStatus  
        }  
        return {  
            "total_nodes": len(self.executions),  
            "total_time": total_time,  
            "status_counts": status_counts,  
            "node_details": [  
                {  
                    "name": e.node_name,  
                    "status": e.status.value,  
                    "duration": e.duration,  
                    "error": e.error,  
                }  
                for e in self.executions  
            ],  
        }  
  
    def to_dict(self) -> Dict[str, Any]:  
        """  
        安全序列化上下文，以便持久化  
        - 使用 pydantic_encoder 处理 Message / BaseModel 等复杂对象  
        - history 仅保留字符串或 JSON 友好格式，防止过大或不可序列化  
        """  
        def _safe(value: Any) -> Any:  
            try:  
                return pydantic_encoder(value)  
            except Exception:  
                return str(value)[:200]  
  
        history_dump = {  
            k: _safe(v)  
            for k, v in self.history.items()  
            if k == "last_output" or isinstance(k, str)  
        }  
  
        return {  
            "input": _safe(self.input),  
            "state": {k: _safe(v) for k, v in self.state.items()},  
            "history": history_dump,  
            "metadata": {k: _safe(v) for k, v in self.metadata.items()},  
        }  
  
  
# ========================= 工作流引擎 =========================  
class Workflow:  
    def __init__(  
        self,  
        name: str = "Workflow",  
        event_bus: Optional[EventBus] = None,  
        storage: Optional[SessionInterface] = None,  
        max_steps: int = 100,  
        enable_retry: bool = False,  
        max_retries: int = 3,  
    ):  
        self.name = name  
        self.event_bus = event_bus or EventBus()  
        self.storage = storage  
        self.max_steps = max_steps  
        self.enable_retry = enable_retry  
        self.max_retries = max_retries  
  
        self.nodes: Dict[str, Callable] = {}  
        self.edges: Dict[str, List[Tuple[str, Optional[Callable]]]] = {}  
        self.entry_point: Optional[str] = None  
  
        self._validated = False  
        self._validation_errors: List[str] = []  
  
    # ---------- DAG 构建 API ----------  
    def add_node(self, name: str, func: Callable) -> "Workflow":  
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
        condition: Optional[Callable[[WorkflowContext], bool]] = None,  
    ) -> "Workflow":  
        if source not in self.nodes:  
            raise ValueError(f"Source node '{source}' not found")  
        if target not in self.nodes:  
            raise ValueError(f"Target node '{target}' not found")  
  
        self.edges.setdefault(source, []).append((target, condition))  
        self._validated = False  
        logger.debug("Edge added", source=source, target=target)  
        return self  
  
    def set_entry_point(self, name: str) -> "Workflow":  
        if name not in self.nodes:  
            raise ValueError(f"Node '{name}' not found")  
        self.entry_point = name  
        self._validated = False  
        return self  
  
    # ---------- DAG 验证 ----------  
    def validate(self) -> bool:  
        if self._validated:  
            return len(self._validation_errors) == 0  
  
        self._validation_errors.clear()  
  
        if not self.entry_point:  
            self._validation_errors.append("No entry point defined")  
        elif self.entry_point not in self.nodes:  
            self._validation_errors.append(f"Entry point '{self.entry_point}' not in nodes")  
  
        try:  
            self._detect_cycles()  
        except WorkflowCycleError as e:  
            self._validation_errors.append(str(e))  
  
        unreachable = self._find_unreachable_nodes()  
        if unreachable:  
            logger.warning("Workflow has unreachable nodes", nodes=list(unreachable))  
  
        dead_nodes = self._find_dead_nodes()  
        if dead_nodes:  
            logger.warning("Workflow has dead-end nodes", nodes=list(dead_nodes))  
  
        self._validated = True  
        if self._validation_errors:  
            logger.error("Workflow validation failed", errors=self._validation_errors)  
            return False  
  
        logger.info("Workflow validation passed", name=self.name)  
        return True  
  
    def _detect_cycles(self):  
        visited: Set[str] = set()  
        rec_stack: Set[str] = set()  
  
        def dfs(node: str, path: List[str]):  
            visited.add(node)  
            rec_stack.add(node)  
            path.append(node)  
  
            for neighbor, _ in self.edges.get(node, []):  
                if neighbor not in visited:  
                    dfs(neighbor, path)  
                elif neighbor in rec_stack:  
                    cycle_start = path.index(neighbor)  
                    cycle = " → ".join(path[cycle_start:] + [neighbor])  
                    raise WorkflowCycleError(f"Cycle detected: {cycle}")  
  
            rec_stack.remove(node)  
            path.pop()  
  
        for node in self.nodes:  
            if node not in visited:  
                dfs(node, [])  
  
    def _find_unreachable_nodes(self) -> Set[str]:  
        if not self.entry_point:  
            return set(self.nodes.keys())  
  
        reachable: Set[str] = set()  
        queue = [self.entry_point]  
  
        while queue:  
            current = queue.pop(0)  
            if current in reachable:  
                continue  
            reachable.add(current)  
            for neighbor, _ in self.edges.get(current, []):  
                if neighbor not in reachable:  
                    queue.append(neighbor)  
  
        return set(self.nodes.keys()) - reachable  
  
    def _find_dead_nodes(self) -> Set[str]:  
        """简单检测：没有出边且不是入口的节点"""  
        dead = set()  
        for node in self.nodes:  
            if not self.edges.get(node) and node != self.entry_point:  
                dead.add(node)  
        return dead  
  
    def get_validation_errors(self) -> List[str]:  
        return self._validation_errors.copy()  
  
    # ---------- 执行入口 ----------  
    async def execute(self, input_data: Any, session_id: Optional[str] = None) -> Any:  
        if not self.validate():  
            errors = "\n".join(self._validation_errors)  
            raise WorkflowError(f"Workflow validation failed:\n{errors}")  
  
        context = WorkflowContext(input=input_data)  
        if session_id:  
            context.metadata["session_id"] = session_id  
  
        await self.event_bus.publish(  
            WorkflowEvent(type="workflow_started", data={"name": self.name, "input": str(input_data)[:100]})  
        )  
  
        try:  
            result = await self._execute_loop(context, session_id)  
            await self.event_bus.publish(  
                WorkflowEvent(  
                    type="workflow_completed",  
                    data={"name": self.name, "summary": context.get_execution_summary()},  
                )  
            )  
            return result  
        except Exception as e:  
            await self.event_bus.publish(  
                WorkflowEvent(type="workflow_error", error=str(e), data={"name": self.name})  
            )  
            raise  
  
    async def _execute_loop(self, context: WorkflowContext, session_id: Optional[str]) -> Any:  
        current_node = self.entry_point  
        steps = 0  
  
        while current_node and steps < self.max_steps:  
            steps += 1  
            node_name = current_node  
  
            logger.debug("Executing node", node=node_name, step=steps)  
            result = await self._execute_node(node_name, context)  
  
            normalized = self._normalize_result(result)  
            context.history[node_name] = normalized  
            context.history["last_output"] = normalized  
  
            if isinstance(result, Next):  
                current_node = result.node  
                if result.input is not None:  
                    normalized_input = self._normalize_result(result.input)  
                    context.history["last_output"] = normalized_input  
                    context.state["_next_input"] = normalized_input  
            else:  
                current_node = await self._find_next_node(node_name, context)  
  
            if self.storage and session_id:  
                await self._persist_state(session_id, steps, node_name, context)  
  
        if steps >= self.max_steps:  
            raise WorkflowError(  
                f"Workflow exceeded max steps: {self.max_steps}",  
                context={"steps": steps, "last_node": current_node},  
            )  
  
        return context.history.get("last_output") 
    
    async def _persist_state(  
        self,  
        session_id: str,  
        steps: int,  
        last_node: Optional[str],  
        context: WorkflowContext,  
    ):  
        """状态持久化统一入口"""  
        try:  
            await self.storage.set(  
                f"workflow:{session_id}",  
                {  
                    "step": steps,  
                    "last_node": last_node,  
                    "context": context.to_dict(),  
                },  
            )  
        except Exception as e:  
            logger.warning("Failed to persist workflow state", error=str(e)) 
  
    async def _execute_node(self, node_name: str, context: WorkflowContext) -> Any:  
        execution = NodeExecution(node_name=node_name, status=NodeStatus.RUNNING, start_time=time.time())  
        await self.event_bus.publish(WorkflowEvent(type="node_started", data={"node": node_name}))  
  
        node_callable = self.nodes[node_name]  
  
        try:  
            if self.enable_retry:  
                result = await self._execute_with_retry(node_callable, context)  
            else:  
                result = await self._execute_once(node_callable, context)  
  
            execution.status = NodeStatus.SUCCESS  
            execution.output_data = self._normalize_result(result)  
            execution.end_time = time.time()  
  
            await self.event_bus.publish(  
                WorkflowEvent(  
                    type="node_completed",  
                    data={"node": node_name, "duration": execution.duration, "result": str(result)[:100]},  
                )  
            )  
            return result  
  
        except Exception as e:  
            execution.status = NodeStatus.FAILED  
            execution.error = str(e)  
            execution.end_time = time.time()  
  
            await self.event_bus.publish(  
                WorkflowEvent(type="node_error", error=str(e), data={"node": node_name})  
            )  
            raise WorkflowError(f"Node '{node_name}' execution failed: {e}", context={"node": node_name}) from e  
  
        finally:  
            context.add_execution(execution)
    
    async def _run_agent_node(self, agent: Agent, context: WorkflowContext) -> Any:  
        """  
        Agent 节点默认读取 last_output 作为输入；  
        如果上一节点通过 Next.input 传递了自定义输入，则优先使用  
        """  
        user_input = context.state.pop("_next_input", None) or context.history.get("last_output", context.input)  
  
        if isinstance(user_input, str):  
            user_input = Message.user(user_input)  
        elif isinstance(user_input, dict):  
            user_input = Message(**user_input)  
        elif isinstance(user_input, list) and user_input and isinstance(user_input[0], dict):  
            user_input = [Message(**msg) for msg in user_input]  
  
        output = await agent.run(user_input)  
        return output.model_dump() if hasattr(output, "model_dump") else output  
  
    async def _find_next_node(self, current: str, context: WorkflowContext) -> Optional[str]:  
        edges = self.edges.get(current, [])  
        for target, condition in edges:  
            if condition is None:  
                return target  
            try:  
                if inspect.iscoroutinefunction(condition):  
                    if await condition(context):  
                        return target  
                else:  
                    if condition(context):  
                        return target  
            except Exception as e:  
                logger.error("Condition evaluation failed", source=current, target=target, error=str(e))  
        return None  
  
    # 在 Workflow 类里增加一个通用执行方法  
    async def _execute_once(self, node_callable: Callable, context: WorkflowContext) -> Any:  
        if isinstance(node_callable, Agent):  
            return await self._run_agent_node(node_callable, context)  
        if callable(node_callable):  
            return await ensure_awaitable(node_callable, context)  
        raise WorkflowError(f"Node '{node_callable}' is not callable")  
    
    async def _execute_with_retry(self, func: Callable, context: WorkflowContext) -> Any:  
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
                    error=str(e),  
                )  
                if attempt < self.max_retries - 1:  
                    await asyncio.sleep(2 ** attempt)  
        # 所有重试都失败  
        raise last_error    
  
    def _normalize_result(self, result: Any) -> Any:  
        """  
        将节点输出转换为可序列化/可在历史中保存的格式  
        - BaseModel -> dict  
        - AgentOutput -> content/dict  
        - Message -> OpenAI 字典  
        - 原生类型保持不变  
        - 其他对象转为字符串（并截断）  
        """  
        if isinstance(result, BaseModel):  
            return result.model_dump()  
        if hasattr(result, "model_dump"):  
            data = result.model_dump()  
            return data.get("content", data)  
        if isinstance(result, Message):  
            return result.to_openai_format()  
        if isinstance(result, (str, int, float, bool, type(None))):  
            return result  
        if isinstance(result, (list, dict)):  
            return result  
        return str(result)[:500]  
  
    async def _persist_state(  
        self,  
        session_id: str,  
        steps: int,  
        current_node: Optional[str],  
        context: WorkflowContext,  
    ):  
        """状态持久化的统一入口，使用 try/except 避免影响主流程"""  
        try:  
            await self.storage.set(  
                f"workflow:{session_id}",  
                {  
                    "step": steps,  
                    "last_node": current_node,  
                    "context": context.to_dict(),  
                },  
            )  
        except Exception as e:  
            logger.warning("Failed to persist workflow state", error=str(e))  
  
    # ---------- 可视化/调试 ----------  
    def to_mermaid(self) -> str:  
        lines = ["graph TD"]  
        for node in self.nodes:  
            label = f"[{node}]" if node == self.entry_point else f"({node})"  
            lines.append(f"    {node}{label}")  
        for source, targets in self.edges.items():  
            for target, condition in targets:  
                label = f"|condition|" if condition else ""  
                lines.append(f"    {source} --{label}--> {target}")  
        return "\n".join(lines)  
  
    def print_structure(self):  
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
