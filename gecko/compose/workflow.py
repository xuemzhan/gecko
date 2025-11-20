# gecko/compose/workflow.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, Callable, Optional, List, Set, Tuple
from pydantic import BaseModel

from gecko.core.events import EventBus, RunEvent
from gecko.core.agent import Agent
from gecko.compose.nodes import Next, ensure_awaitable

class WorkflowContext(BaseModel):
    """Workflow 全局上下文容器"""
    input: Any
    history: Dict[str, Any] = {}  # 记录每个节点的输出
    state: Dict[str, Any] = {}    # 用户自定义全局状态
    
    # 允许任意类型
    model_config = {"arbitrary_types_allowed": True}

class Workflow:
    """
    Gecko v2 动态工作流引擎
    支持：循环、条件跳转、动态路由
    """
    def __init__(self, name: str = "Workflow", event_bus: Optional[EventBus] = None):
        self.name = name
        self.event_bus = event_bus or EventBus()
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, List[Tuple[str, Callable | None]]] = {} # from -> [(to, condition)]
        self.start_node: str | None = None
        
        # 熔断保护
        self.max_steps = 100 

    def add_node(self, name: str, func: Callable) -> 'Workflow':
        self.nodes[name] = func
        if self.start_node is None:
            self.start_node = name
        return self

    def set_entry_point(self, name: str) -> 'Workflow':
        if name not in self.nodes:
            raise ValueError(f"Node {name} not found")
        self.start_node = name
        return self

    def add_edge(self, source: str, target: str, condition: Optional[Callable[[Any], bool]] = None) -> 'Workflow':
        """
        定义默认流向。
        注意：如果节点返回 Next('target')，将优先于 add_edge 定义的静态路由。
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node not found")
        
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append((target, condition))
        return self

    async def execute(self, input_data: Any) -> Any:
        if not self.start_node:
            raise ValueError("No entry point defined")

        context = WorkflowContext(input=input_data)
        current_node_name = self.start_node
        steps_count = 0
        
        await self.event_bus.publish(RunEvent(type="workflow_started", data={"name": self.name}))

        while current_node_name and current_node_name != "END":
            if steps_count >= self.max_steps:
                raise RuntimeError(f"Workflow exceeded max steps ({self.max_steps})")
            
            steps_count += 1
            
            # 1. 获取节点函数
            node_func = self.nodes[current_node_name]
            
            # 2. 准备输入 (默认传入 context，如果节点是 Agent 则特殊处理)
            # 简单起见，统一传 context，节点内部自己解析
            # 或者：如果是 Agent，取 context.input 或上一步结果
            
            await self.event_bus.publish(RunEvent(type="node_started", data={"node": current_node_name}))
            
            try:
                # 执行节点
                if isinstance(node_func, Agent):
                    # Agent 适配：输入取上一轮结果或初始 input
                    agent_input = context.history.get("last_output", context.input)
                    result = await node_func.run(agent_input)
                else:
                    # 普通函数
                    result = await ensure_awaitable(node_func, context)
                
                # 记录结果
                context.history[current_node_name] = result
                context.history["last_output"] = result # 方便链式调用
                
                await self.event_bus.publish(RunEvent(type="node_completed", data={"node": current_node_name, "result": str(result)}))

            except Exception as e:
                await self.event_bus.publish(RunEvent(type="node_error", error=str(e)))
                raise e

            # 3. 路由决议 (Routing)
            
            # 情况 A: 节点显式返回了跳转指令
            if isinstance(result, Next):
                current_node_name = result.node
                if result.input is not None:
                    # 更新下一次的隐式输入
                    context.history["last_output"] = result.input
                continue

            # 情况 B: 查找静态路由表 (Edges)
            next_node = None
            possible_edges = self.edges.get(current_node_name, [])
            
            for target, condition in possible_edges:
                if condition:
                    # 判断条件
                    if await ensure_awaitable(condition, context):
                        next_node = target
                        break
                else:
                    # 无条件边 (Default)
                    next_node = target
                    break
            
            current_node_name = next_node # 如果为 None，循环结束
            
        await self.event_bus.publish(RunEvent(type="workflow_completed", data={"output": context.history.get("last_output")}))
        return context.history.get("last_output")