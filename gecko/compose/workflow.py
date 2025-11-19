# gecko/compose/workflow.py
from __future__ import annotations
import asyncio
import networkx as nx
import anyio
from typing import Any, Dict, List, Callable, Optional
from gecko.core.events import RunEvent, EventBus
from gecko.core.exceptions import WorkflowError
from gecko.core.message import Message
from gecko.core.agent import Agent
from gecko.compose.team import Team
from gecko.compose.nodes import Condition, Loop, Parallel

class Workflow:
    """
    Gecko DAG Workflow 引擎核心
    - 基于 NetworkX 建模有向无环图（DAG）
    - 支持顺序、并行、条件分支、循环
    - 异步执行：AnyIO TaskGroup 并行子图
    - 事件广播：每个节点执行前后 publish 事件
    - 可视化：to_mermaid() 生成 Markdown 图（补：添加条件标签）
    """
    def __init__(self, name: Optional[str] = None, event_bus: Optional[EventBus] = None):
        self.name = name or "GeckoWorkflow"
        self.graph = nx.DiGraph()  # NetworkX 图对象
        self.event_bus = event_bus or EventBus()
        self.nodes: Dict[str, Callable] = {}  # 节点注册表：name -> callable/Agent/Team
        # 内置起点/终点
        self.add_node("start", lambda ctx: ctx.get("input"))  # 输入节点
        self.add_node("end", lambda ctx: ctx)   # 输出节点（补：返回全 context）

    def add_node(self, name: str, node_func: Callable):
        """添加节点：支持 callable / Agent / Team"""
        if name in self.nodes:
            raise ValueError(f"节点 {name} 已存在")
        self.nodes[name] = node_func
        self.graph.add_node(name)
        return

    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None):
        """添加边：支持条件分支（condition 为 callable，返回 bool）"""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"节点 {from_node} 或 {to_node} 未定义")
        self.graph.add_edge(from_node, to_node, condition=condition)
        return self

    def validate(self):
        """验证 DAG：无环 + 连通（补：检查孤立节点）"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise WorkflowError("Workflow 包含循环")
        isolated = list(nx.isolates(self.graph))
        if isolated and set(isolated) - {"start", "end"}:
            raise WorkflowError(f"孤立节点: {isolated}")
        return self

    async def execute(self, input_data: Any) -> Any:
        """异步执行 Workflow（补：超时可选扩展）"""
        self.validate()
        context = {"input": input_data, "output": None}  # 全局上下文共享
        await self.event_bus.publish(RunEvent(type="workflow_started", data={"name": self.name}))

        # 拓扑排序执行顺序
        exec_order = list(nx.topological_sort(self.graph))

        async with anyio.create_task_group() as tg:
            for node_name in exec_order:
                if node_name in ["start", "end"]:
                    continue  # 内置跳过
                predecessors = list(self.graph.predecessors(node_name))
                # 检查条件边：只有满足条件的父节点输出才执行
                skip = False
                if predecessors:
                    for pred in predecessors:
                        edge_data = self.graph.get_edge_data(pred, node_name)
                        cond = edge_data.get("condition")
                        if cond and not await cond(context):
                            skip = True
                            break
                if skip:
                    continue

                # 调度节点执行（支持并行：TaskGroup 自动并发独立子图）
                tg.start_soon(self._run_node, node_name, context)
                
        # 最终执行终点节点，收集输出
        context["output"] = self.nodes["end"](context)
        await self.event_bus.publish(RunEvent(type="workflow_completed", data={"output": context["output"]}))
        return context["output"]

    async def _run_node(self, node_name: str, context: Dict):
        """内部节点执行器：对象优先 + 函数智能调用（最终修复版）"""
        await self.event_bus.publish(RunEvent(type="node_started", data={"node": node_name}))
        try:
            node_obj = self.nodes[node_name]

            # === 1. 对象节点优先处理（Team / Loop / Parallel / Agent）===
            if isinstance(node_obj, (Agent, Team, Loop, Parallel)):
                if hasattr(node_obj, "execute"):
                    result = await node_obj.execute(context)
                elif isinstance(node_obj, Agent):
                    # 强制 await Agent.run，防止模型未 await 警告
                    messages = [Message(role="user", content=str(context.get("input", "")))]
                    agent_output = await node_obj.run(messages)  # 保险 await
                    result = getattr(agent_output, "content", str(agent_output))
                else:
                    result = node_obj(context)  # fallback（理论不走）

            # === 2. 函数节点（@step 装饰的 def / async def）===
            elif callable(node_obj):
                if asyncio.iscoroutinefunction(node_obj):
                    result = await node_obj(context)
                else:
                    result = node_obj(context)

            # === 3. 其他类型（不支持）===
            else:
                raise TypeError(f"节点 {node_name} 类型不支持: {type(node_obj)}")

            context[node_name] = result
            await self.event_bus.publish(RunEvent(type="node_completed", data={"node": node_name, "result": result}))
        except Exception as e:
            error_msg = f"节点 {node_name} 执行失败: {e}"
            await self.event_bus.publish(RunEvent(type="node_error", data={"node": node_name, "error": error_msg}))
            raise WorkflowError(error_msg) from e

    def to_mermaid(self) -> str:
        """导出 Mermaid 图：支持条件边标签（修复版）"""
        lines = ["graph TD"]
        for from_node, to_node in self.graph.edges():
            edge_data = self.graph.get_edge_data(from_node, to_node)
            cond = edge_data.get("condition")
            
            if cond is None:
                label = ""
            elif callable(cond):
                # 直接传函数：lambda、def、partial 等
                label = f"|{getattr(cond, '__name__', 'condition')}|"
            elif isinstance(cond, Condition):
                # Condition 实例：优先用自定义 name，其次用 predicate 名称
                cond_name = getattr(cond, "name", None) or getattr(cond.predicate, "__name__", "condition")
                label = f"|{cond_name}|"
            else:
                label = "|condition|"
            
            lines.append(f"    {from_node} -->{label} {to_node}")
        
        # 添加节点样式（可选美化）
        for node in self.nodes:
            if node in ["start", "end"]:
                lines.append(f"    {node}[\"{node}\"]:::endpoint")
            else:
                lines.append(f"    {node}[\"{node}\"]")
        lines.append("    classDef endpoint fill:#e1f5fe,stroke:#333")
        
        return "\n".join(lines)