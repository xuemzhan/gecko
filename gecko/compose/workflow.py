# gecko/compose/workflow.py
from __future__ import annotations
import asyncio
import networkx as nx
import anyio
from typing import Any, Dict, Callable, Optional, List
from gecko.core.events import RunEvent, EventBus
from gecko.core.exceptions import WorkflowError
from gecko.core.message import Message
from gecko.core.agent import Agent
from gecko.compose.team import Team
from gecko.compose.nodes import Condition, Loop, Parallel
from gecko.core.utils import ensure_awaitable  # [修复] 导入辅助函数

class Workflow:
    """
    Gecko DAG Workflow 引擎核心
    """
    def __init__(self, name: Optional[str] = None, event_bus: Optional[EventBus] = None):
        self.name = name or "GeckoWorkflow"
        self.graph = nx.DiGraph()
        self.event_bus = event_bus or EventBus()
        self.nodes: Dict[str, Callable] = {}
        
        # 内置起点：从 context 中提取 'input'
        self.add_node("start", lambda ctx: ctx.get("input") if isinstance(ctx, dict) else ctx)
        # 注意：end 节点在 add_edge 时动态处理，或作为逻辑终点

    def add_node(self, name: str, node_func: Callable):
        """注册节点"""
        if name in self.nodes:
            raise ValueError(f"节点 {name} 已存在")
        self.nodes[name] = node_func
        self.graph.add_node(name)
        return self

    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None):
        """添加边，支持条件判断"""
        # 自动注册 end 节点作为占位符，方便拓扑排序
        if to_node == "end" and "end" not in self.nodes:
             self.nodes["end"] = lambda ctx: ctx 
             self.graph.add_node("end")

        if from_node not in self.nodes:
            raise ValueError(f"源节点 {from_node} 未定义")
        if to_node not in self.nodes:
             raise ValueError(f"目标节点 {to_node} 未定义")
            
        self.graph.add_edge(from_node, to_node, condition=condition)
        return self

    def validate(self):
        """检查 DAG 合法性"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise WorkflowError("Workflow 包含循环，无法执行拓扑排序")
        return self

    async def execute(self, input_data: Any, return_context: bool = False) -> Any:
        """
        异步执行 Workflow
        
        Args:
            input_data: 输入数据
            return_context: 是否返回完整上下文字典（Loop 节点需要设为 True）
        """
        self.validate()
        
        # 初始化上下文
        context = {"input": input_data, "output": None}
        await self.event_bus.publish(RunEvent(type="workflow_started", data={"name": self.name}))

        # 获取执行顺序
        exec_order = list(nx.topological_sort(self.graph))

        # 使用 TaskGroup 执行
        async with anyio.create_task_group() as tg:
            for node_name in exec_order:
                if node_name == "start":
                    continue
                
                # 遇到 end 节点跳过执行逻辑，它只是一个标记
                if node_name == "end":
                    continue

                # 检查前置节点的条件
                predecessors = list(self.graph.predecessors(node_name))
                skip = False
                if predecessors:
                    for pred in predecessors:
                        edge_data = self.graph.get_edge_data(pred, node_name)
                        cond = edge_data.get("condition")
                        # [修复] 条件判断也要支持异步/同步混合
                        if cond and not await ensure_awaitable(cond, context):
                            skip = True
                            break
                if skip:
                    continue

                # 调度节点执行
                tg.start_soon(self._run_node, node_name, context)
                
        # [修复] 如果请求返回上下文，直接返回 context 字典
        if return_context: # type: ignore
            return context

        # [修复] 智能推断返回值
        # 1. 如果有显式的 "end" 节点
        final_output = context.get("output")
        if "end" in self.nodes:
            predecessors = list(self.graph.predecessors("end"))
            if len(predecessors) == 1:
                # 只有一个节点指向 end，返回该节点的执行结果
                final_output = context.get(predecessors[0])
            elif len(predecessors) > 1:
                # 多个节点指向 end，返回结果字典
                final_output = {k: context.get(k) for k in predecessors}
        
        # 2. 如果没有显式 output 且没有 end 节点逻辑，尝试返回最后一个非 end 节点的结果
        # 这解决了 test_simple_dag 断言 21 的问题
        if final_output is None and len(exec_order) >= 2:
             # 最后一个通常是 end，倒数第二个是最后一个实际执行的节点
             last_real_node = exec_order[-2] if exec_order[-1] == "end" else exec_order[-1]
             if last_real_node != "start":
                 final_output = context.get(last_real_node)

        await self.event_bus.publish(RunEvent(type="workflow_completed", data={"output": final_output}))
        return final_output

    async def _run_node(self, node_name: str, context: Dict):
        """内部节点执行逻辑"""
        await self.event_bus.publish(RunEvent(type="node_started", data={"node": node_name}))
        try:
            node_obj = self.nodes[node_name]
            result = None

            # 1. 处理高级对象节点
            if isinstance(node_obj, (Agent, Team, Loop, Parallel)):
                 if hasattr(node_obj, "execute"):
                    result = await node_obj.execute(context)
                 elif isinstance(node_obj, Agent):
                    # [修复] 修正 Agent 作为节点时的调用方式
                    # 尝试从 context 中获取输入，优先取 input，否则取上一步的结果
                    content = context.get("input")
                    # 简化的上下文传递逻辑，实际可能需要更复杂的 prompt 组装
                    messages = [Message(role="user", content=str(content))]
                    agent_output = await node_obj.run(messages)
                    result = agent_output.content
            
            # 2. 处理函数节点
            elif callable(node_obj):
                # [修复] 使用 ensure_awaitable 统一处理
                result = await ensure_awaitable(node_obj, context)
            
            else:
                 raise TypeError(f"节点 {node_name} 类型不支持: {type(node_obj)}")

            # 将结果写入上下文
            context[node_name] = result
            await self.event_bus.publish(RunEvent(type="node_completed", data={"node": node_name, "result": result}))
            
        except Exception as e:
            error_msg = f"节点 {node_name} 执行失败: {e}"
            # 打印错误堆栈以便调试
            import traceback
            traceback.print_exc()
            
            await self.event_bus.publish(RunEvent(type="node_error", data={"node": node_name, "error": error_msg}))
            raise WorkflowError(error_msg) from e

    def to_mermaid(self) -> str:
        """生成 Mermaid 流程图代码"""
        lines = ["graph TD"]
        for from_node, to_node in self.graph.edges():
            edge_data = self.graph.get_edge_data(from_node, to_node)
            cond = edge_data.get("condition")
            
            label = ""
            if cond:
                if isinstance(cond, Condition) and cond.name:
                    label = f"|{cond.name}|"
                elif hasattr(cond, "__name__"):
                    label = f"|{cond.__name__}|"
                else:
                    label = "|condition|"
            
            lines.append(f"    {from_node} -->{label} {to_node}")
        
        # 标记起止节点
        for node in self.nodes:
            if node in ["start", "end"]:
                lines.append(f"    {node}[\"{node}\"]:::endpoint")
            else:
                lines.append(f"    {node}[\"{node}\"]")
        lines.append("    classDef endpoint fill:#f9f,stroke:#333,stroke-width:2px")
        
        return "\n".join(lines)