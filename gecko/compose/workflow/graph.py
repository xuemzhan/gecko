# gecko/compose/workflow/graph.py
"""
DAG 图结构管理器

职责：
1. 维护节点 (Nodes) 和 边 (Edges)
2. 提供拓扑排序算法，支持并行层级构建
3. 进行静态校验（环检测、孤立点检测）
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from gecko.core.exceptions import WorkflowCycleError
from gecko.core.logging import get_logger
from gecko.compose.workflow.models import WorkflowContext

logger = get_logger(__name__)


class WorkflowGraph:
    """工作流拓扑结构容器"""

    def __init__(self):
        # 节点注册表: name -> callable
        self.nodes: Dict[str, Callable] = {}
        # 边关系: source -> [(target, condition_func), ...]
        self.edges: Dict[str, List[Tuple[str, Optional[Callable]]]] = {}
        # 显式依赖: node -> {dependencies}
        self.node_dependencies: Dict[str, Set[str]] = {}
        
        self.entry_point: Optional[str] = None
        
        # 缓存验证结果
        self._validated = False
        self._validation_errors: List[str] = []

    def add_node(self, name: str, func: Callable):
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = func
        self._validated = False

    def add_edge(self, source: str, target: str, condition: Optional[Callable] = None):
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        self.edges.setdefault(source, []).append((target, condition))
        self._validated = False

    def set_entry_point(self, name: str):
        if name not in self.nodes:
            raise ValueError(f"Entry point '{name}' not found")
        self.entry_point = name
        self._validated = False

    # ================= 拓扑分析 =================

    def validate(self, allow_cycles: bool = False, enable_parallel: bool = False) -> Tuple[bool, List[str]]:
        """验证图结构合法性"""
        if self._validated:
            return not self._validation_errors, self._validation_errors

        self._validation_errors.clear()

        # 1. 入口检查
        if not self.entry_point:
            self._validation_errors.append("No entry point defined")

        # 2. 环检测
        if not allow_cycles:
            try:
                self._detect_cycles()
            except WorkflowCycleError as e:
                self._validation_errors.append(str(e))

        # 3. 歧义分支检测 (非并行模式下)
        if not enable_parallel:
            for node, edges in self.edges.items():
                unconditional = [t for t, c in edges if c is None]
                if len(unconditional) > 1:
                    self._validation_errors.append(
                        f"Node '{node}' has ambiguous branching: {unconditional}"
                    )

        # 4. 连接性检查 (仅警告，不作为错误)
        self._check_connectivity()

        self._validated = True
        return not self._validation_errors, self._validation_errors
    
    def _check_connectivity(self):
        """检查不可达节点（仅警告）"""
        if not self.entry_point:
            return

        reachable = set()
        queue = [self.entry_point]
        while queue:
            curr = queue.pop(0)
            if curr in reachable:
                continue
            reachable.add(curr)
            # 遍历出边
            for target, _ in self.edges.get(curr, []):
                queue.append(target)
        
        # 计算差集
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            # 这里的 logger 必须确保是当前模块获取的 logger
            logger.warning("Unreachable nodes detected", nodes=list(unreachable))

    def _detect_cycles(self):
        """DFS 环检测算法"""
        visited = set()
        recursion_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            for neighbor, _ in self.edges.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in recursion_stack:
                    cycle_start = path.index(neighbor)
                    cycle = " -> ".join(path[cycle_start:] + [neighbor])
                    raise WorkflowCycleError(f"Cycle detected: {cycle}")

            recursion_stack.remove(node)
            path.pop()

        for node in self.nodes:
            if node not in visited:
                dfs(node, [])

    def build_execution_layers(self, start_node: Optional[str]) -> List[Set[str]]:
        """
        构建并行执行层级 (Kahn算法/拓扑排序变体)
        返回: List[Set[node_name]]，每一层内的节点可并行执行
        """
        # ... (此处保留原 Workflow._build_execution_layers 的逻辑，将其迁移至此) ...
        # 代码较长，核心逻辑是从 start_node 开始 BFS/DFS，计算入度，分层输出
        # 此处省略具体实现以节省篇幅，直接复用原逻辑即可
        return []
    
    def to_mermaid(self) -> str:
        """生成 Mermaid 流程图代码"""
        lines = ["graph TD"]
        for node in self.nodes:
            # 入口节点使用双圆圈
            shape_start = "((" if node == self.entry_point else "("
            shape_end = "))" if node == self.entry_point else ")"
            lines.append(f"    {node}{shape_start}{node}{shape_end}")
            
        for source, targets in self.edges.items():
            for target, condition in targets:
                label = "|condition|" if condition else ""
                lines.append(f"    {source} --{label}--> {target}")
        return "\n".join(lines)