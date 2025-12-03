# gecko/compose/workflow/graph.py
"""
DAG 图结构管理器 (v0.4 Enhanced)

职责：
1. 维护节点 (Nodes) 和 边 (Edges)
2. 提供拓扑排序算法，支持并行层级构建 (Phase 1 核心)
3. 进行静态校验（环检测、孤立点检测）
"""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

from gecko.core.exceptions import WorkflowCycleError
from gecko.core.logging import get_logger

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
        """添加节点"""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = func
        self._validated = False

    def add_edge(self, source: str, target: str, condition: Optional[Callable] = None):
        """添加边"""
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        self.edges.setdefault(source, []).append((target, condition))
        self._validated = False

    def set_entry_point(self, name: str):
        """设置起始节点"""
        if name not in self.nodes:
            raise ValueError(f"Entry point '{name}' not found")
        self.entry_point = name
        self._validated = False

    # ================= 拓扑分析 (Phase 1 Core) =================

    def validate(self, allow_cycles: bool = False, enable_parallel: bool = False) -> Tuple[bool, List[str]]:
        """
        验证图结构合法性
        
        Args:
            allow_cycles: 是否允许环 (v0.4 暂不支持环的自动调度，通常为 False)
            enable_parallel: 是否启用并行模式。
                             若启用，允许一个节点有多个无条件出边 (分叉)。
        """
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

        # 3. 歧义分支检测 
        # [v0.4 优化] 如果启用了并行 (enable_parallel=True)，则允许无条件的多个出边 (Fork)
        # 只有在传统串行模式下，才认为多个无条件出边是歧义
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
    
    def build_execution_layers(self, start_node: Optional[str]) -> List[Set[str]]:
        r"""
        [核心算法] 构建并行执行层级 (Kahn算法/拓扑排序变体)
        
        功能:
        将 DAG 图转换为一系列执行层。每一层中的节点都只依赖于上一层的输出，
        因此同一层内的节点天然无依赖，可安全并行执行。
        
        示例:
             A
            / \
           B   C
            \ /
             D
        输出: [{A}, {B, C}, {D}]
        
        Args:
            start_node: 以此节点为根，遍历所有可达节点。
            
        Returns:
            List[Set[str]]: 分层列表
        """
        if not start_node or start_node not in self.nodes:
            return []

        # 1. 初始化数据结构
        # 仅跟踪从 start_node 可达的子图，忽略不可达的孤立点
        in_degree: Dict[str, int] = defaultdict(int)
        adj_list: Dict[str, List[str]] = defaultdict(list)
        reachable_nodes: Set[str] = {start_node}
        
        # 2. BFS 遍历构建子图 (计算入度和邻接表)
        queue = deque([start_node])
        
        # 必须显式初始化 start_node 的入度为 0
        in_degree[start_node] = 0

        while queue:
            curr = queue.popleft()
            
            # 获取所有出边的目标节点
            targets = [target for target, _ in self.edges.get(curr, [])]
            
            for target in targets:
                adj_list[curr].append(target)
                in_degree[target] += 1
                
                if target not in reachable_nodes:
                    reachable_nodes.add(target)
                    queue.append(target)

        # 3. 拓扑分层 (Kahn's Algorithm)
        layers: List[Set[str]] = []
        
        # 初始层：所有入度为 0 的节点 (在子图中通常只有 start_node)
        current_layer = {node for node in reachable_nodes if in_degree[node] == 0}

        while current_layer:
            layers.append(current_layer)
            next_layer = set()
            
            for node in current_layer:
                # 遍历当前层节点的所有下游
                for neighbor in adj_list[node]:
                    in_degree[neighbor] -= 1
                    # 当依赖全部满足 (入度归零) 时，加入下一层
                    if in_degree[neighbor] == 0:
                        next_layer.add(neighbor)
            
            current_layer = next_layer

        return layers

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