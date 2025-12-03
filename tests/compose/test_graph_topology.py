# tests/compose/test_graph_topology.py
import pytest
from gecko.compose.workflow.graph import WorkflowGraph

class TestGraphTopology:
    
    def test_linear_layers(self):
        """测试线性依赖: A -> B -> C"""
        g = WorkflowGraph()
        g.add_node("A", lambda: None)
        g.add_node("B", lambda: None)
        g.add_node("C", lambda: None)
        
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        
        # 预期层级: [{A}, {B}, {C}]
        layers = g.build_execution_layers("A")
        assert len(layers) == 3
        assert layers[0] == {"A"}
        assert layers[1] == {"B"}
        assert layers[2] == {"C"}

    def test_diamond_layers(self):
        r"""
        测试菱形依赖 (并行):
          A
         / \
        B   C
         \ /
          D
        """
        g = WorkflowGraph()
        for n in ["A", "B", "C", "D"]:
            g.add_node(n, lambda: None)
            
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        
        # 预期层级: [{A}, {B, C}, {D}]
        layers = g.build_execution_layers("A")
        assert len(layers) == 3
        assert layers[0] == {"A"}
        assert layers[1] == {"B", "C"} or layers[1] == {"C", "B"}
        assert layers[2] == {"D"}

    def test_complex_dependencies(self):
        """
        测试复杂依赖:
        A -> B -> D
        A -> C -> D
        C -> E
        
        层级应为: [{A}, {B, C}, {E}, {D}] 
        注意: D 依赖 B(第2层) 和 C(第2层)，所以 D 必须在 B,C 之后。
        E 依赖 C，可以在第3层。D 具体在第3还是第4层取决于算法贪婪程度，
        Kahn算法通常会尽早调度。
        
        标准 Kahn:
        L1: {A} (removes A) -> in-degrees: B=0, C=0
        L2: {B, C} (removes B, C) -> in-degrees: D=0, E=0
        L3: {D, E}
        """
        g = WorkflowGraph()
        for n in ["A", "B", "C", "D", "E"]:
            g.add_node(n, lambda: None)
            
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        g.add_edge("C", "E")
        
        layers = g.build_execution_layers("A")
        
        # 验证依赖关系约束
        flat_layers = []
        for i, layer in enumerate(layers):
            for node in layer:
                flat_layers.append((node, i))
        
        layer_map = dict(flat_layers)
        
        assert layer_map["A"] < layer_map["B"]
        assert layer_map["A"] < layer_map["C"]
        assert layer_map["B"] < layer_map["D"]
        assert layer_map["C"] < layer_map["D"]
        assert layer_map["C"] < layer_map["E"]
        
        # 验证并行性: B 和 C 应该在同一层
        assert layer_map["B"] == layer_map["C"]

    def test_unreachable_nodes(self):
        """测试不可达节点不包含在执行计划中"""
        g = WorkflowGraph()
        g.add_node("A", lambda: None)
        g.add_node("B", lambda: None) # 孤立
        g.add_node("C", lambda: None) # 孤立
        
        g.add_edge("A", "B")
        
        # C 是不可达的
        layers = g.build_execution_layers("A")
        
        flattened = set().union(*layers)
        assert "A" in flattened
        assert "B" in flattened
        assert "C" not in flattened