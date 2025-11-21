# tests/compose/test_nodes.py
"""
Nodes 模块单元测试

覆盖率目标：100%
测试范围：
1. Next 控制流指令模型
2. step 装饰器的元数据保留机制
3. step 装饰器的同步/异步统一包装能力
4. 节点属性标记 (_is_step, _step_name)
"""
import pytest
import inspect
import asyncio
from pydantic import ValidationError

from gecko.compose.nodes import Next, step

# ========================= 1. Next 模型测试 =========================

def test_next_model_initialization():
    """测试 Next 对象初始化"""
    # 1. 仅指定节点
    n1 = Next(node="TargetNode")
    assert n1.node == "TargetNode"
    assert n1.input is None

    # 2. 指定节点和输入
    n2 = Next(node="TargetNode", input={"data": 123})
    assert n2.node == "TargetNode"
    assert n2.input == {"data": 123}

def test_next_model_validation():
    """测试 Next 对象字段验证"""
    # 缺少必填字段 node
    with pytest.raises(ValidationError):
        Next(input="data") # type: ignore

    # input 可以是任意类型
    n = Next(node="A", input=123)
    assert n.input == 123

# ========================= 2. step 装饰器元数据测试 =========================

def test_step_metadata_preservation():
    """
    核心测试：验证 wraps 是否保留了原始函数的元数据
    这对 Workflow 的智能参数注入至关重要
    """
    
    @step()
    def sample_func(a: int, b: int = 1) -> int:
        """This is a docstring."""
        return a + b

    # 1. 验证名称和文档
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "This is a docstring."
    
    # 2. 验证签名 (Signature)
    sig = inspect.signature(sample_func)
    assert "a" in sig.parameters
    assert sig.parameters["a"].annotation == int
    assert "b" in sig.parameters
    assert sig.parameters["b"].default == 1
    
    # 3. 验证该函数变成了协程函数 (因为 wrapper 是 async def)
    assert inspect.iscoroutinefunction(sample_func)

def test_step_attributes_injection():
    """测试装饰器注入的特殊属性"""
    
    # Case 1: 默认名称
    @step()
    def func_a(): pass
    
    assert getattr(func_a, "_is_step") is True
    assert getattr(func_a, "_step_name") == "func_a"
    
    # Case 2: 自定义名称
    @step(name="CustomNodeName")
    def func_b(): pass
    
    assert getattr(func_b, "_is_step") is True
    assert getattr(func_b, "_step_name") == "CustomNodeName"

# ========================= 3. step 执行逻辑测试 =========================

@pytest.mark.asyncio
async def test_step_execution_sync():
    """测试装饰同步函数"""
    @step()
    def sync_add(x, y):
        return x + y
    
    # 装饰后应当可以被 await
    result = await sync_add(10, 20)
    assert result == 30

@pytest.mark.asyncio
async def test_step_execution_async():
    """测试装饰异步函数"""
    @step()
    async def async_mult(x, y):
        await asyncio.sleep(0.01)
        return x * y
    
    result = await async_mult(10, 20)
    assert result == 200

@pytest.mark.asyncio
async def test_step_execution_args_kwargs():
    """测试参数透传 (*args, **kwargs)"""
    @step()
    def messy_args(*args, **kwargs):
        return sum(args) + kwargs.get("val", 0)
    
    result = await messy_args(1, 2, 3, val=4)
    assert result == 10  # 1+2+3+4

@pytest.mark.asyncio
async def test_step_exception_propagation():
    """测试异常传播"""
    @step()
    def failing_func():
        raise ValueError("Boom")
    
    with pytest.raises(ValueError, match="Boom"):
        await failing_func()