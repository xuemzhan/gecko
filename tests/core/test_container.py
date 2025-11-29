# tests/core/test_container.py
import pytest
from gecko.core.container import Container, Lifetime

class ServiceA:
    pass

class ServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a

@pytest.mark.asyncio
async def test_container_transient_registration():
    container = Container()
    container.register(ServiceA, lifetime=Lifetime.TRANSIENT)
    
    a1 = container.resolve(ServiceA)
    a2 = container.resolve(ServiceA)
    
    assert isinstance(a1, ServiceA)
    assert a1 is not a2 # Transient 每次都是新实例

@pytest.mark.asyncio
async def test_container_singleton_registration():
    container = Container()
    container.register(ServiceA, lifetime=Lifetime.SINGLETON)
    
    a1 = container.resolve(ServiceA)
    a2 = container.resolve(ServiceA)
    
    assert a1 is a2 # Singleton 必须相同

@pytest.mark.asyncio
async def test_container_dependency_injection():
    """测试构造函数自动注入"""
    container = Container()
    container.register(ServiceA)
    container.register(ServiceB)
    
    b = container.resolve(ServiceB)
    
    assert isinstance(b, ServiceB)
    assert isinstance(b.service_a, ServiceA)

@pytest.mark.asyncio
async def test_container_scopes():
    """测试作用域"""
    container = Container()
    container.register(ServiceA, lifetime=Lifetime.SCOPED)
    
    async with container.create_scope() as scope1:
        a1 = scope1.resolve(ServiceA)
        a2 = scope1.resolve(ServiceA)
        assert a1 is a2 # 同一 Scope 内单例
        
        async with container.create_scope() as scope2:
            a3 = scope2.resolve(ServiceA)
            assert a1 is not a3 # 不同 Scope 实例不同