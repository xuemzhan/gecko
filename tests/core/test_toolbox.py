# tests/core/test_toolbox.py
import pytest
import asyncio
from typing import Type
from pydantic import BaseModel, Field

from gecko.core.toolbox import ToolBox
from gecko.core.exceptions import ToolNotFoundError
from gecko.plugins.tools.base import BaseTool, ToolResult

# 1. 定义参数模型 (适配新版 BaseTool)
class MockArgs(BaseModel):
    # 允许任意参数，方便测试
    model_config = {"extra": "allow"}
    key: str = Field(default="")
    index: int = Field(default=0)

class EmptyArgs(BaseModel):
    pass

# 2. 定义 Mock 工具 (适配新版 BaseTool 接口)
class MockTool(BaseTool):
    """测试用的 Mock 工具"""
    name: str = "mock_tool"
    description: str = "测试工具"
    args_schema: Type[BaseModel] = MockArgs
    
    # 新版 BaseTool 要求实现 _run 而不是 execute
    async def _run(self, args: MockArgs) -> ToolResult: # type: ignore
        await asyncio.sleep(0.1)  # 模拟耗时
        # 返回 ToolResult 或 字符串
        return ToolResult(content=f"Mock result: {args.model_dump()}")


class SlowTool(BaseTool):
    """慢速工具（用于测试超时）"""
    name: str = "slow_tool"
    description: str = "慢速工具"
    args_schema: Type[BaseModel] = EmptyArgs
    
    async def _run(self, args: EmptyArgs) -> str: # type: ignore
        await asyncio.sleep(10)  # 超过默认超时
        return "This should timeout"


class ErrorTool(BaseTool):
    """会抛出异常的工具"""
    name: str = "error_tool"
    description: str = "错误工具"
    args_schema: Type[BaseModel] = EmptyArgs
    
    async def _run(self, args: EmptyArgs) -> str: # type: ignore
        raise ValueError("Intentional error")


@pytest.fixture
def toolbox():
    """创建测试用工具箱"""
    return ToolBox(
        tools=[MockTool()],
        max_concurrent=2,
        default_timeout=1.0
    )


class TestToolBoxBasics:
    """测试基础功能"""
    
    def test_register_tool(self, toolbox):
        """测试工具注册"""
        assert len(toolbox.list_tools()) == 1
        assert toolbox.has_tool("mock_tool")
        
    def test_unregister_tool(self, toolbox):
        """测试工具注销"""
        toolbox.unregister("mock_tool")
        assert len(toolbox.list_tools()) == 0
        assert not toolbox.has_tool("mock_tool")
    
    def test_duplicate_registration(self, toolbox):
        """测试重复注册"""
        # 默认允许替换
        toolbox.register(MockTool(), replace=True)
        assert len(toolbox.list_tools()) == 1
        
        # 不允许替换时应抛出异常
        with pytest.raises(ValueError):
            toolbox.register(MockTool(), replace=False)
    
    def test_openai_schema(self, toolbox):
        """测试 OpenAI Schema 生成"""
        schema = toolbox.to_openai_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "mock_tool"
        # 验证 Schema 中是否包含 properties
        assert "properties" in schema[0]["function"]["parameters"]


class TestToolExecution:
    """测试工具执行"""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, toolbox):
        """测试成功执行"""
        result = await toolbox.execute(
            "mock_tool",
            {"key": "value"}
        )
        assert "Mock result" in result
        assert "value" in result
    
    @pytest.mark.asyncio
    async def test_execute_not_found(self, toolbox):
        """测试工具不存在"""
        # ToolBox.execute 会抛出 ToolNotFoundError
        with pytest.raises(ToolNotFoundError):
            await toolbox.execute("non_existent", {})
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """测试超时"""
        toolbox = ToolBox(
            tools=[SlowTool()],
            default_timeout=0.5
        )
        
        result = await toolbox.execute_with_result(
            "slow_tool",
            {}
        )
        assert result.is_error
        # 注意：ToolBox 中的超时提示是英文 "timed out"
        assert "timed out" in result.result
    
    @pytest.mark.asyncio
    async def test_execute_error(self):
        """测试工具执行异常"""
        toolbox = ToolBox(tools=[ErrorTool()])
        
        result = await toolbox.execute_with_result(
            "error_tool",
            {}
        )
        assert result.is_error
        assert "Intentional error" in result.result


class TestBatchExecution:
    """测试批量执行"""
    
    @pytest.mark.asyncio
    async def test_execute_many(self, toolbox):
        """测试并发执行"""
        tool_calls = [
            {"id": f"call_{i}", "name": "mock_tool", "arguments": {"index": i}}
            for i in range(5)
        ]
        
        results = await toolbox.execute_many(tool_calls)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.call_id == f"call_{i}"
            assert not result.is_error
    
    @pytest.mark.asyncio
    async def test_execute_many_empty(self, toolbox):
        """测试空列表"""
        results = await toolbox.execute_many([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_execute_many_mixed(self):
        """测试混合成功和失败"""
        toolbox = ToolBox(
            tools=[MockTool(), ErrorTool()],
            default_timeout=1.0
        )
        
        tool_calls = [
            {"id": "1", "name": "mock_tool", "arguments": {}},
            {"id": "2", "name": "error_tool", "arguments": {}},
            {"id": "3", "name": "mock_tool", "arguments": {}},
        ]
        
        results = await toolbox.execute_many(tool_calls)
        
        assert len(results) == 3
        assert not results[0].is_error
        assert results[1].is_error
        assert not results[2].is_error


class TestStatistics:
    """测试统计功能"""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, toolbox):
        """测试统计追踪"""
        # 执行几次
        for i in range(3):
            await toolbox.execute("mock_tool", {})
        
        stats = toolbox.get_stats()
        assert stats["mock_tool"]["calls"] == 3
        assert stats["mock_tool"]["errors"] == 0
        assert stats["mock_tool"]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_stats_with_errors(self):
        """测试包含错误的统计"""
        toolbox = ToolBox(tools=[ErrorTool()])
        
        for i in range(2):
            # 使用 execute_with_result 以避免抛出异常
            await toolbox.execute_with_result("error_tool", {})
        
        stats = toolbox.get_stats()
        assert stats["error_tool"]["calls"] == 2
        assert stats["error_tool"]["errors"] == 2
        assert stats["error_tool"]["success_rate"] == 0.0
    
    # [修改点] 加上 async 标记，改为异步函数
    @pytest.mark.asyncio
    async def test_reset_stats(self, toolbox):
        """测试统计重置"""
        # 先制造一些数据
        # [修改点] 使用 await 替代 asyncio.run()
        await toolbox.execute("mock_tool", {})
        
        toolbox.reset_stats()
        stats = toolbox.get_stats()
        
        # 验证所有统计数据都被重置
        if stats:
            assert all(s["calls"] == 0 for s in stats.values())
        else:
            # 如果重置是清空字典，这也是一种有效状态
            assert len(stats) == 0 or all(s["calls"] == 0 for s in stats.values())


class TestRetry:
    """测试重试机制"""
    
    @pytest.mark.asyncio
    async def test_retry_disabled(self):
        """测试禁用重试"""
        toolbox = ToolBox(
            tools=[ErrorTool()],
            enable_retry=False
        )
        
        result = await toolbox.execute_with_result("error_tool", {})
        assert result.is_error
        # 验证没有重试日志或行为（通过执行时间或 Mock 验证，这里简单验证结果）
    
    @pytest.mark.asyncio
    async def test_retry_enabled(self):
        """测试启用重试"""
        # ErrorTool 每次都会失败，所以重试最终也会失败，但过程会不同
        toolbox = ToolBox(
            tools=[ErrorTool()],
            enable_retry=True,
            max_retries=2
        )
        
        # 由于重试有 sleep，这个测试会稍微慢一点
        result = await toolbox.execute_with_result("error_tool", {})
        assert result.is_error