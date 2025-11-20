# gecko/core/toolbox.py
"""
ToolBox（Phase 2 优化版）

改进：
1. 并发执行优化
2. 超时控制
3. 错误隔离
4. 执行统计
"""
from __future__ import annotations
import asyncio
import time
from typing import List, Dict, Any, Optional
import anyio

from gecko.plugins.tools.base import BaseTool, ToolProtocol
from gecko.core.logging import get_logger
from gecko.core.exceptions import ToolError, ToolNotFoundError, ToolTimeoutError
from gecko.config import settings

logger = get_logger(__name__)

class ToolBox:
    """
    Agent 工具箱（优化版）
    
    新增功能：
    1. 并发执行控制
    2. 超时保护
    3. 执行统计
    4. 错误隔离
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        max_concurrent: int = 5,  # ✅ 最大并发数
        default_timeout: float | None = None,  # ✅ 默认超时
    ):
        self._tools: Dict[str, BaseTool] = {}
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout or settings.tool_execution_timeout
        
        # ✅ 执行统计
        self._execution_count: Dict[str, int] = {}
        self._error_count: Dict[str, int] = {}
        self._total_time: Dict[str, float] = {}
        
        # 注册工具
        if tools:
            for tool in tools:
                self.register(tool)

    # ========== 工具注册 ==========

    def register(self, tool: BaseTool):
        """注册工具"""
        if tool.name in self._tools:
            logger.warning("Tool already registered, replacing", tool=tool.name)
        
        self._tools[tool.name] = tool
        self._execution_count[tool.name] = 0
        self._error_count[tool.name] = 0
        self._total_time[tool.name] = 0.0
        
        logger.debug("Tool registered", tool=tool.name)

    def unregister(self, tool_name: str):
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug("Tool unregistered", tool=tool_name)

    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """列出所有工具"""
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    # ========== OpenAI Schema ==========

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """生成 OpenAI function calling 格式"""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return schemas

    # ========== 执行（单个工具） ==========

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> str:
        """
        执行单个工具
        
        参数:
            name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒），None 使用默认值
        
        返回:
            工具执行结果（字符串）
        
        抛出:
            ToolNotFoundError: 工具不存在
            ToolTimeoutError: 执行超时
            ToolError: 执行失败
        """
        # 1. 检查工具是否存在
        tool = self.get(name)
        if not tool:
            raise ToolNotFoundError(name)
        
        # 2. 确定超时时间
        actual_timeout = timeout or self.default_timeout
        
        # 3. 执行（带超时）
        start_time = time.time()
        
        logger.debug("Executing tool", tool=name, timeout=actual_timeout)
        
        try:
            # 使用 anyio 的超时控制
            with anyio.fail_after(actual_timeout):
                result = await tool.execute(arguments)
            
            # 更新统计
            execution_time = time.time() - start_time
            self._execution_count[name] += 1
            self._total_time[name] += execution_time
            
            logger.info(
                "Tool executed successfully",
                tool=name,
                duration=f"{execution_time:.3f}s"
            )
            
            return str(result)
        
        except TimeoutError:
            # 超时
            self._error_count[name] += 1
            logger.error(
                "Tool execution timeout",
                tool=name,
                timeout=actual_timeout
            )
            raise ToolTimeoutError(name, actual_timeout)
        
        except Exception as e:
            # 其他错误
            self._error_count[name] += 1
            logger.error(
                "Tool execution failed",
                tool=name,
                error=str(e)
            )
            raise ToolError(
                f"Tool '{name}' execution failed: {e}",
                context={"tool": name, "arguments": arguments}
            ) from e

    # ========== 并发执行（多个工具） ==========

    async def execute_many(
        self,
        tool_calls: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        并发执行多个工具
        
        参数:
            tool_calls: 工具调用列表，格式:
                [
                    {
                        "name": "tool1",
                        "arguments": {...},
                        "id": "call_123"  # 可选
                    },
                    ...
                ]
            timeout: 单个工具的超时时间
        
        返回:
            结果列表，格式:
                [
                    {
                        "id": "call_123",
                        "name": "tool1",
                        "result": "...",
                        "is_error": False
                    },
                    ...
                ]
        """
        if not tool_calls:
            return []
        
        logger.info(
            "Executing tools concurrently",
            count=len(tool_calls),
            max_concurrent=self.max_concurrent
        )
        
        results: List[Dict[str, Any]] = []
        
        # 信号量控制并发数
        semaphore = anyio.Semaphore(self.max_concurrent)
        
        async def _execute_one(call: Dict[str, Any]):
            """执行单个工具（带并发控制）"""
            async with semaphore:
                tool_name = call.get("name")
                arguments = call.get("arguments", {})
                call_id = call.get("id", "")
                
                try:
                    result = await self.execute(tool_name, arguments, timeout)
                    results.append({
                        "id": call_id,
                        "name": tool_name,
                        "result": result,
                        "is_error": False
                    })
                except Exception as e:
                    results.append({
                        "id": call_id,
                        "name": tool_name,
                        "result": str(e),
                        "is_error": True
                    })
        
        # 并发执行
        async with anyio.create_task_group() as tg:
            for call in tool_calls:
                tg.start_soon(_execute_one, call)
        
        logger.info("All tools executed", total=len(results))
        
        return results

    # ========== 统计信息 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = {}
        
        for tool_name in self._tools:
            exec_count = self._execution_count.get(tool_name, 0)
            error_count = self._error_count.get(tool_name, 0)
            total_time = self._total_time.get(tool_name, 0.0)
            
            avg_time = total_time / exec_count if exec_count > 0 else 0.0
            success_rate = (exec_count - error_count) / exec_count if exec_count > 0 else 1.0
            
            stats[tool_name] = {
                "executions": exec_count,
                "errors": error_count,
                "total_time": total_time,
                "avg_time": avg_time,
                "success_rate": success_rate
            }
        
        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print("\n=== ToolBox Statistics ===")
        for tool_name, data in stats.items():
            print(f"\n{tool_name}:")
            print(f"  Executions: {data['executions']}")
            print(f"  Errors: {data['errors']}")
            print(f"  Success Rate: {data['success_rate']:.1%}")
            print(f"  Avg Time: {data['avg_time']:.3f}s")
            print(f"  Total Time: {data['total_time']:.3f}s")

    def reset_stats(self):
        """重置统计信息"""
        for tool_name in self._tools:
            self._execution_count[tool_name] = 0
            self._error_count[tool_name] = 0
            self._total_time[tool_name] = 0.0