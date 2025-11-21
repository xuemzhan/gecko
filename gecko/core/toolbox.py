# gecko/core/toolbox.py
"""
ToolBox - Agent 工具箱

核心功能：
1. 工具注册与管理
2. 单个/批量工具执行
3. 并发控制与超时管理
4. 执行统计与监控
5. OpenAI Function Calling Schema 生成

优化点：
- 修复并发控制的信号量使用方式（在任务内部 acquire/release）
- 线程安全的统计数据（使用 threading.Lock）
- 统一的返回值类型（ToolExecutionResult）
- 完善的错误处理和日志
- 可配置的重试机制
"""
from __future__ import annotations

import asyncio
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anyio import create_task_group, fail_after, to_thread
from anyio import get_cancelled_exc_class  # ✅ 用于捕获取消异常

from gecko.config import settings
from gecko.core.exceptions import ToolError, ToolNotFoundError, ToolTimeoutError
from gecko.core.logging import get_logger
from gecko.plugins.tools.base import BaseTool

logger = get_logger(__name__)


# ===== 返回值模型 =====

@dataclass
class ToolExecutionResult:
    """
    工具执行结果（统一返回类型）
    
    属性:
        tool_name: 工具名称
        call_id: 调用 ID（用于关联请求）
        result: 执行结果（成功时为字符串，失败时为错误信息）
        is_error: 是否执行失败
        duration: 执行耗时（秒）
        metadata: 附加信息
    """
    tool_name: str
    call_id: str
    result: str
    is_error: bool
    duration: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "result": self.result,
            "is_error": self.is_error,
            "duration": self.duration,
            "metadata": self.metadata,
        }


# ===== 工具箱主类 =====

class ToolBox:
    """
    Agent 工具箱
    
    负责工具的注册、执行、并发控制和统计。
    
    示例:
        ```python
        # 创建工具箱
        toolbox = ToolBox(
            tools=[search_tool, calculator],
            max_concurrent=5,
            default_timeout=30.0
        )
        
        # 单个工具执行
        result = await toolbox.execute(
            "search",
            {"query": "Python asyncio"}
        )
        print(result)  # "搜索结果：..."
        
        # 批量并发执行
        tool_calls = [
            {"id": "1", "name": "search", "arguments": {"query": "AI"}},
            {"id": "2", "name": "calculator", "arguments": {"expression": "2+2"}},
        ]
        results = await toolbox.execute_many(tool_calls)
        for r in results:
            print(f"{r.tool_name}: {r.result}")
        
        # 查看统计
        toolbox.print_stats()
        ```
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        max_concurrent: int = 5,
        default_timeout: float | None = None,
        enable_retry: bool = False,
        max_retries: int = 2,
    ):
        """
        初始化工具箱
        
        参数:
            tools: 初始工具列表
            max_concurrent: 最大并发执行数（默认 5）
            default_timeout: 默认超时时间（秒，默认读取配置）
            enable_retry: 是否启用工具执行重试（默认 False）
            max_retries: 最大重试次数（默认 2）
        """
        # 工具存储
        self._tools: Dict[str, BaseTool] = {}
        
        # 并发与超时配置
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout or settings.tool_execution_timeout
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        
        # 统计数据（线程安全）
        self._stats_lock = threading.Lock()
        self._execution_count: Dict[str, int] = defaultdict(int)
        self._error_count: Dict[str, int] = defaultdict(int)
        self._total_time: Dict[str, float] = defaultdict(float)
        
        # 注册初始工具
        if tools:
            for tool in tools:
                self.register(tool)
    
    # ====================== 工具管理 ======================
    
    def register(self, tool: BaseTool, replace: bool = True) -> "ToolBox":
        """
        注册工具
        
        参数:
            tool: 工具实例（必须继承 BaseTool）
            replace: 如果工具名已存在，是否替换（默认 True）
        
        返回:
            self（支持链式调用）
        
        异常:
            ValueError: 工具名已存在且 replace=False
            TypeError: 工具不继承 BaseTool
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"工具必须继承 BaseTool，收到类型: {type(tool).__name__}"
            )
        
        if tool.name in self._tools and not replace:
            raise ValueError(
                f"工具 '{tool.name}' 已注册，如需覆盖请设置 replace=True"
            )
        
        if tool.name in self._tools:
            logger.warning(
                "Tool replaced",
                tool_name=tool.name,
                old_type=type(self._tools[tool.name]).__name__,
                new_type=type(tool).__name__
            )
        
        self._tools[tool.name] = tool
        
        # 初始化统计数据
        with self._stats_lock:
            if tool.name not in self._execution_count:
                self._execution_count[tool.name] = 0
                self._error_count[tool.name] = 0
                self._total_time[tool.name] = 0.0
        
        logger.info("Tool registered", tool_name=tool.name, description=tool.description[:50])
        return self
    
    def unregister(self, tool_name: str) -> "ToolBox":
        """
        注销工具
        
        参数:
            tool_name: 工具名称
        
        返回:
            self（支持链式调用）
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info("Tool unregistered", tool_name=tool_name)
        else:
            logger.warning("Attempt to unregister non-existent tool", tool_name=tool_name)
        
        return self
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        获取工具实例
        
        参数:
            name: 工具名称
        
        返回:
            工具实例，如果不存在返回 None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """
        获取所有已注册的工具
        
        返回:
            工具列表
        """
        return list(self._tools.values())
    
    def has_tool(self, name: str) -> bool:
        """
        检查工具是否存在
        
        参数:
            name: 工具名称
        
        返回:
            是否存在
        """
        return name in self._tools
    
    # ====================== OpenAI Schema ======================
    
    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """
        生成 OpenAI Function Calling 所需的 schema
        
        返回:
            工具定义列表，格式符合 OpenAI API 规范
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        return schemas
    
    # ====================== 单个工具执行 ======================
    
    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        call_id: str = "",
    ) -> str:
        """
        执行单个工具（简化版，仅返回字符串）
        
        参数:
            name: 工具名称
            arguments: 工具参数（字典）
            timeout: 超时时间（秒），None 则使用默认值
            call_id: 调用 ID（可选，用于日志追踪）
        
        返回:
            执行结果字符串
        
        异常:
            ToolNotFoundError: 工具不存在
            ToolTimeoutError: 执行超时
            ToolError: 执行失败
        """
        result = await self.execute_with_result(name, arguments, timeout, call_id)
        
        if result.is_error:
            raise ToolError(
                f"工具 '{name}' 执行失败: {result.result}",
                context={
                    "tool_name": name,
                    "arguments": arguments,
                    "call_id": call_id,
                }
            )
        
        return result.result
    
    async def execute_with_result(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        call_id: str = "",
    ) -> ToolExecutionResult:
        """
        执行单个工具（完整版，返回结构化结果）
        
        参数:
            name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒）
            call_id: 调用 ID
        
        返回:
            ToolExecutionResult 包含结果和元数据
        """
        # 检查工具是否存在
        tool = self.get(name)
        if not tool:
            self._update_stats(name, 0, is_error=True)
            raise ToolNotFoundError(name)
        
        actual_timeout = timeout or self.default_timeout
        start_time = time.time()
        
        logger.debug(
            "Tool execution started",
            tool=name,
            timeout=actual_timeout,
            call_id=call_id
        )
        
        # 执行（带重试）
        if self.enable_retry:
            result_str, is_error = await self._execute_with_retry(
                tool, arguments, actual_timeout, name
            )
        else:
            result_str, is_error = await self._execute_once(
                tool, arguments, actual_timeout, name
            )
        
        duration = time.time() - start_time
        
        # 更新统计
        self._update_stats(name, duration, is_error=is_error)
        
        # 记录日志
        if is_error:
            logger.error(
                "Tool execution failed",
                tool=name,
                duration=f"{duration:.3f}s",
                error=result_str[:200],
                call_id=call_id
            )
        else:
            logger.info(
                "Tool execution succeeded",
                tool=name,
                duration=f"{duration:.3f}s",
                result_length=len(result_str),
                call_id=call_id
            )
        
        return ToolExecutionResult(
            tool_name=name,
            call_id=call_id,
            result=result_str,
            is_error=is_error,
            duration=duration,
        )
    
    async def _execute_once(
    self,
    tool: BaseTool,
    arguments: Dict[str, Any],
    timeout: float,
    tool_name: str,
    ) -> tuple[str, bool]:
        """
        单次执行工具（内部方法）
        
        返回:
            (result_string, is_error)
        """
        try:
            # ✅ 使用同步 with（不是 async with）
            with fail_after(timeout):
                result = await tool.execute(arguments)
        except TimeoutError:
            # ✅ 捕获标准库的 TimeoutError（anyio.fail_after 会抛出这个）
            return f"工具执行超时（{timeout}秒）", True
        except get_cancelled_exc_class():
            # ✅ 防御性捕获：某些边缘情况可能只抛出 CancelledError
            return f"工具执行被取消（{timeout}秒）", True
        except Exception as e:
            logger.exception("Tool execution exception", tool=tool_name)
            return f"执行异常: {str(e)}", True
        
        # 确保返回字符串
        if isinstance(result, str):
            return result, False
        else:
            return str(result), False
    
    async def _execute_with_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        timeout: float,
        tool_name: str,
    ) -> tuple[str, bool]:
        """
        带重试的工具执行
        
        返回:
            (result_string, is_error)
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result_str, is_error = await self._execute_once(
                    tool, arguments, timeout, tool_name
                )
                
                if not is_error:
                    if attempt > 0:
                        logger.info(
                            "Tool succeeded after retry",
                            tool=tool_name,
                            attempt=attempt + 1
                        )
                    return result_str, False
                
                last_error = result_str
                
                # 如果还有重试机会，等待后重试
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(
                        "Tool failed, retrying",
                        tool=tool_name,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        wait_time=wait_time,
                        error=result_str[:100]
                    )
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Tool exception, retrying",
                        tool=tool_name,
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
        
        # 所有重试都失败
        return f"工具执行失败（已重试 {self.max_retries} 次）: {last_error}", True
    
    # ====================== 批量并发执行 ======================
    
    async def execute_many(
        self,
        tool_calls: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[ToolExecutionResult]:
        """
        并发执行多个工具，保持结果顺序与输入一致
        
        参数:
            tool_calls: 工具调用列表
            timeout: 单个工具的超时时间（秒）
        
        返回:
            ToolExecutionResult 列表，顺序与输入一致
        """
        if not tool_calls:
            return []
        
        logger.info(
            "Executing tools concurrently",
            count=len(tool_calls),
            max_concurrent=self.max_concurrent
        )
        
        # 预分配结果数组
        results: List[Optional[ToolExecutionResult]] = [None] * len(tool_calls)
        
        # ✅ 使用 anyio.Semaphore（同步创建）
        from anyio import Semaphore
        semaphore = Semaphore(self.max_concurrent)
        
        async def _run_one(idx: int, call: Dict[str, Any]):
            """执行单个工具（在信号量控制下）"""
            # ✅ 在任务内部使用信号量
            async with semaphore:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})
                call_id = call.get("id", "")
                
                if not tool_name:
                    results[idx] = ToolExecutionResult(
                        tool_name="unknown",
                        call_id=call_id,
                        result="缺少工具名称",
                        is_error=True,
                    )
                    return
                
                try:
                    result = await self.execute_with_result(
                        tool_name,
                        arguments,
                        timeout,
                        call_id
                    )
                    results[idx] = result
                except Exception as e:
                    # 捕获所有异常，确保不影响其他任务
                    logger.error(
                        "Tool execution failed in batch",
                        tool=tool_name,
                        index=idx,
                        error=str(e)
                    )
                    results[idx] = ToolExecutionResult(
                        tool_name=tool_name,
                        call_id=call_id,
                        result=f"执行异常: {str(e)}",
                        is_error=True,
                    )
        
        # ✅ 并发执行
        async with create_task_group() as tg:
            for idx, call in enumerate(tool_calls):
                tg.start_soon(_run_one, idx, call)
        
        # 确保所有结果都已填充
        final_results = []
        for idx, result in enumerate(results):
            if result is None:
                # 理论上不应该发生，但作为保险
                logger.error("Missing result in batch execution", index=idx)
                final_results.append(ToolExecutionResult(
                    tool_name="unknown",
                    call_id="",
                    result="结果缺失",
                    is_error=True,
                ))
            else:
                final_results.append(result)
        
        logger.info(
            "Batch execution completed",
            total=len(final_results),
            successful=sum(1 for r in final_results if not r.is_error),
            failed=sum(1 for r in final_results if r.is_error)
        )
        
        return final_results
    
    # ====================== 统计信息 ======================
    
    def _update_stats(self, tool_name: str, duration: float, is_error: bool = False):
        """
        更新统计数据（线程安全）
        """
        with self._stats_lock:
            self._execution_count[tool_name] += 1
            self._total_time[tool_name] += duration
            if is_error:
                self._error_count[tool_name] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有工具的统计信息
        """
        with self._stats_lock:
            stats = {}
            
            for tool_name in self._tools:
                exec_count = self._execution_count.get(tool_name, 0)
                error_count = self._error_count.get(tool_name, 0)
                total_time = self._total_time.get(tool_name, 0.0)
                
                stats[tool_name] = {
                    "executions": exec_count,
                    "errors": error_count,
                    "total_time": total_time,
                    "avg_time": total_time / exec_count if exec_count > 0 else 0.0,
                    "success_rate": (exec_count - error_count) / exec_count if exec_count > 0 else 1.0,
                }
            
            return stats
    
    def print_stats(self):
        """
        打印统计信息到控制台（格式化输出）
        """
        stats = self.get_stats()
        
        if not stats:
            print("\n=== ToolBox Statistics ===")
            print("No tools registered or executed.")
            return
        
        print("\n" + "=" * 60)
        print("ToolBox Statistics".center(60))
        print("=" * 60)
        
        for tool_name, data in sorted(stats.items()):
            print(f"\n📦 {tool_name}")
            print(f"   Executions:   {data['executions']}")
            print(f"   Errors:       {data['errors']}")
            print(f"   Success Rate: {data['success_rate']:.1%}")
            print(f"   Avg Time:     {data['avg_time']:.3f}s")
            print(f"   Total Time:   {data['total_time']:.3f}s")
        
        print("\n" + "=" * 60 + "\n")
    
    def reset_stats(self):
        """
        重置所有统计数据
        """
        with self._stats_lock:
            for tool_name in self._tools:
                self._execution_count[tool_name] = 0
                self._error_count[tool_name] = 0
                self._total_time[tool_name] = 0.0
        
        logger.info("Statistics reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取工具箱的全局摘要
        """
        stats = self.get_stats()
        
        total_executions = sum(s["executions"] for s in stats.values())
        total_errors = sum(s["errors"] for s in stats.values())
        total_time = sum(s["total_time"] for s in stats.values())
        
        return {
            "tool_count": len(self._tools),
            "total_executions": total_executions,
            "total_errors": total_errors,
            "total_time": total_time,
            "overall_success_rate": (
                (total_executions - total_errors) / total_executions
                if total_executions > 0 else 1.0
            ),
            "avg_time_per_call": (
                total_time / total_executions
                if total_executions > 0 else 0.0
            ),
        }
    
    # ====================== 工具箱信息 ======================
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ToolBox(tools={len(self._tools)}, "
            f"max_concurrent={self.max_concurrent}, "
            f"default_timeout={self.default_timeout}s)"
        )
    
    def __len__(self) -> int:
        """返回工具数量"""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """支持 'tool_name' in toolbox 语法"""
        return tool_name in self._tools