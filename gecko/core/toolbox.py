# gecko/core/toolbox.py
"""
ToolBox - Agent 工具箱

核心功能：
1. 工具注册与管理（支持实例注入与注册表加载）
2. 单个/批量工具执行
3. 并发控制与超时管理
4. 执行统计与监控
5. OpenAI Function Calling Schema 生成

优化日志：
- [Refactor] 集成 ToolRegistry，支持通过字符串名称加载工具
- [Refactor] 适配新版 BaseTool 接口
- [Fix] 修复并发控制的信号量使用方式
- [Feat] 线程安全的统计数据
- [Fix] 补全 get_summary, reset_stats 及魔术方法
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from anyio import create_task_group, fail_after, Semaphore
from anyio import get_cancelled_exc_class

from gecko.config import settings
from gecko.core.exceptions import ToolError, ToolNotFoundError
from gecko.core.logging import get_logger
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import ToolRegistry

logger = get_logger(__name__)


# ===== 返回值模型 =====

@dataclass
class ToolExecutionResult:
    """
    工具执行结果（ToolBox 层面的封装）
    
    包含工具本身的返回内容，以及 ToolBox 记录的执行元数据（耗时、ID等）。
    
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    既支持直接传入 BaseTool 实例，也支持通过名称从 ToolRegistry 加载。
    
    示例:
        ```python
        # 混合加载工具
        toolbox = ToolBox(
            tools=["calculator", MyCustomTool()],
            max_concurrent=5
        )
        
        # 执行
        result = await toolbox.execute("calculator", {"expression": "1+1"})
        
        # 获取统计
        summary = toolbox.get_summary()
        print(f"Success Rate: {summary['overall_success_rate']:.2%}")
        ```
    """

    def __init__(
        self,
        tools: Optional[List[Union[BaseTool, str]]] = None,
        max_concurrent: int = 5,
        default_timeout: Optional[float] = None,
        enable_retry: bool = False,
        max_retries: int = 2,
    ):
        """
        初始化工具箱
        
        参数:
            tools: 初始工具列表（支持 BaseTool 实例或注册表中的字符串名称）
            max_concurrent: 最大并发执行数
            default_timeout: 默认超时时间（秒）
            enable_retry: 是否启用重试
            max_retries: 最大重试次数
        """
        # 工具存储
        self._tools: Dict[str, BaseTool] = {}
        
        # 配置
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
            for item in tools:
                self.add_tool(item)
    
    # ====================== 工具管理 ======================
    
    def add_tool(self, item: Union[BaseTool, str], **kwargs) -> "ToolBox":
        """
        添加工具（高层接口）
        
        支持：
        1. 字符串：从 ToolRegistry 加载
        2. 实例：直接注册
        
        参数:
            item: 工具名称或实例
            **kwargs: 如果是字符串加载，kwargs 将传递给工具构造函数
        """
        tool_instance: Optional[BaseTool] = None
        
        if isinstance(item, str):
            try:
                tool_instance = ToolRegistry.load_tool(item, **kwargs)
            except Exception as e:
                logger.error(f"Failed to load tool '{item}' from registry: {e}")
                # 此时可以选择抛出异常，或者仅记录错误跳过，这里选择抛出以便尽早发现配置错误
                raise ToolNotFoundError(f"Registry load failed for '{item}': {e}") from e
        elif isinstance(item, BaseTool):
            tool_instance = item
        else:
            raise TypeError(f"Tool must be BaseTool or str, got {type(item)}")
            
        if tool_instance:
            self.register(tool_instance)
            
        return self

    def register(self, tool: BaseTool, replace: bool = True) -> "ToolBox":
        """
        注册工具实例（底层接口）
        
        参数:
            tool: 工具实例
            replace: 是否替换同名工具
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Tool must inherit from BaseTool, got {type(tool)}")
        
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        
        self._tools[tool.name] = tool
        
        # 初始化统计
        with self._stats_lock:
            if tool.name not in self._execution_count:
                self._execution_count[tool.name] = 0
                self._error_count[tool.name] = 0
                self._total_time[tool.name] = 0.0
        
        logger.debug("Tool registered", tool_name=tool.name)
        return self
    
    def unregister(self, tool_name: str) -> "ToolBox":
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info("Tool unregistered", tool_name=tool_name)
        return self
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """获取所有已注册工具"""
        return list(self._tools.values())
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    # ====================== Schema 生成 ======================
    
    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """
        生成 OpenAI Function Calling Schema
        
        直接调用 BaseTool.openai_schema 属性
        """
        return [t.openai_schema for t in self._tools.values()]
    
    # ====================== 执行逻辑 ======================
    
    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        call_id: str = "",
    ) -> str:
        """
        执行单个工具（简易版）
        
        返回:
            结果字符串
        """
        result = await self.execute_with_result(name, arguments, timeout, call_id)
        if result.is_error:
            raise ToolError(
                f"Tool execution failed: {result.result}",
                context={"tool": name, "args": arguments}
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
        执行单个工具（完整版）
        
        包含：超时控制、重试逻辑、统计记录
        """
        tool = self.get(name)
        if not tool:
            self._update_stats(name, 0, is_error=True)
            raise ToolNotFoundError(name)
        
        actual_timeout = timeout or self.default_timeout
        start_time = time.time()
        
        # 选择执行策略
        if self.enable_retry:
            result_str, is_error = await self._execute_with_retry(
                tool, arguments, actual_timeout
            )
        else:
            result_str, is_error = await self._execute_once(
                tool, arguments, actual_timeout
            )
            
        duration = time.time() - start_time
        self._update_stats(name, duration, is_error)
        
        return ToolExecutionResult(
            tool_name=name,
            call_id=call_id,
            result=result_str,
            is_error=is_error,
            duration=duration
        )

    async def _execute_once(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        timeout: float
    ) -> tuple[str, bool]:
        """
        单次执行封装
        
        处理超时和异常，适配 BaseTool 的 ToolResult 返回值
        """
        try:
            with fail_after(timeout):
                # BaseTool.execute 已经处理了参数校验和内部异常，返回 ToolResult
                res: ToolResult = await tool.execute(arguments)
                return res.content, res.is_error
                
        except TimeoutError:
            return f"Execution timed out after {timeout}s", True
        except get_cancelled_exc_class():
            return "Execution cancelled", True
        except Exception as e:
            logger.exception("Unexpected tool execution error", tool=tool.name)
            return f"System error: {str(e)}", True

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        timeout: float
    ) -> tuple[str, bool]:
        """带重试的执行逻辑"""
        last_result = ""
        
        for attempt in range(self.max_retries + 1):
            content, is_error = await self._execute_once(tool, arguments, timeout)
            
            if not is_error:
                return content, False
            
            last_result = content
            
            # 如果是超时或系统错误，尝试重试
            # 如果是 BaseTool 返回的业务逻辑错误（如参数不对），通常重试无用，但在通用层我们还是给机会
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                logger.warning(
                    "Tool failed, retrying",
                    tool=tool.name,
                    attempt=attempt + 1,
                    error=content[:100]
                )
                await asyncio.sleep(wait_time)
        
        return last_result, True

    # ====================== 批量执行 ======================
    
    async def execute_many(
        self,
        tool_calls: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[ToolExecutionResult]:
        """
        并发批量执行
        
        参数:
            tool_calls: [{"name": "...", "arguments": {...}, "id": "..."}, ...]
            
        返回:
            结果列表（顺序与输入一致）
        """
        if not tool_calls:
            return []
            
        results: List[Optional[ToolExecutionResult]] = [None] * len(tool_calls)
        
        # 使用 anyio 信号量控制并发
        semaphore = Semaphore(self.max_concurrent)
        
        async def _worker(idx: int, call: Dict[str, Any]):
            async with semaphore:
                name = call.get("name", "")
                args = call.get("arguments", {})
                cid = call.get("id", "")
                
                if not name:
                    results[idx] = ToolExecutionResult(
                        tool_name="unknown", call_id=cid, result="Missing tool name", is_error=True
                    )
                    return

                try:
                    # 复用 execute_with_result 以获得完整的统计和重试支持
                    results[idx] = await self.execute_with_result(name, args, timeout, cid)
                except Exception as e:
                    results[idx] = ToolExecutionResult(
                        tool_name=name, call_id=cid, result=f"Batch error: {e}", is_error=True
                    )

        async with create_task_group() as tg:
            for i, call in enumerate(tool_calls):
                tg.start_soon(_worker, i, call)
                
        # 过滤 None (理论上不应该存在)
        return [r for r in results if r is not None]

    # ====================== 统计与辅助 ======================
    
    def _update_stats(self, name: str, duration: float, is_error: bool):
        """更新统计数据（线程安全）"""
        with self._stats_lock:
            self._execution_count[name] += 1
            self._total_time[name] += duration
            if is_error:
                self._error_count[name] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取详细统计快照"""
        with self._stats_lock:
            stats = {}
            for name in self._tools:
                cnt = self._execution_count[name]
                err = self._error_count[name]
                total = self._total_time[name]
                stats[name] = {
                    "calls": cnt,
                    "errors": err,
                    "avg_time": (total / cnt) if cnt > 0 else 0.0,
                    "success_rate": ((cnt - err) / cnt) if cnt > 0 else 1.0
                }
            return stats
            
    def print_stats(self):
        """打印格式化统计信息"""
        stats = self.get_stats()
        print("\n=== ToolBox Statistics ===")
        if not stats:
            print("No tools executed.")
        for name, data in stats.items():
            print(f"🔧 {name:<15} Calls: {data['calls']:<5} Errors: {data['errors']:<5} "
                  f"Avg: {data['avg_time']:.3f}s Rate: {data['success_rate']:.1%}")
        print("=" * 30 + "\n")
        
    def reset_stats(self):
        """
        重置所有统计数据
        """
        with self._stats_lock:
            self._execution_count.clear()
            self._error_count.clear()
            self._total_time.clear()
        logger.info("Statistics reset")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取工具箱的全局摘要
        """
        stats = self.get_stats()
        
        total_executions = sum(s["calls"] for s in stats.values())
        total_errors = sum(s["errors"] for s in stats.values())
        total_time_sum = sum(
            self._total_time.get(name, 0.0) for name in stats.keys()
        )
        
        return {
            "tool_count": len(self._tools),
            "total_executions": total_executions,
            "total_errors": total_errors,
            "total_time": total_time_sum,
            "overall_success_rate": (
                (total_executions - total_errors) / total_executions
                if total_executions > 0 else 1.0
            ),
            "avg_time_per_call": (
                total_time_sum / total_executions
                if total_executions > 0 else 0.0
            ),
        }

    # ====================== 魔术方法 ======================

    def __repr__(self) -> str:
        return f"ToolBox(tools={len(self._tools)}, concurrent={self.max_concurrent})"
    
    def __len__(self) -> int:
        """返回已注册工具的数量"""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """支持 'tool_name' in toolbox 语法"""
        return tool_name in self._tools