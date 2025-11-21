# gecko/core/toolbox.py  
"""  
ToolBox（优化版）  
  
改进点：  
1. 使用 anyio.Semaphore + create_task_group 控制并发，结果顺序与输入一致  
2. 使用 anyio.fail_after 实现工具级超时，直接捕获内置 TimeoutError，兼容所有 anyio 版本  
3. 工具执行日志更详细，统计信息完善  
4. 工具返回值统一转换为字符串，调用方无需关心类型  
"""  
  
from __future__ import annotations  
  
import time  
from typing import Any, Callable, Dict, List, Optional  
  
from anyio import create_task_group, fail_after, Semaphore  
  
from gecko.config import settings  
from gecko.core.exceptions import ToolError, ToolNotFoundError, ToolTimeoutError  
from gecko.core.logging import get_logger  
from gecko.plugins.tools.base import BaseTool  
  
logger = get_logger(__name__)  
  
  
class ToolBox:  
    """  
    Agent 工具箱  
    负责注册工具、执行单个或多个工具调用，并在内部维护并发/超时/统计逻辑  
    """  
  
    def __init__(  
        self,  
        tools: Optional[List[BaseTool]] = None,  
        max_concurrent: int = 5,  
        default_timeout: float | None = None,  
    ):  
        self._tools: Dict[str, BaseTool] = {}  
        self.max_concurrent = max_concurrent  
        self.default_timeout = default_timeout or settings.tool_execution_timeout  
  
        # 统计数据  
        self._execution_count: Dict[str, int] = {}  
        self._error_count: Dict[str, int] = {}  
        self._total_time: Dict[str, float] = {}  
  
        if tools:  
            for tool in tools:  
                self.register(tool)  
  
    # ====================== 工具管理 ======================  
    def register(self, tool: BaseTool, replace: bool = True):  
        if tool.name in self._tools and not replace:  
            raise ValueError(f"工具 '{tool.name}' 已注册，如需覆盖请设置 replace=True")  
        if tool.name in self._tools:  
            logger.warning("Tool already registered, will be replaced", tool=tool.name)  
  
        self._tools[tool.name] = tool  
        self._execution_count[tool.name] = 0  
        self._error_count[tool.name] = 0  
        self._total_time[tool.name] = 0.0  
        logger.debug("Tool registered", tool=tool.name)  
  
    def unregister(self, tool_name: str):  
        if tool_name in self._tools:  
            del self._tools[tool_name]  
            logger.debug("Tool unregistered", tool=tool_name)  
  
    def get(self, name: str) -> Optional[BaseTool]:  
        return self._tools.get(name)  
  
    def list_tools(self) -> List[BaseTool]:  
        return list(self._tools.values())  
  
    def has_tool(self, name: str) -> bool:  
        return name in self._tools  
  
    # ====================== OpenAI Schema ======================  
    def to_openai_schema(self) -> List[Dict[str, Any]]:  
        """  
        生成 OpenAI Function Calling 所需的 schema  
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
        timeout: Optional[float] = None  
    ) -> str:  
        """  
        以异步方式执行单个工具，具备超时与异常处理  
        """  
        tool = self.get(name)  
        if not tool:  
            raise ToolNotFoundError(name)  
  
        actual_timeout = timeout or self.default_timeout  
        start_time = time.time()  
        logger.debug("Executing tool", tool=name, timeout=actual_timeout)  
  
        try:  
            with fail_after(actual_timeout):  
                result = await tool.execute(arguments)  
        except TimeoutError:  
            self._error_count[name] += 1  
            logger.error("Tool execution timeout", tool=name, timeout=actual_timeout)  
            raise ToolTimeoutError(name, actual_timeout)  
        except Exception as e:  
            self._error_count[name] += 1  
            logger.exception("Tool execution failed", tool=name)  
            raise ToolError(  
                f"Tool '{name}' execution failed: {e}",  
                context={"tool": name, "arguments": arguments}  
            ) from e  
        else:  
            duration = time.time() - start_time  
            self._execution_count[name] += 1  
            self._total_time[name] += duration  
            logger.info("Tool executed successfully", tool=name, duration=f"{duration:.3f}s")  
  
            return result if isinstance(result, str) else str(result)  
  
    # ====================== 并发执行 ======================  
    async def execute_many(  
        self,  
        tool_calls: List[Dict[str, Any]],  
        timeout: Optional[float] = None  
    ) -> List[Dict[str, Any]]:  
        """  
        并发执行多个工具，并保持结果顺序与输入一致  
        """  
        if not tool_calls:  
            return []  
  
        logger.info("Executing tools concurrently", count=len(tool_calls), max_concurrent=self.max_concurrent)  
  
        results: List[Dict[str, Any]] = [None] * len(tool_calls)  
        semaphore = Semaphore(self.max_concurrent)  
  
        async def _run_one(idx: int, call: Dict[str, Any]):  
            tool_name = call.get("name")  
            arguments = call.get("arguments", {})  
            call_id = call.get("id", "")  
  
            try:  
                output = await self.execute(tool_name, arguments, timeout)  
                results[idx] = {  
                    "id": call_id,  
                    "name": tool_name,  
                    "result": output,  
                    "is_error": False,  
                }  
            except Exception as e:  
                results[idx] = {  
                    "id": call_id,  
                    "name": tool_name,  
                    "result": str(e),  
                    "is_error": True,  
                }  
  
        async with create_task_group() as tg:  
            for idx, call in enumerate(tool_calls):  
                await semaphore.acquire()  
                tg.start_soon(self._execute_with_sem, semaphore, _run_one, idx, call)  
  
        logger.info("All tools executed", total=len(tool_calls))  
        return results  
  
    async def _execute_with_sem(self, sem: Semaphore, fn: Callable, *args):  
        """  
        辅助方法：在信号量控制下执行函数，确保并发限制生效  
        """  
        try:  
            await fn(*args)  
        finally:  
            sem.release()  
  
    # ====================== 统计信息 ======================  
    def get_stats(self) -> Dict[str, Any]:  
        stats = {}  
        for name in self._tools:  
            exec_count = self._execution_count.get(name, 0)  
            error_count = self._error_count.get(name, 0)  
            total_time = self._total_time.get(name, 0.0)  
  
            stats[name] = {  
                "executions": exec_count,  
                "errors": error_count,  
                "total_time": total_time,  
                "avg_time": total_time / exec_count if exec_count else 0.0,  
                "success_rate": (exec_count - error_count) / exec_count if exec_count else 1.0,  
            }  
        return stats  
  
    def print_stats(self):  
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
        for name in self._tools:  
            self._execution_count[name] = 0  
            self._error_count[name] = 0  
            self._total_time[name] = 0.0  
