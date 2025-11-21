from __future__ import annotations  
  
from typing import Any, Dict, List, Optional  
  
import anyio  
  
from gecko.plugins.tools.base import ToolResult  
from gecko.plugins.tools.registry import ToolRegistry  
  
  
class ToolExecutor:  
    @staticmethod  
    async def concurrent_execute(  
        tool_calls: List[Dict[str, Any]],  
        *,  
        raise_on_error: bool = False,  
        max_concurrent: int = 5,  
    ) -> List[ToolResult]:  
        results: List[Optional[ToolResult]] = [None] * len(tool_calls)  
  
        async def _run_one(idx: int, call: Dict[str, Any]):  
            tool_name = call.get("name")  
            arguments = call.get("arguments", {})  
            tool = ToolRegistry.get(tool_name)  
            if not tool:  
                res = ToolResult(content=f"工具 {tool_name} 未找到", is_error=True)  
            else:  
                try:  
                    res = await tool.execute(arguments)  
                except Exception as e:  
                    res = ToolResult(content=f"执行失败: {e}", is_error=True)  
            results[idx] = res  
            if raise_on_error and res.is_error:  
                raise RuntimeError(res.content)  
  
        async with anyio.create_task_group() as tg:  
            sem = anyio.Semaphore(max_concurrent)  
            for idx, call in enumerate(tool_calls):  
                await sem.acquire()  
                tg.start_soon(ToolExecutor._run_with_sem, sem, _run_one, idx, call)  
  
        return [r for r in results if r]  
  
    @staticmethod  
    async def _run_with_sem(sem: anyio.Semaphore, fn, *args):  
        try:  
            await fn(*args)  
        finally:  
            sem.release()  
