# gecko/plugins/tools/executor.py
from __future__ import annotations

import anyio
from typing import List, Dict
from gecko.plugins.tools.base import ToolResponse
from gecko.plugins.tools.registry import ToolRegistry

class ToolExecutor:
    """并行工具执行器"""
    @staticmethod
    async def concurrent_execute(tool_calls: List[Dict[str, Any]]) -> List[ToolResponse]:
        results: List[ToolResponse] = []

        async def _run_one(call: Dict[str, Any]):
            tool_name = call.get("name")
            arguments = call.get("arguments", {})
            tool = ToolRegistry.get(tool_name)
            
            if not tool:
                results.append(ToolResponse(tool_name=tool_name, content=f"工具 {tool_name} 未找到", is_error=True))
                return

            try:
                content = await tool.execute(arguments)
                results.append(ToolResponse(tool_name=tool_name, content=content))
            except Exception as e:
                results.append(ToolResponse(tool_name=tool_name, content=f"执行失败: {str(e)}", is_error=True))

        async with anyio.create_task_group() as tg:
            for call in tool_calls:
                tg.start_soon(_run_one, call)

        return results