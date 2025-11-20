# gecko/core/toolbox.py
from typing import List, Dict, Any, Optional
from gecko.plugins.tools.base import BaseTool, ToolProtocol

class ToolBox:
    """
    Agent 实例专属的工具箱
    """
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool):
        """注册工具实例"""
        if tool.name in self._tools:
            # 允许覆盖，或者抛错，这里选择覆盖以便动态更新
            pass
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

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

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in toolbox")
        
        # 未来在这里插入生命周期检查
        # if not tool.is_ready: await tool.startup()
        
        return await tool.execute(arguments)