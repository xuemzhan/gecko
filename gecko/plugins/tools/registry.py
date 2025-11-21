from __future__ import annotations  
  
from typing import Dict, List, Optional, Type  
  
from gecko.plugins.tools.base import BaseTool  
  
  
class ToolRegistry:  
    _tools: Dict[str, BaseTool | Type[BaseTool]] = {}  
  
    @classmethod  
    def register(cls, tool: BaseTool | Type[BaseTool], *, replace: bool = True):  
        instance = tool if isinstance(tool, BaseTool) else tool()  
        if instance.name in cls._tools and not replace:  
            raise ValueError(f"工具 '{instance.name}' 已注册")  
        cls._tools[instance.name] = tool  # 保留原对象（类或实例）  
  
    @classmethod  
    def get(cls, name: str) -> Optional[BaseTool]:  
        tool = cls._tools.get(name)  
        if tool is None:  
            return None  
        if isinstance(tool, type):  
            tool = tool()  
            cls._tools[name] = tool  
        return tool  
  
    @classmethod  
    def list_all(cls) -> List[str]:  
        return list(cls._tools.keys())  
  
  
def tool(cls: Type[BaseTool]):  
    ToolRegistry.register(cls, replace=True)  
    return cls  
