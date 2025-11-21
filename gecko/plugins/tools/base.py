from __future__ import annotations  
  
from typing import Any, Dict, Optional, Type  
  
from pydantic import BaseModel, Field, ValidationError  
  
from gecko.core.utils import ensure_awaitable  
  
  
class ToolResult(BaseModel):  
    content: str  
    is_error: bool = False  
    metadata: Dict[str, Any] = Field(default_factory=dict)  
  
  
class ToolArgsModel(BaseModel):  
    """可选：工具参数模型。"""  
    pass  
  
  
class BaseTool(BaseModel):  
    name: str = Field(..., description="工具唯一名称")  
    description: str = Field(..., description="工具功能描述")  
    parameters: Dict[str, Any] = Field(  
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},  
        description="OpenAI 风格的参数 schema",  
    )  
    args_model: Optional[Type[ToolArgsModel]] = None  # 子类如需校验参数，可设置该字段  
  
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:  
        """  
        通用执行入口：  
        1. 如声明 args_model，则先用 Pydantic 校验  
        2. 调用 _execute_impl（可同步/异步）  
        3. 返回 ToolResult  
        """  
        payload = arguments  
        if self.args_model:  
            try:  
                payload = self.args_model(**arguments)  
            except ValidationError as e:  
                return ToolResult(content=str(e), is_error=True)  
  
        result = await ensure_awaitable(self._execute_impl, payload)  
        if isinstance(result, ToolResult):  
            return result  
        return ToolResult(content=str(result))  
  
    def _execute_impl(self, arguments: Any) -> ToolResult:  
        """子类需要实现具体逻辑，参数为校验后的对象或原始 dict"""  
        raise NotImplementedError("子类必须实现 _execute_impl")  
