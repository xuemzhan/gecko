from __future__ import annotations  
  
from typing import Any, Dict, Type  
  
from pydantic import BaseModel, Field  
  
from gecko.plugins.tools.base import BaseTool, ToolResult  
from gecko.plugins.tools.registry import tool  
  
  
class CalculatorArgs(BaseModel):  
    expression: str = Field(..., description="数学表达式，例如：(1 + 2) * 3")  
  
  
@tool  
class CalculatorTool(BaseTool):  
    name: str = "calculator"  
    description: str = "执行安全的数学计算，支持加减乘除等"  
    parameters: Dict[str, Any] = CalculatorArgs.model_json_schema()  
    args_model: Type[CalculatorArgs] = CalculatorArgs  
  
    def _execute_impl(self, args: CalculatorArgs) -> ToolResult:  
        expr = args.expression.strip()  
        allowed = set("0123456789.+-*/() ")  
        if not all(c in allowed for c in expr):  
            return ToolResult(content="错误：表达式包含非法字符", is_error=True)  
  
        try:  
            result = eval(expr, {"__builtins__": {}})  
            return ToolResult(content=f"计算结果：{result}")  
        except Exception as e:  
            return ToolResult(content=f"计算错误：{e}", is_error=True)  
