# gecko/plugins/tools/standard/calculator.py
"""
安全计算器工具

基于 AST 解析的安全数学表达式计算，防止代码注入。
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Type, Union, Dict

from pydantic import BaseModel, Field

from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import register_tool


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ..., 
        description="数学表达式，支持加减乘除、幂运算及常用数学函数 (sqrt, log, sin, etc.)。例如: '2 + 2 * sqrt(4)'"
    )


@register_tool("calculator")
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "用于执行精确的数学计算。"
    args_schema: Type[BaseModel] = CalculatorArgs

    async def _run(self, args: CalculatorArgs) -> ToolResult: # type: ignore
        expr = args.expression.strip()
        
        # 长度限制防止 DoS
        if len(expr) > 500:
             return ToolResult(content="错误：表达式过长", is_error=True)

        try:
            result = self._safe_eval(expr)
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=f"计算错误: {str(e)}", is_error=True)

    def _safe_eval(self, expr: str) -> Union[int, float]:
        """
        基于 AST 的安全求值
        仅允许特定的节点类型和函数。
        """
        # 支持的操作符
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.BitXor: operator.xor,
            ast.USub: operator.neg,
        }

        # 支持的函数
        functions = {
            "sqrt": math.sqrt,
            "log": math.log,
            "ln": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "pi": math.pi,
            "e": math.e,
        }

        def _eval(node):
            # 1. 数字
            if isinstance(node, ast.Constant): 
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"不支持的常量类型: {type(node.value)}")

            # 2. 二元运算 (a + b)
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in operators:
                    left = _eval(node.left)
                    right = _eval(node.right)

                    # [新增] 安全防御：限制幂运算的指数大小
                    if op_type == ast.Pow:
                        # 检查指数是否为数字且过大 (例如限制为 1000)
                        if isinstance(right, (int, float)) and right > 1000:
                            raise ValueError(f"指数过大: {right} (最大允许 1000)")

                    return operators[op_type](left, right)
            
                raise ValueError(f"不支持的操作符: {op_type}")

            # 3. 一元运算 (-a)
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in operators:
                    return operators[op_type](_eval(node.operand))
                raise ValueError(f"不支持的一元操作符: {op_type}")

            # 4. 函数调用 (sqrt(4))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in functions:
                        # 递归计算参数
                        args = [_eval(arg) for arg in node.args]
                        return functions[func_name](*args)
                    raise ValueError(f"禁止调用的函数: {func_name}")
                raise ValueError("不支持的复杂函数调用")

            # 5. 变量名 (pi, e)
            elif isinstance(node, ast.Name):
                if node.id in functions:
                    val = functions[node.id]
                    if isinstance(val, (int, float)):
                        return val
                raise ValueError(f"未知变量: {node.id}")
            
            # [优化建议] 显式拦截属性访问和下标访问，给出更明确的提示
            elif isinstance(node, (ast.Attribute, ast.Subscript, ast.List, ast.Dict, ast.Tuple)):
                raise ValueError("出于安全考虑，禁止使用属性访问、下标或复杂数据结构")

            raise ValueError(f"非法表达式结构: {type(node)}")

        # 解析并求值
        tree = ast.parse(expr, mode='eval')
        return _eval(tree.body)