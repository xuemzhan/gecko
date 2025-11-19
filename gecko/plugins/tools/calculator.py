# gecko/plugins/tools/calculator.py
from gecko.plugins.tools.base import BaseTool
from gecko.plugins.tools.registry import tool

@tool
class CalculatorTool(BaseTool):
    name: str = "calculator"                                    # 必须带类型注解
    description: str = "执行安全的数学计算，支持加减乘除、幂运算等"
    parameters: dict = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如：(123 + 456) * 7) / 8"
            }
        },
        "required": ["expression"]
    }

    async def execute(self, arguments: dict) -> str:
        expression = arguments.get("expression", "").strip()
        if not expression:
            return "错误：表达式为空"

        # 安全白名单
        allowed_chars = set("0123456789.+-*/() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"

        try:
            result = eval(expression, {"__builtins__": {}})
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{str(e)}"