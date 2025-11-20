# gecko/core/structure.py
import json
import re
from typing import List, Type, TypeVar, Any, Dict, Optional, Literal
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Strategy = Literal["auto", "function_calling", "json_mode"]

class StructureEngine:
    """
    结构化输出引擎
    负责将非结构化的 LLM 输出可靠地转换为 Pydantic 对象
    """
    
    @staticmethod
    def to_openai_tool(model: Type[BaseModel]) -> Dict[str, Any]:
        """将 Pydantic 模型转换为 OpenAI Tool Schema"""
        schema = model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": schema.get("title", "extract_data"),
                "description": schema.get("description", "Extract structured data"),
                "parameters": schema
            }
        }

    @classmethod
    async def parse(
        cls, 
        content: str, 
        model_class: Type[T], 
        raw_tool_calls: Optional[List[Dict]] = None
    ) -> T:
        """
        统一解析入口
        优先级：Tool Calls -> JSON String -> Markdown JSON
        """
        # 1. 优先尝试 Tool Call (Function Calling)
        if raw_tool_calls:
            for call in raw_tool_calls:
                # 假设 tool call 的 arguments 就是我们想要的数据
                try:
                    args = call["function"]["arguments"]
                    data = json.loads(args) if isinstance(args, str) else args
                    return model_class(**data)
                except (json.JSONDecodeError, ValidationError):
                    continue # 尝试下一个

        # 2. 尝试解析内容中的 JSON
        try:
            return cls._extract_json(content, model_class)
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {str(e)}")

    @staticmethod
    def _extract_json(text: str, model_class: Type[T]) -> T:
        text = text.strip()
        errors = []

        # 策略 A: 纯 JSON
        try:
            data = json.loads(text)
            return model_class(**data)
        except Exception as e:
            errors.append(f"Raw JSON failed: {e}")
            
        # 策略 B: Markdown Code Block
        # [优化] 正则支持 ```json 和 纯 ```
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return model_class(**data)
            except Exception as e:
                errors.append(f"Markdown JSON failed: {e}")
                
        # 策略 C: 暴力查找最外层 {}
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                data = json.loads(json_str)
                return model_class(**data)
        except Exception as e:
             errors.append(f"Brute force JSON failed: {e}")

        raise ValueError(f"No valid JSON found. Errors: {'; '.join(errors)}")