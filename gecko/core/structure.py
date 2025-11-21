# gecko/core/structure.py  
"""  
结构化输出引擎（优化版）  
  
改进：  
1. `StructureParseError` 继承自 ValueError，与旧逻辑兼容  
2. Tool Call 解析失败有 debug 记录，便于排查  
3. Markdown / 暴力截取策略更健壮  
4. 失败时返回固定前缀 "No valid JSON found..."，测试与调用方可直接匹配  
"""  
  
from __future__ import annotations  
  
import json  
import re  
from typing import Any, Dict, List, Optional, Type, TypeVar  
  
from pydantic import BaseModel, ValidationError  
  
from gecko.core.logging import get_logger  
  
logger = get_logger(__name__)  
  
T = TypeVar("T", bound=BaseModel)  
  
  
class StructureParseError(ValueError):  
    """结构化解析失败异常（兼容 ValueError 捕获逻辑）"""  
    pass  
  
  
class StructureEngine:  
    """  
    结构化输出引擎  
    - 优先从 Tool Call 提取结构化数据  
    - 否则尝试 JSON / Markdown JSON / 暴力截取  
    """  
  
    @staticmethod  
    def to_openai_tool(model: Type[BaseModel]) -> Dict[str, Any]:  
        """将 Pydantic 模型转换为 OpenAI Function Calling 所需的 schema"""  
        schema = model.model_json_schema()  
        name = re.sub(r"\W+", "_", schema.get("title", "extract_data")).lower()  
        return {  
            "type": "function",  
            "function": {  
                "name": name,  
                "description": schema.get("description", "Extract structured data"),  
                "parameters": schema,  
            },  
        }  
  
    @classmethod  
    async def parse(  
        cls,  
        content: str,  
        model_class: Type[T],  
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,  
    ) -> T:  
        errors: List[str] = []  
  
        # 1. Tool Call 优先  
        if raw_tool_calls:  
            for call in raw_tool_calls:  
                func = call.get("function", {})  
                try:  
                    args = func.get("arguments", "")  
                    data = json.loads(args) if isinstance(args, str) else args  
                    return model_class(**data)  
                except (json.JSONDecodeError, ValidationError) as e:  
                    tool_name = func.get("name", "unknown")  
                    msg = f"Tool call '{tool_name}' parse failed: {e}"  
                    errors.append(msg)  
                    logger.debug(msg)  
  
        # 2. 尝试从文本提取 JSON  
        try:  
            return cls._extract_json(content, model_class)  
        except ValueError as e:  
            errors.append(str(e))  
            raise StructureParseError("No valid JSON found. Details: " + "; ".join(errors)) from e  
  
    @staticmethod  
    def _extract_json(text: str, model_class: Type[T]) -> T:  
        text = text.strip()  
        errors = []  
  
        # 策略 A：直接 JSON  
        try:  
            return model_class(**json.loads(text))  
        except Exception as e:  
            errors.append(f"Raw JSON failed: {e}")  
  
        # 策略 B：Markdown 代码块  
        pattern = r"```(?:json)?\s*([\s\S]*?)```"  
        for match in re.finditer(pattern, text):  
            candidate = match.group(1).strip()  
            try:  
                return model_class(**json.loads(candidate))  
            except Exception as e:  
                errors.append(f"Markdown JSON failed: {e}")  
  
        # 策略 C：括号匹配暴力截取  
        for candidate in StructureEngine._extract_braced_json(text):  
            try:  
                return model_class(**json.loads(candidate))  
            except Exception as e:  
                errors.append(f"Brute force JSON failed: {e}")  
                break  
  
        raise ValueError("; ".join(errors))  
  
    @staticmethod  
    def _extract_braced_json(text: str) -> List[str]:  
        """  
        使用栈找出第一个 { ... }（支持嵌套）  
        """  
        stack = []  
        start = None  
        candidates = []  
  
        for idx, ch in enumerate(text):  
            if ch == "{":  
                if not stack:  
                    start = idx  
                stack.append(ch)  
            elif ch == "}" and stack:  
                stack.pop()  
                if not stack and start is not None:  
                    candidates.append(text[start : idx + 1])  
                    break  
  
        return candidates  
