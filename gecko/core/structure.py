# gecko/core/structure.py
"""
结构化输出引擎

负责将 LLM 的文本/工具调用输出解析为 Pydantic 模型。

核心功能：
1. 多策略 JSON 提取（Tool Call、直接 JSON、Markdown、暴力截取）
2. Schema 验证与修复
3. 带反馈的重试机制
4. OpenAI Function Calling Schema 生成

优化点：
1. 改进错误信息收集和报告
2. 更智能的 JSON 提取算法
3. Schema 自动修复
4. 详细的调试日志
5. 可配置的解析策略
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ===== 自定义异常 =====

class StructureParseError(ValueError):
    """
    结构化解析失败异常
    
    属性:
        message: 错误信息
        attempts: 所有尝试的解析策略及其错误
        raw_content: 原始内容
    """
    def __init__(
        self,
        message: str,
        attempts: Optional[List[Dict[str, str]]] = None,
        raw_content: Optional[str] = None
    ):
        super().__init__(message)
        self.attempts = attempts or []
        self.raw_content = raw_content
    
    def get_detailed_error(self) -> str:
        """获取详细错误信息"""
        lines = [f"结构化解析失败: {self.args[0]}"]
        
        if self.attempts:
            lines.append("\n尝试的解析策略:")
            for i, attempt in enumerate(self.attempts, 1):
                strategy = attempt.get("strategy", "unknown")
                error = attempt.get("error", "unknown error")
                lines.append(f"  {i}. {strategy}: {error}")
        
        if self.raw_content:
            preview = self.raw_content[:200].replace("\n", "\\n")
            lines.append(f"\n原始内容预览: {preview}...")
        
        return "\n".join(lines)


# ===== 结构化输出引擎 =====

class StructureEngine:
    """
    结构化输出引擎
    
    提供多种策略将文本解析为 Pydantic 模型。
    
    示例:
        ```python
        from pydantic import BaseModel
        
        class User(BaseModel):
            name: str
            age: int
        
        # 从工具调用解析
        result = await StructureEngine.parse(
            content="",
            model_class=User,
            raw_tool_calls=[{
                "function": {
                    "arguments": '{"name": "Alice", "age": 25}'
                }
            }]
        )
        
        # 从文本解析
        result = await StructureEngine.parse(
            content='{"name": "Bob", "age": 30}',
            model_class=User
        )
        ```
    """
    
    # ===== Schema 生成 =====
    
    @staticmethod
    def to_openai_tool(model: Type[BaseModel]) -> Dict[str, Any]:
        """
        将 Pydantic 模型转换为 OpenAI Function Calling 所需的 schema
        
        参数:
            model: Pydantic 模型类
        
        返回:
            OpenAI tool 定义
        
        示例:
            ```python
            from pydantic import BaseModel, Field
            
            class SearchQuery(BaseModel):
                query: str = Field(description="搜索关键词")
                max_results: int = Field(default=5, description="最大结果数")
            
            tool = StructureEngine.to_openai_tool(SearchQuery)
            # 可用于 OpenAI API 的 tools 参数
            ```
        """
        schema = model.model_json_schema()
        
        # 提取模型名称（移除特殊字符）
        name = re.sub(r"\W+", "_", schema.get("title", "extract_data")).lower()
        
        # 移除 Pydantic 内部字段
        if "title" in schema:
            del schema["title"]
        if "$defs" in schema:
            # 展开 definitions
            schema = StructureEngine._flatten_schema(schema)
        
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", f"Extract {name} data"),
                "parameters": schema,
            },
        }
    
    @staticmethod
    def _flatten_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        展开 schema 中的 $ref 引用
        
        简化版实现，处理常见情况
        """
        defs = schema.pop("$defs", {})
        if not defs:
            return schema
            
        def resolve_ref(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_key = obj["$ref"].split("/")[-1]
                    if ref_key in defs:
                        # 递归解析引用
                        return resolve_ref(defs[ref_key])
                return {k: resolve_ref(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_ref(item) for item in obj]
            return obj

        return resolve_ref(schema)
    
    # ===== 解析方法 =====
    
    @classmethod
    async def parse(
        cls,
        content: str,
        model_class: Type[T],
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
        strict: bool = True,
        auto_fix: bool = True,
    ) -> T:
        """
        解析文本为 Pydantic 模型
        
        参数:
            content: 文本内容
            model_class: 目标 Pydantic 模型类
            raw_tool_calls: 原始工具调用列表（优先使用）
            strict: 是否严格模式（False 时会尝试修复）
            auto_fix: 是否自动修复常见问题
        
        返回:
            模型实例
        
        异常:
            StructureParseError: 解析失败
        """
        attempts = []
        
        # 策略 1: 从 Tool Calls 提取
        if raw_tool_calls:
            for idx, call in enumerate(raw_tool_calls):
                try:
                    result = cls._parse_from_tool_call(call, model_class)
                    logger.info(
                        "Parsed from tool call",
                        model=model_class.__name__,
                        tool_call_index=idx
                    )
                    return result
                except Exception as e:
                    attempts.append({
                        "strategy": f"tool_call_{idx}",
                        "error": str(e)
                    })
                    logger.debug("Tool call parse failed", index=idx, error=str(e))
        
        # 策略 2-5: 从文本提取
        try:
            return cls._extract_json(content, model_class, strict=strict, auto_fix=auto_fix)
        except Exception as e:
            # 收集所有尝试的错误
            if hasattr(e, 'attempts'):
                attempts.extend(e.attempts)
            else:
                attempts.append({
                    "strategy": "text_extraction",
                    "error": str(e)
                })
            
            # 构建详细错误信息
            error_details = "\n".join(
                f"  - {a['strategy']}: {a['error'][:100]}"
                for a in attempts
            )
            
            raise StructureParseError(
                f"无法解析为 {model_class.__name__}。尝试了 {len(attempts)} 种策略:\n{error_details}",
                attempts=attempts,
                raw_content=content
            ) from e
    
    # ===== 内部解析方法 =====
    
    @classmethod
    def _parse_from_tool_call(
        cls,
        call: Dict[str, Any],
        model_class: Type[T]
    ) -> T:
        """从单个工具调用中解析"""
        func = call.get("function", {})
        args = func.get("arguments", "")
        
        # 解析参数
        if isinstance(args, str):
            data = json.loads(args)
        elif isinstance(args, dict):
            data = args
        else:
            raise ValueError(f"Invalid arguments type: {type(args)}")
        
        # 验证并创建模型实例
        return model_class(**data)
    
    @classmethod
    def _extract_json(
        cls,
        text: str,
        model_class: Type[T],
        strict: bool = True,
        auto_fix: bool = True,
    ) -> T:
        """
        从文本中提取 JSON 并解析为模型
        
        尝试多种策略：
        1. 直接解析整个文本
        2. 提取 Markdown 代码块
        3. 暴力括号匹配
        4. 清理并重试
        """
        text = text.strip()
        attempts = []
        
        # 0. 快速失败检查 (Fail-Fast)
        if not text or ('{' not in text and '[' not in text):
            raise StructureParseError(
                "Content does not contain JSON-like structure (missing '{' or '[')",
                raw_content=text
            )

        # 策略 A: 直接解析 (最快)
        try:
            data = json.loads(text)
            return cls.validate(data, model_class)
        except Exception as e:
            attempts.append({"strategy": "direct_json", "error": str(e)})
        
        # 策略 B: Markdown 代码块
        markdown_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
        for match in re.finditer(markdown_pattern, text):
            candidate = match.group(1).strip()
            try:
                data = json.loads(candidate)
                return cls.validate(data, model_class)
            except Exception as e:
                attempts.append({"strategy": "markdown", "error": str(e)})
        
        # 策略 C: 暴力括号匹配 (限制尝试次数)
        json_candidates = cls._extract_braced_json(text)
        for idx, candidate in enumerate(json_candidates):
            try:
                data = json.loads(candidate)
                return cls.validate(data, model_class)
            except Exception as e:
                if idx < 3: # 仅记录前3次
                    attempts.append({"strategy": f"braced_{idx}", "error": str(e)})
        
        # 策略 D: 清理并重试
        if auto_fix:
            cleaned = cls._clean_json_string(text)
            if cleaned != text:
                try:
                    data = json.loads(cleaned)
                    logger.info("Parsed after cleaning", model=model_class.__name__)
                    return cls.validate(data, model_class)
                except Exception as e:
                    attempts.append({"strategy": "cleaned_json", "error": str(e)})
        
        # 构建错误详情
        error_details = "\n".join(f"  - {a['strategy']}: {a['error'][:100]}" for a in attempts)
        raise StructureParseError(
            f"无法解析为 {model_class.__name__}。尝试了 {len(attempts)} 种策略:\n{error_details}",
            attempts=attempts,
            raw_content=text
        )
    
    @staticmethod
    def _extract_braced_json(text: str) -> List[str]:
        """
        使用栈提取所有 {...} 块
        
        改进：
        - 返回所有可能的 JSON 对象，不仅仅是第一个
        - 按长度排序，优先尝试最长的
        """
        candidates = []
        stack = []
        start = None
        
        # 简单优化：只在看起来像 JSON 的区域搜索
        search_start = text.find('{')
        if search_start == -1:
            return []
            
        for idx, ch in enumerate(text[search_start:], start=search_start):
            if ch == "{":
                if not stack:
                    start = idx
                stack.append(ch)
            elif ch == "}" and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:idx + 1])
                    start = None
                    # 限制最大候选数量，防止DoS
                    if len(candidates) >= 5:
                        break
        
        # 按长度降序排序
        candidates.sort(key=len, reverse=True)
        return candidates
    
    @staticmethod
    def _clean_json_string(text: str) -> str:
        """
        清理常见的 JSON 格式问题
        
        处理：
        - 单引号 -> 双引号
        - 尾部逗号
        - 注释
        - 控制字符
        """
        # 移除注释
        text = re.sub(r'//.*?\n', '\n', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # 尝试修复单引号（简单情况）
        # 注意：这可能会误伤字符串内容，需谨慎
        # text = text.replace("'", '"')
        
        # 移除尾部逗号
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    # ===== 辅助方法 =====
    
    @classmethod
    def validate(cls, data: Dict[str, Any], model_class: Type[T]) -> T:
        """
        验证数据并创建模型实例
        
        参数:
            data: 数据字典
            model_class: 模型类
        
        返回:
            模型实例
        
        异常:
            ValidationError: 验证失败
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            logger.error(
                "Model validation failed",
                model=model_class.__name__,
                errors=e.errors()
            )
            raise
    
    @classmethod
    def get_schema_diff(
        cls,
        data: Dict[str, Any],
        model_class: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        比较数据与 Schema 的差异
        
        参数:
            data: 实际数据
            model_class: 期望的模型
        
        返回:
            差异信息
        """
        schema = model_class.model_json_schema()
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        
        data_keys = set(data.keys())
        schema_keys = set(properties.keys())
        
        return {
            "missing_required": list(required - data_keys),
            "extra_fields": list(data_keys - schema_keys),
            "type_mismatches": cls._check_type_mismatches(data, properties),
        }
    
    @staticmethod
    def _check_type_mismatches(
        data: Dict[str, Any],
        properties: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """检查类型不匹配"""
        mismatches = []
        
        for key, value in data.items():
            if key not in properties:
                continue
            
            expected_type = properties[key].get("type")
            actual_type = type(value).__name__
            
            # 简化的类型检查
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            
            expected_python_type = type_map.get(expected_type)
            if expected_python_type and not isinstance(value, expected_python_type):
                mismatches.append({
                    "field": key,
                    "expected": expected_type,
                    "actual": actual_type,
                })
        
        return mismatches


# ===== 工具函数 =====

def parse_structured_output(
    content: str,
    model_class: Type[T],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> T:
    """
    同步版本的结构化输出解析（便捷函数）
    
    参数:
        content: 文本内容
        model_class: 目标模型类
        tool_calls: 工具调用列表
    
    返回:
        模型实例
    """
    import asyncio
    
    # 创建事件循环执行异步解析
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        StructureEngine.parse(content, model_class, tool_calls)
    )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取第一个有效的 JSON 对象
    
    参数:
        text: 文本内容
    
    返回:
        JSON 字典，如果未找到返回 None
    """
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 尝试 Markdown
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    for match in re.finditer(pattern, text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # 尝试括号匹配
    candidates = StructureEngine._extract_braced_json(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    return None