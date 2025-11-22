# gecko/plugins/tools/base.py
"""
工具基类定义

核心功能：
1. 定义统一的工具接口 BaseTool
2. 强制参数校验 (Pydantic)
3. 自动生成 OpenAI Function Schema
4. 统一的错误处理与结果封装
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

from pydantic import BaseModel, Field, ValidationError

from gecko.core.utils import ensure_awaitable


class ToolResult(BaseModel):
    """工具执行结果封装"""
    content: str = Field(..., description="工具执行的文本输出")
    is_error: bool = Field(default=False, description="是否执行出错")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外的元数据")


class BaseTool(BaseModel, ABC):
    """
    工具抽象基类
    
    所有自定义工具必须继承此类，并提供 args_schema 用于参数校验。
    
    示例:
        ```python
        class MyArgs(BaseModel):
            query: str
            
        class MyTool(BaseTool):
            name = "my_tool"
            description = "My awesome tool"
            args_schema = MyArgs
            
            async def _run(self, args: MyArgs) -> ToolResult:
                return ToolResult(content=f"Echo: {args.query}")
        ```
    """
    name: str = Field(..., description="工具唯一标识名称 (e.g., 'calculator')")
    description: str = Field(..., description="工具功能描述，用于 LLM 决策")
    
    # 排除 args_schema 不参与 BaseTool 本身的序列化，仅用于元编程
    args_schema: Type[BaseModel] = Field(..., exclude=True, description="参数定义的 Pydantic 模型类")

    @property
    def openai_schema(self) -> Dict[str, Any]:
        """
        自动生成 OpenAI Function Calling Schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            }
        }
    
    # 兼容旧代码的属性
    @property
    def parameters(self) -> Dict[str, Any]:
        return self.args_schema.model_json_schema()

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """
        执行工具（模板方法）
        
        负责：
        1. 参数校验
        2. 异常捕获
        3. 调用具体的 _run 实现
        """
        try:
            # 1. Pydantic 校验
            validated_args = self.args_schema(**arguments)
        except ValidationError as e:
            return ToolResult(
                content=f"参数校验错误: {str(e)}",
                is_error=True
            )
        except Exception as e:
            return ToolResult(
                content=f"参数解析错误: {str(e)}",
                is_error=True
            )

        try:
            # 2. 执行业务逻辑 (支持同步/异步)
            result = await ensure_awaitable(self._run, validated_args)
            
            # 3. 结果标准化
            if isinstance(result, ToolResult):
                return result
            return ToolResult(content=str(result))
            
        except Exception as e:
            return ToolResult(
                content=f"工具执行内部错误: {str(e)}",
                is_error=True
            )

    @abstractmethod
    async def _run(self, args: BaseModel) -> Union[ToolResult, str]:
        """
        工具的具体实现逻辑
        
        参数:
            args: 已校验的 Pydantic 对象
        """
        pass