from __future__ import annotations  
  
from typing import Any, Dict, Type  
  
from duckduckgo_search import DDGS  
from pydantic import BaseModel, Field  
  
from gecko.plugins.tools.base import BaseTool, ToolResult  
from gecko.plugins.tools.registry import tool  
  
  
class DuckDuckGoArgs(BaseModel):  
    query: str = Field(..., description="搜索关键词")  
  
  
@tool  
class DuckDuckGoSearch(BaseTool):  
    name: str = "duckduckgo_search"  
    description: str = "使用 DuckDuckGo 搜索互联网，返回前 5 条结果（无需 API Key）"  
    parameters: Dict[str, Any] = DuckDuckGoArgs.model_json_schema()  
    args_model: Type[DuckDuckGoArgs] = DuckDuckGoArgs  
  
    async def _execute_impl(self, args: DuckDuckGoArgs) -> ToolResult:  
        query = args.query.strip()  
        if not query:  
            return ToolResult(content="错误：搜索关键词为空", is_error=True)  
  
        try:  
            with DDGS() as ddgs:  
                results = list(ddgs.text(query, max_results=5))  
        except Exception as e:  
            return ToolResult(content=f"搜索失败：{e}", is_error=True)  
  
        if not results:  
            return ToolResult(content="未找到相关结果")  
  
        lines = [f"{i+1}. {r['title']}\n   {r['href']}" for i, r in enumerate(results)]  
        return ToolResult(content="搜索结果：\n" + "\n".join(lines))  
