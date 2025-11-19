# gecko/plugins/tools/duckduckgo.py
from duckduckgo_search import DDGS
from gecko.plugins.tools.base import BaseTool
from gecko.plugins.tools.registry import tool

@tool
class DuckDuckGoSearch(BaseTool):
    name: str = "duckduckgo_search"                             # 必须带类型注解
    description: str = "使用 DuckDuckGo 搜索互联网，返回前 5 条结果（无需 API Key）"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        },
        "required": ["query"]
    }

    async def execute(self, arguments: dict) -> str:
        query = arguments.get("query", "").strip()
        if not query:
            return "错误：搜索关键词为空"

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "未找到相关结果"
            lines = [f"{i+1}. {r['title']}\n   {r['href']}" for i, r in enumerate(results)]
            return "搜索结果：\n" + "\n".join(lines)
        except Exception as e:
            return f"搜索失败：{str(e)}"