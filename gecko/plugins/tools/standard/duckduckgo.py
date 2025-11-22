# gecko/plugins/tools/standard/duckduckgo.py
"""
DuckDuckGo 搜索工具 (标准版)

基于 duckduckgo-search 库实现。
重构改进：
1. 继承新的 BaseTool，使用 Pydantic V2 定义参数。
2. 使用 run_sync 将同步的网络 I/O 卸载到线程池，防止阻塞 Event Loop。
3. 增强错误处理和结果格式化。
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from gecko.core.logging import get_logger
from gecko.core.utils import run_sync
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import register_tool

logger = get_logger(__name__)


class DuckDuckGoArgs(BaseModel):
    query: str = Field(
        ..., 
        description="搜索关键词",
        min_length=1,
        max_length=200
    )
    max_results: int = Field(
        default=5, 
        description="返回的最大结果数量 (1-10)", 
        ge=1, 
        le=10
    )


@register_tool("duckduckgo_search")
class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = (
        "使用 DuckDuckGo 搜索引擎搜索互联网信息。"
        "当需要获取实时新闻、具体事实或不知道的信息时使用。"
        "无需 API Key。"
    )
    args_schema: Type[BaseModel] = DuckDuckGoArgs

    async def _run(self, args: DuckDuckGoArgs) -> ToolResult: # type: ignore
        """
        执行搜索
        注意：DDGS 库主要是同步 IO，必须通过 run_sync 卸载到线程池
        """
        # 检查依赖
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult(
                content=(
                    "错误：未安装 duckduckgo_search 库。\n"
                    "请运行: pip install duckduckgo-search"
                ),
                is_error=True
            )

        query = args.query.strip()
        
        # 定义同步执行函数
        def _search_sync() -> List[Dict[str, str]]:
            results = []
            try:
                # 使用上下文管理器确保 session 关闭
                with DDGS() as ddgs:
                    # text() 方法是生成器，需要转 list
                    # backend="api" 通常更稳定
                    raw_results = ddgs.text(
                        keywords=query,
                        max_results=args.max_results,
                        backend="api" 
                    )
                    # 立即消费生成器以捕获可能的网络异常
                    results = list(raw_results)
            except Exception as e:
                logger.error("DuckDuckGo search failed", error=str(e))
                raise e
            return results

        try:
            # 异步非阻塞执行
            # 这里的 run_sync 引用自 gecko.core.utils，底层使用 to_thread
            raw_data = await run_sync(_search_sync()) # type: ignore

            if not raw_data:
                return ToolResult(content=f"未找到关于 '{query}' 的相关结果。")

            return self._format_results(raw_data)

        except Exception as e:
            return ToolResult(
                content=f"搜索请求失败: {str(e)}",
                is_error=True
            )

    def _format_results(self, results: List[Dict[str, Any]]) -> ToolResult:
        """格式化搜索结果为易读文本"""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No Title')
            link = r.get('href', '#')
            body = r.get('body', '')
            
            lines.append(f"[{i}] {title}")
            lines.append(f"    Link: {link}")
            lines.append(f"    Snippet: {body}\n")
        
        formatted_text = "DuckDuckGo 搜索结果:\n" + "\n".join(lines)
        
        return ToolResult(
            content=formatted_text,
            metadata={"source": "duckduckgo", "count": len(results)}
        )