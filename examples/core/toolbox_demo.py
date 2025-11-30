# examples/toolbox_demo.py
import asyncio
from gecko.core.toolbox import ToolBox
from gecko.plugins.tools.standard.calculator import CalculatorTool
from gecko.plugins.tools.standard.duckduckgo import DuckDuckGoSearchTool


async def main():
    # 1. 创建工具箱
    toolbox = ToolBox(
        tools=[CalculatorTool(), DuckDuckGoSearchTool()], # type: ignore
        max_concurrent=3,
        default_timeout=10.0,
        enable_retry=True,
        max_retries=2,
    )
    
    print(f"工具箱初始化完成: {toolbox}")
    print(f"已注册工具: {[t.name for t in toolbox.list_tools()]}\n")
    
    # 2. 单个工具执行
    print("=== 单个工具执行 ===")
    try:
        result = await toolbox.execute(
            "calculator",
            {"expression": "(10 + 5) * 2"},
            call_id="calc_001"
        )
        print(f"计算结果: {result}\n")
    except Exception as e:
        print(f"执行失败: {e}\n")
    
    # 3. 批量并发执行
    print("=== 批量并发执行 ===")
    tool_calls = [
        {
            "id": "call_1",
            "name": "calculator",
            "arguments": {"expression": "2 + 2"}
        },
        {
            "id": "call_2",
            "name": "duckduckgo_search",
            "arguments": {"query": "Python asyncio"}
        },
        {
            "id": "call_3",
            "name": "calculator",
            "arguments": {"expression": "100 / 5"}
        },
    ]
    
    results = await toolbox.execute_many(tool_calls)
    
    for r in results:
        status = "❌" if r.is_error else "✅"
        print(f"{status} {r.tool_name} ({r.call_id})")
        print(f"   结果: {r.result[:100]}")
        print(f"   耗时: {r.duration:.3f}s\n")
    
    # 4. 查看统计
    print("=== 执行统计 ===")
    toolbox.print_stats()
    
    # 5. 获取摘要
    summary = toolbox.get_summary()
    print("全局摘要:")
    print(f"  总执行次数: {summary['total_executions']}")
    print(f"  总错误次数: {summary['total_errors']}")
    print(f"  整体成功率: {summary['overall_success_rate']:.1%}")
    print(f"  平均耗时: {summary['avg_time_per_call']:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())