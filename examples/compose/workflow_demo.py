# examples/compose/workflow_demo.py
"""
Workflow 分支与状态示例 (v0.5)

展示特性：
1. 条件分支 (Conditional Branching)
2. 类型安全的数据获取 (get_last_output_as)
3. 状态隔离 (COW) 演示
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from gecko.compose.nodes import Next, step
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.core.logging import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger(__name__)


@step(name="InputAnalyzer")
async def analyze_input(user_input: str, context: WorkflowContext):
    """
    节点 1: 分析用户输入
    """
    logger.info(f"🔍 分析输入: {user_input}")
    
    # 在 Context State 中存储原始查询
    # v0.5 中，State 是 Copy-On-Write 的，这里的修改在当前节点生效
    # 并会在该层执行完毕后合并回主 Context
    context.state["original_query"] = user_input
    
    return len(user_input)


@step(name="QuickResponse")
def quick_response(length: int):
    """分支 A: 快速回复 (同步函数，自动卸载到线程池)"""
    logger.info("⚡️ 执行快速回复路径")
    return f"输入太短 ({length} 字符)，请提供更多细节。"


@step(name="DeepThinking")
async def deep_thinking(context: WorkflowContext):
    """分支 B: 深度思考"""
    logger.info("🧠 执行深度思考路径")
    # 读取上游存入的状态
    query = context.state.get("original_query", "")
    return f"针对 '{query}' 的深度分析报告..."


@step(name="FinalSummary")
async def final_summary(result: Any):
    logger.info("✅ 生成最终摘要")
    return f"=== Workflow Result ===\n{result}"


async def main():
    wf = Workflow(name="DemoFlow", max_steps=10)
    
    # 添加节点
    wf.add_node("Analyze", analyze_input)
    wf.add_node("Quick", quick_response)
    wf.add_node("Deep", deep_thinking)
    wf.add_node("Summary", final_summary)
    
    # 拓扑结构
    wf.set_entry_point("Analyze")
    
    # [v0.5] 使用 get_last_output_as(int) 确保类型转换安全
    # 如果输入长度 < 5 -> Quick
    wf.add_edge("Analyze", "Quick", lambda ctx: ctx.get_last_output_as(int) < 5)
    # 如果输入长度 >= 5 -> Deep
    wf.add_edge("Analyze", "Deep", lambda ctx: ctx.get_last_output_as(int) >= 5)
    
    # 汇聚
    wf.add_edge("Quick", "Summary")
    wf.add_edge("Deep", "Summary")
    
    if not wf.validate():
        print("❌ Workflow 验证失败")
        return

    print("\n" + "="*40)
    print("Case 1: 短输入 (走快速分支)")
    print("="*40)
    res1 = await wf.execute("Hi")
    print(f"Result: {res1}")
    
    print("\n" + "="*40)
    print("Case 2: 长输入 (走深度分支)")
    print("="*40)
    res2 = await wf.execute("Hello World!")
    print(f"Result: {res2}")


if __name__ == "__main__":
    asyncio.run(main())