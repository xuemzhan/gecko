# examples/workflow_demo.py
"""
Workflow 编排示例

展示 Gecko Workflow 引擎的核心特性：
1. 条件分支 (Conditional Branching)
2. 循环与跳转 (Next Instruction)
3. 上下文状态共享 (Context State)
4. 混合节点编排 (Agent + Function)

运行前提：
    export ZHIPU_API_KEY="your_api_key"
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from gecko.compose.nodes import Next, step
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.logging import get_logger
from gecko.plugins.models.zhipu import glm_4_5_air

logger = get_logger(__name__)


# ========================= 1. 定义节点 =========================

@step(name="InputAnalyzer")
async def analyze_input(user_input: str, context: WorkflowContext):
    """
    节点 1: 分析用户输入
    将输入存入 context，并返回输入长度供后续判断
    """
    logger.info(f"🔍 分析输入: {user_input}")
    
    # 在 Context 中存储状态
    context.state["original_query"] = user_input
    context.state["loop_count"] = 0
    
    return len(user_input)


@step(name="QuickResponse")
def quick_response(length: int):
    """
    节点 2A: 快速回复 (分支 A)
    当输入较短时，直接返回简单规则回复
    """
    logger.info("⚡️ 执行快速回复路径")
    return f"输入太短 ({length} 字符)，请提供更多细节。"


@step(name="DeepThinking")
async def deep_thinking_agent(context: WorkflowContext):
    """
    节点 2B: 深度思考 (分支 B) - 使用 Agent
    当输入较长时，调用 LLM 进行分析
    """
    logger.info("🧠 执行深度思考路径 (Agent)")
    query = context.state["original_query"]
    
    # 构建一个简单的 Zhipu Agent
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        return "Error: No API Key found"

    model = glm_4_5_air(api_key=api_key)
    agent = (
        AgentBuilder()
        .with_model(model)
        .with_session_id("demo_session")
        .build()
    )
    
    # 执行 Agent
    result = await agent.run(f"请简要分析这句话的情感：{query}")
    return result.content


@step(name="RefinementLoop")
def refinement_loop(context: WorkflowContext):
    """
    节点 3: 优化循环 (Loop)
    模拟一个自我修正循环：如果结果包含 "Error"，重试最多 3 次
    """
    last_output = context.get_last_output()
    loop_count = context.state["loop_count"]
    
    logger.info(f"🔄 检查结果 (Loop {loop_count}): {str(last_output)[:20]}...")
    
    # 模拟：如果是 Error 且重试次数未到，通过 Next 跳转回 DeepThinking
    if "Error" in str(last_output) and loop_count < 2:
        context.state["loop_count"] += 1
        logger.warning("⚠️ 检测到错误，触发重试循环...")
        
        # [Fix] 目标节点名称必须与 Workflow.add_node 中注册的名称一致 ("Deep")
        return Next(node="Deep", input=context.state["original_query"])
    
    return last_output


@step(name="FinalSummary")
async def final_summary(result: Any):
    """
    节点 4: 最终汇总
    """
    logger.info("✅ 生成最终摘要")
    return f"=== Workflow Result ===\n{result}"


# ========================= 2. 构建与运行 =========================

async def main():
    # 1. 创建 Workflow
    wf = Workflow(name="DemoFlow", max_steps=20)
    
    # 2. 添加节点
    # 注意：Workflow 注册的名称是 key，Next 指令跳转必须使用这个 key
    wf.add_node("Analyze", analyze_input)
    wf.add_node("Quick", quick_response)
    wf.add_node("Deep", deep_thinking_agent)  # 注册名为 "Deep"
    wf.add_node("LoopCheck", refinement_loop)
    wf.add_node("Summary", final_summary)
    
    # 3. 定义边与条件 (Topology)
    
    # 入口 -> 分析
    wf.set_entry_point("Analyze")
    
    # 分析 -> 分支 (根据输入长度)
    wf.add_edge("Analyze", "Quick", lambda ctx: ctx.get_last_output() < 5)
    wf.add_edge("Analyze", "Deep", lambda ctx: ctx.get_last_output() >= 5)
    
    # 分支汇聚 -> 循环检查
    wf.add_edge("Quick", "LoopCheck")
    wf.add_edge("Deep", "LoopCheck")
    
    # 循环检查 -> 结束
    wf.add_edge("LoopCheck", "Summary")
    
    # 4. 验证结构
    if not wf.validate():
        print("❌ Workflow 验证失败")
        return

    # 打印结构图
    wf.print_structure()
    
    print("\n" + "="*40)
    print("Case 1: 短输入 (走快速分支)")
    print("="*40)
    res1 = await wf.execute("Hi")
    print(f"\n{res1}")
    
    print("\n" + "="*40)
    print("Case 2: 长输入 (走 Agent 分支)")
    print("="*40)
    # 提示：确保环境变量 ZHIPU_API_KEY 已设置
    # 如果未设置 API Key，Agent 会返回 "Error: No API Key found"，从而触发 LoopCheck 的重试逻辑
    res2 = await wf.execute("我今天非常开心，想写代码！")
    print(f"\n{res2}")


if __name__ == "__main__":
    # 使用 uvloop (如果安装了) 或标准循环
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
        
    asyncio.run(main())