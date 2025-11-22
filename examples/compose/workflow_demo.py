# examples/workflow_demo.py
"""
Workflow 编排示例 (Updated for V0.2)

展示 Gecko Workflow 引擎的核心特性：
1. 条件分支 (Conditional Branching)
2. 循环与跳转 (Next Instruction)
3. [Updated] 状态自动更新 (Next.update_state) - Phase 2 新特性
4. [Updated] 显式循环支持 (allow_cycles) - Phase 2 新特性

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
# [Updated] 使用新的模型类
from gecko.plugins.models.presets.zhipu import ZhipuChat

logger = get_logger(__name__)


# ========================= 1. 定义节点 =========================

@step(name="InputAnalyzer")
async def analyze_input(user_input: str, context: WorkflowContext):
    """
    节点 1: 分析用户输入
    """
    logger.info(f"🔍 分析输入: {user_input}")
    
    # 在 Context 中存储状态
    context.state["original_query"] = user_input
    
    # [Phase 2 Update] 这里的 loop_count 初始化可以通过 Next(..., update_state=...) 在后续节点完成，
    # 或者在此处初始化。为了演示，我们这里只存 query。
    
    return len(user_input)


@step(name="QuickResponse")
def quick_response(length: int):
    """
    节点 2A: 快速回复 (分支 A)
    """
    logger.info("⚡️ 执行快速回复路径")
    return f"输入太短 ({length} 字符)，请提供更多细节。"


@step(name="DeepThinking")
async def deep_thinking_agent(context: WorkflowContext):
    """
    节点 2B: 深度思考 (分支 B) - 使用 Agent
    """
    logger.info("🧠 执行深度思考路径 (Agent)")
    query = context.state["original_query"]
    
    # 构建 Agent
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        return "Error: No API Key found"

    # [Updated] 使用 ZhipuChat
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")
    agent = (
        AgentBuilder()
        .with_model(model)
        .with_session_id("demo_session")
        .build()
    )
    
    # 执行 Agent
    # 注意：由于 Phase 1 移除了隐式拆包，Agent 这里接收的是字符串 query (从 state 获取)
    # 如果 DeepThinking 的上游节点返回了 dict，这里需要显式处理。
    result = await agent.run(f"请简要分析这句话的情感：{query}")
    return result.content # type: ignore


@step(name="RefinementLoop")
def refinement_loop(context: WorkflowContext):
    """
    节点 3: 优化循环 (Loop)
    [Updated] 使用 Phase 2 的 update_state 特性简化状态管理
    """
    last_output = context.get_last_output()
    # 这里的 loop_count 如果不存在则默认为 0
    loop_count = context.state.get("loop_count", 0)
    
    logger.info(f"🔄 检查结果 (Loop {loop_count}): {str(last_output)[:20]}...")
    
    # 模拟：如果是 Error 且重试次数未到
    if "Error" in str(last_output) and loop_count < 2:
        logger.warning("⚠️ 检测到错误，触发重试循环...")
        
        # [Phase 2 Feature] 使用 update_state 在跳转时自动更新计数器
        # 这样就不需要手动操作 context.state["loop_count"] += 1
        return Next(
            node="Deep", 
            input=context.state["original_query"],
            update_state={"loop_count": loop_count + 1}
        )
    
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
    # [Phase 2 Feature] 显式开启循环支持 (allow_cycles=True)
    # 虽然这里主要靠 Next 跳转，但开启此选项是 V0.2 的推荐做法，避免静态检查误报复杂拓扑
    wf = Workflow(name="DemoFlow", max_steps=20, allow_cycles=True)
    
    # 2. 添加节点
    wf.add_node("Analyze", analyze_input)
    wf.add_node("Quick", quick_response)
    wf.add_node("Deep", deep_thinking_agent)  # 注册名为 "Deep"
    wf.add_node("LoopCheck", refinement_loop)
    wf.add_node("Summary", final_summary)
    
    # 3. 定义边与条件 (Topology)
    
    # 入口 -> 分析
    wf.set_entry_point("Analyze")
    
    # 分析 -> 分支 (根据输入长度)
    # [Phase 1 Update] get_last_output_as(int) 确保类型安全
    wf.add_edge("Analyze", "Quick", lambda ctx: ctx.get_last_output_as(int) < 5)
    wf.add_edge("Analyze", "Deep", lambda ctx: ctx.get_last_output_as(int) >= 5)
    
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
    res2 = await wf.execute("我今天非常开心，想写代码！")
    print(f"\n{res2}")


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
        
    asyncio.run(main())