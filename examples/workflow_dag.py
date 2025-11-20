# examples/workflow_dag.py
import asyncio
from gecko.compose.workflow import Workflow
from gecko.compose.nodes import step, Next
from gecko.compose.team import Team
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.models.zhipu import glm_4_5_air 

# 工厂函数：创建新的 Agent 实例
def make_agent():
    # 调高 temperature 增加变化
    return AgentBuilder().with_model(glm_4_5_air(temperature=0.7)).build()

@step("research")
async def research(context):
    """
    调研节点：优先使用上一轮的反馈（如果有），否则使用初始输入
    """
    # 1. 尝试获取上一步传来的“修正指令”（由 Next.input 注入到 last_output）
    # 2. 如果没有，使用全局初始输入 context.input
    topic = context.history.get("last_output") or context.input
    
    # 如果上一步是 Team 的结果（List），说明是刚从 Team 过来但没经过 Check 修改（理论上不会，但为了健壮性）
    if isinstance(topic, list): 
        topic = context.input

    print(f"\n🔍 [Research] 正在调研: {topic}")
    
    agent = make_agent()
    # 这里的 prompt 决定了输出长度
    output = await agent.run([Message(role="user", content=f"{topic}")])
    return output.content

# 定义 Team 节点
team_node = Team(members=[make_agent(), make_agent()])

@step("check_quality")
async def check_quality(context):
    """
    质检节点：决定是否通过，或者打回重做
    """
    # Team 的输出在 history 中，key 是节点名 "team_review"
    raw_result = context.history.get("team_review")
    
    # [修复 1] 将 List[str] 合并为单个 String
    if isinstance(raw_result, list):
        combined_text = "\n---\n".join(str(r) for r in raw_result)
    else:
        combined_text = str(raw_result)
        
    text_len = len(combined_text)
    print(f"🧐 [Check] 当前内容长度: {text_len} 字符")

    # [修复 2] 获取或初始化循环计数器 (防止死循环)
    loop_count = context.state.get("loop_count", 0)
    
    # 设定阈值：比如长度小于 100 且重试次数少于 3 次
    if text_len < 100 and loop_count < 2:
        new_count = loop_count + 1
        context.state["loop_count"] = new_count
        print(f"⚠️ [Check] 内容太短，第 {new_count} 次打回重做...")
        
        # [修复 3] 修改 Prompt，强制要求长文，改变 Agent 的行为
        new_prompt = f"之前的内容太短了（只有{text_len}字）。请针对 '{context.input}' 写一篇不少于 200 字的详细分析报告。"
        
        # 返回 Next 指令：
        # - node: 跳转回 research 节点
        # - input: 将 new_prompt 传递给 research 节点
        return Next(node="research", input=new_prompt)
    
    print("✅ [Check] 质量达标 (或已达最大重试次数)")
    return f"最终报告 (经过 {loop_count} 次修正):\n{combined_text}"

async def main():
    workflow = Workflow("ResearchLoop")
    
    # 1. 注册节点
    workflow.add_node("research", research)
    workflow.add_node("team_review", team_node)
    workflow.add_node("check", check_quality)
    
    # 2. 定义流向
    workflow.set_entry_point("research")
    workflow.add_edge("research", "team_review")
    workflow.add_edge("team_review", "check")
    # check -> research 的边由代码逻辑动态控制
    
    print("🚀 启动工作流...")
    # 初始输入简单一点，故意诱导第一次生成较短的内容
    output = await workflow.execute("简述 AI Agent")
    print("\n🎉 工作流结束 Result:\n", output)

if __name__ == "__main__":
    asyncio.run(main())