# examples/workflow_dag.py
import asyncio
import os
from typing import List

from gecko.compose.workflow import Workflow
from gecko.compose.nodes import step, Next
# [Refactor Note] 引入 MemberResult
from gecko.compose.team import Team, MemberResult
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
# [Refactor Note] 使用新的 Model Preset 类
from gecko.plugins.models.presets.zhipu import ZhipuChat 

# 检查 API Key
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    print("⚠️ Warning: ZHIPU_API_KEY not found in env. Mocking behavior might be needed.")

# 工厂函数：创建新的 Agent 实例
def make_agent(role_name: str = "Assistant"):
    # [Refactor] 使用 ZhipuChat 类
    model = ZhipuChat(api_key=api_key, model="glm-4-flash", temperature=0.7) # type: ignore
    return AgentBuilder().with_model(model).with_session_id(f"agent_{role_name}").build()

@step("research")
async def research(context):
    """
    调研节点：优先使用上一轮的反馈（如果有），否则使用初始输入
    """
    # 1. 尝试获取上一步传来的“修正指令”（由 Next.input 注入到 last_output）
    # 2. 如果没有，使用全局初始输入 context.input
    topic = context.get_last_output()
    
    # 如果是第一次运行，last_output 默认为 input
    # 如果是从 Loop 回跳过来，last_output 是 "new_prompt"
    
    print(f"\n🔍 [Research] 正在调研: {topic}")
    
    agent = make_agent("Researcher")
    output = await agent.run([Message(role="user", content=f"{topic}")])
    return output.content # type: ignore

# 定义 Team 节点
# [Refactor] Team 现在是类型安全的，成员可以是 Agent
team_node = Team(members=[make_agent("Reviewer_1"), make_agent("Reviewer_2")], name="ReviewBoard")

@step("check_quality")
async def check_quality(context):
    """
    质检节点：决定是否通过，或者打回重做
    """
    # Team 的输出在 history 中，key 是节点名 "team_review"
    raw_result = context.history.get("team_review")
    
    # [Refactor Note] Team 现在返回 List[MemberResult]
    combined_text = ""
    if isinstance(raw_result, list):
        valid_contents = []
        for res in raw_result:
            # 显式检查类型和成功状态
            if isinstance(res, MemberResult):
                if res.is_success:
                    valid_contents.append(str(res.result))
                else:
                    print(f"⚠️ 忽略失败的专家意见: {res.error}")
        combined_text = "\n---\n".join(valid_contents)
    else:
        # 防御性代码
        combined_text = str(raw_result)
        
    text_len = len(combined_text)
    print(f"🧐 [Check] 当前有效内容长度: {text_len} 字符")

    # 获取或初始化循环计数器 (使用 WorkflowContext.state)
    loop_count = context.state.get("loop_count", 0)
    
    # 设定阈值：比如长度小于 100 且重试次数少于 2 次
    if text_len < 100 and loop_count < 2:
        new_count = loop_count + 1
        # 更新状态
        context.state["loop_count"] = new_count
        print(f"⚠️ [Check] 内容太短，第 {new_count} 次打回重做...")
        
        new_prompt = f"之前的内容太短了（只有{text_len}字）。请针对 '{context.input}' 写一篇不少于 200 字的详细分析报告。"
        
        # 返回 Next 指令：
        # - node: 跳转回 research 节点
        # - input: 将 new_prompt 传递给 research 节点
        # - [Phase 2 Feature] 也可以使用 update_state={"loop_count": new_count} 来更新状态
        return Next(node="research", input=new_prompt)
    
    print("✅ [Check] 质量达标 (或已达最大重试次数)")
    return f"最终报告 (经过 {loop_count} 次修正):\n{combined_text}"

async def main():
    # [Phase 2 Feature] 显式开启 allow_cycles，虽然这里我们用 Next 跳转，但这是推荐做法
    workflow = Workflow("ResearchLoop", allow_cycles=True)
    
    # 1. 注册节点
    workflow.add_node("research", research)
    workflow.add_node("team_review", team_node)
    workflow.add_node("check", check_quality)
    
    # 2. 定义流向
    workflow.set_entry_point("research")
    workflow.add_edge("research", "team_review")
    workflow.add_edge("team_review", "check")
    # check -> research 的边由代码逻辑动态控制 (Next)
    
    print("🚀 启动工作流...")
    
    if not api_key:
        print("🚫 缺少 API Key，演示将失败或使用 Mock 数据。")
        return

    # 初始输入简单一点，故意诱导第一次生成较短的内容
    output = await workflow.execute("简述 AI Agent")
    print("\n🎉 工作流结束 Result:\n", output)

if __name__ == "__main__":
    asyncio.run(main())