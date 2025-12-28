# examples/compose/workflow_dag_demo.py
"""
Workflow DAG & Loop 示例 (v0.5)

展示复杂拓扑编排：
1. 循环与状态更新 (Next + update_state)
2. Team 节点集成 (Workflow 嵌套 Team)
3. 动态输入注入

运行前提：
    export ZHIPU_API_KEY="your_api_key"
"""
import asyncio
import os

from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.compose.nodes import step, Next
from gecko.compose.team import Team, MemberResult
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.core.logging import setup_logging
from gecko.plugins.models.presets.zhipu import ZhipuChat 

setup_logging(level="INFO")

# 检查 API Key
api_key = os.getenv("ZHIPU_API_KEY")

# 工厂函数：创建新的 Agent 实例
def make_agent(role_name: str = "Assistant"):
    if not api_key:
        # Mock 模式，防止无 Key 报错
        class MockAgent:
            async def run(self, x): 
                from gecko.core.output import AgentOutput
                return AgentOutput(content=f"[{role_name} View]: ok")
        return MockAgent()

    model = ZhipuChat(api_key=api_key, model="glm-4-flash", temperature=0.7)
    return AgentBuilder().with_model(model).with_session_id(f"agent_{role_name}").build()


@step("research")
async def research(context: WorkflowContext):
    """
    调研节点：优先使用上一轮的反馈（修正指令），否则使用初始输入
    """
    # 1. 如果是从 Loop 回跳过来，Next.input 会注入到 last_output
    # 2. 如果是第一次运行，last_output 默认为 context.input
    topic = context.get_last_output()
    
    print(f"\n🔍 [Research] 正在调研: {topic}")
    
    # 这里简单模拟调研过程
    agent = make_agent("Researcher")
    # 如果是 MockAgent，没有 run 方法会报错吗？NodeExecutor 会处理
    output = await agent.run([Message(role="user", content=f"{topic}")])
    return output.content  # type: ignore


# 定义 Team 节点：由两个评审员组成
team_node = Team(
    members=[make_agent("Reviewer_1"), make_agent("Reviewer_2")],  # type: ignore
    name="ReviewBoard"
)


@step("check_quality")
async def check_quality(context: WorkflowContext):
    """
    质检节点：决定是否通过，或者打回重做
    """
    # 获取 Team 的输出 (Workflow 自动将 Team 的输出放入 history)
    raw_result = context.history.get("team_review")
    
    combined_text = ""
    # [v0.5] Team 返回 List[MemberResult]
    if isinstance(raw_result, list):
        valid_contents = []
        for res in raw_result:
            if isinstance(res, MemberResult) and res.is_success:
                valid_contents.append(str(res.result))
        combined_text = "\n---\n".join(valid_contents)
    else:
        combined_text = str(raw_result)
        
    text_len = len(combined_text)
    print(f"🧐 [Check] 当前评审内容长度: {text_len} 字符")

    # 获取循环计数器 (从 State 中)
    loop_count = context.state.get("loop_count", 0)
    
    # 模拟逻辑：如果这是第一次执行 (loop_count < 1)，强制打回重做
    if loop_count < 1:
        new_count = loop_count + 1
        print(f"⚠️ [Check] 质量未达标，第 {new_count} 次打回重做...")
        
        new_prompt = f"之前的内容不够深刻。请针对 '{context.input}' 写一篇更详细的报告。"
        
        # [v0.5 Best Practice] 
        # 使用 Next 指令跳转，并利用 update_state 原子更新状态
        # 这样避免了直接修改 context.state 的副作用担忧
        return Next(
            node="research", 
            input=new_prompt,
            update_state={"loop_count": new_count}
        )
    
    print("✅ [Check] 质量达标")
    return f"最终报告 (经过 {loop_count} 次修正):\n{combined_text}"


async def main():
    # allow_cycles=True 允许静态图存在环（虽然这里是用 Next 动态跳转）
    workflow = Workflow("ResearchLoop", allow_cycles=True)
    
    # 1. 注册节点
    workflow.add_node("research", research)
    workflow.add_node("team_review", team_node)
    workflow.add_node("check", check_quality)
    
    # 2. 定义流向
    workflow.set_entry_point("research")
    workflow.add_edge("research", "team_review")
    workflow.add_edge("team_review", "check")
    # check -> research 的边由 Next 动态控制
    
    print("🚀 启动工作流...")
    
    # 初始输入
    output = await workflow.execute("简述 AI Agent 的未来")
    print("\n🎉 工作流结束 Result:\n", output)


if __name__ == "__main__":
    asyncio.run(main())