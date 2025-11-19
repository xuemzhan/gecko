# examples/complex_workflow.py
import asyncio
from gecko.compose import Workflow, step, Team, Condition, Loop
from gecko.core.builder import AgentBuilder
from gecko.plugins.models.zhipu import glm_4_5_air
from gecko.core.message import Message

@step("research")
async def research(context):
    agent = AgentBuilder().with_model(glm_4_5_air()).build()
    output = await agent.run([Message(role="user", content=context["input"])])
    return output.content

team = Team(
    members=[
        AgentBuilder().with_model(glm_4_5_air()).build(),
        AgentBuilder().with_model(glm_4_5_air()).build()
    ],
    aggregator=lambda results: " ".join(results)
)

async def needs_loop(context):
    return len(context.get("research", "")) < 100  # 示例条件

async def main():
    workflow = Workflow()
    workflow.add_node("research", research)
    workflow.add_node("team_review", team)  # Team 作为节点
    workflow.add_node("loop_refine", Loop(body=Workflow(), condition=needs_loop))  # Loop 示例
    workflow.add_edge("start", "research")
    workflow.add_edge("research", "team_review", condition=Condition(lambda ctx: "AI" in ctx["input"]))  # 条件边
    workflow.add_edge("team_review", "loop_refine")
    workflow.add_edge("loop_refine", "end")

    print(workflow.to_mermaid())  # 可视化

    output = await workflow.execute("2025 AI 趋势")
    print(output)

if __name__ == "__main__":
    asyncio.run(main())