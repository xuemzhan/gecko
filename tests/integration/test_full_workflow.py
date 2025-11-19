# tests/integration/test_full_workflow.py
import pytest
import asyncio
from gecko.compose import Workflow, Team, Loop
from gecko.core.builder import AgentBuilder

@pytest.mark.asyncio
async def test_rag_team_loop_workflow(mock_model):
    research_agent = AgentBuilder().with_model(mock_model).build()
    review_team = Team([research_agent, research_agent])

    workflow = Workflow(name="Enterprise RAG Review Loop")
    workflow.add_node("research", research_agent)
    workflow.add_node("review", review_team)
    workflow.add_node("refine", Loop(
        body=Workflow().add_node("research", research_agent).add_edge("start", "research").add_edge("research", "end"),
        condition=lambda ctx: "incomplete" in str(ctx.get("review", []))
    ))
    workflow.add_edge("start", "research")
    workflow.add_edge("research", "review")
    workflow.add_edge("review", "refine")
    workflow.add_edge("refine", "end")

    output = await workflow.execute("2025 AI 趋势分析报告")
    assert "2025" in str(output)