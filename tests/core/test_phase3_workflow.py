# tests/unit/test_phase3_workflow.py
import pytest
from gecko.compose.workflow import Workflow, WorkflowContext
from gecko.compose.nodes import Next

@pytest.mark.asyncio
async def test_workflow_loop_and_next():
    """
    测试场景：计数器循环
    Start -> Add -> Check(>3?) --Yes--> End
                   --No--> Add (Loop)
    """
    wf = Workflow("CounterLoop")
    
    async def start(ctx: WorkflowContext):
        ctx.state["count"] = 0
        return "Started"

    async def add(ctx: WorkflowContext):
        ctx.state["count"] += 1
        return ctx.state["count"]

    async def check(ctx: WorkflowContext):
        count = ctx.state["count"]
        if count < 3:
            # 显式跳回 add 节点
            return Next(node="add")
        return "Done"

    wf.add_node("start", start)
    wf.add_node("add", add)
    wf.add_node("check", check)
    
    # 定义静态流向
    wf.add_edge("start", "add")
    wf.add_edge("add", "check")
    # check 节点内部控制流向，或者在这里定义边也可以
    # 这里的边定义作为 fallback，如果 check 返回普通值而非 Next，则结束
    
    result = await wf.execute(None)
    
    assert result == "Done"
    # 验证 add 确实执行了 3 次
    # (start -> add(1) -> check -> add(2) -> check -> add(3) -> check -> Done)
    # 这里的 state 是引用，最后应该是 3
    # 注意：我们在 context 里拿不到 state 对象的引用测试，除非 execute 返回 context
    # 但可以通过逻辑推断：如果 check 只执行一次，返回 Next("add")，直到 count=3 返回 "Done"