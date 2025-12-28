# examples/compose/workflow_resume_demo.py
"""
Workflow 静态断点恢复示例 (v0.5)

演示场景：
1. A -> B -> C 顺序执行。
2. B 节点执行时崩溃。
3. Resume 时，应跳过 A (已完成)，重试 B，然后执行 C。
"""
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from gecko.compose.workflow import Workflow, WorkflowContext, CheckpointStrategy
from gecko.compose.nodes import step, Next
from gecko.plugins.storage.factory import create_storage
from gecko.core.logging import setup_logging
from gecko.core.exceptions import WorkflowError

setup_logging(level="INFO")

FAIL_FLAG = True

@step("Step_A")
async def step_a(context: WorkflowContext):
    print("\n>>> 执行节点 A (Init)...")
    await asyncio.sleep(0.1)
    return "Data_A"

@step("Step_B")
async def step_b(context: WorkflowContext):
    global FAIL_FLAG
    print("\n>>> 执行节点 B (Processing)...")
    
    prev = context.get_last_output()
    print(f"    B 收到: {prev}")
    
    if FAIL_FLAG:
        print("    💀 节点 B 崩溃!")
        FAIL_FLAG = False 
        raise RuntimeError("Crash in Node B")
    
    print("    ✅ 节点 B 成功")
    return f"Processed({prev})"

@step("Step_C")
async def step_c(context: WorkflowContext):
    print("\n>>> 执行节点 C (Final)...")
    prev = context.get_last_output()
    print(f"    C 收到: {prev}")
    return f"Final({prev})"

async def main():
    db_file = "./resume_demo.db"
    db_url = f"sqlite:///{db_file}"
    
    if os.path.exists(db_file):
        try: os.remove(db_file)
        except: pass

    # 1. 初始化存储
    storage = await create_storage(db_url)

    try:
        wf = Workflow(
            name="ResumableFlow", 
            storage=storage,  # type: ignore
            checkpoint_strategy=CheckpointStrategy.ALWAYS
        )
        
        wf.add_node("A", step_a)
        wf.add_node("B", step_b)
        wf.add_node("C", step_c)
        
        wf.add_edge("A", "B")
        wf.add_edge("B", "C")
        wf.set_entry_point("A")

        session_id = "crash_test_static"

        print(f"\n{'='*50}")
        print("ROUND 1: 首次运行 (崩溃)")
        print(f"{'='*50}")

        try:
            await wf.execute("Start", session_id=session_id)
        except WorkflowError as e:
            print(f"\n🔴 捕获预期异常: {e}")

        print(f"\n{'='*50}")
        print("ROUND 2: 恢复运行")
        print(f"{'='*50}")
        
        try:
            # Resume 自动加载状态，发现 A 已完成，B 失败
            # 重新调度 B -> C
            final_result = await wf.resume(session_id=session_id)
            print(f"\n🎉 恢复并完成! 结果: {final_result}")
            
        except Exception as e:
            print(f"❌ 恢复失败: {e}")
            raise

    finally:
        await storage.shutdown()

    # 清理
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            if os.path.exists(db_file + "-wal"): os.remove(db_file + "-wal")
            if os.path.exists(db_file + "-shm"): os.remove(db_file + "-shm")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())