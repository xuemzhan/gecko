# examples/compose/workflow_next_resume_demo.py
"""
Workflow Next 指令断点恢复示例 (v0.5)

演示场景：
1. StartNode 返回 Next("NextNode") 指令。
2. 引擎在持久化该指令后、执行 NextNode 前发生崩溃。
3. Resume 时，引擎应检测到 next_pointer，直接跳转到 NextNode，而不重复执行 StartNode。
"""
import asyncio
import os
import sys

# 路径修正，确保能导入 gecko
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from gecko.compose.workflow import Workflow, WorkflowContext, CheckpointStrategy
from gecko.compose.nodes import step, Next
# [v0.5] 使用工厂方法创建存储
from gecko.plugins.storage.factory import create_storage
from gecko.core.logging import setup_logging
from gecko.core.exceptions import WorkflowError

setup_logging(level="INFO")

# 全局标记，模拟第一次运行崩溃
CRASH_FLAG = True

@step("StartNode")
async def start_node(context: WorkflowContext):
    print("\n>>> [StartNode] 执行中...")
    # 动态跳转到 NextNode
    return Next(node="NextNode", input="Jumped Data")

@step("NextNode")
async def next_node(context: WorkflowContext):
    global CRASH_FLAG
    print("\n>>> [NextNode] 准备执行...")
    
    inp = context.get_last_output()
    print(f"    收到输入: {inp}")
    
    if CRASH_FLAG:
        print("    💀 [NextNode] 模拟系统崩溃!")
        CRASH_FLAG = False
        raise RuntimeError("System Crash in NextNode")
    
    print("    ✅ [NextNode] 执行成功")
    return f"Processed({inp})"

async def main():
    db_file = "./next_resume.db"
    db_url = f"sqlite:///{db_file}"
    
    if os.path.exists(db_file):
        os.remove(db_file)

    # 1. 创建存储
    storage = await create_storage(db_url)
    
    try:
        wf = Workflow(
            name="NextResumeFlow", 
            storage=storage, # type: ignore
            # [Key] 必须为 ALWAYS，确保 Next 指令产生时立即持久化
            checkpoint_strategy=CheckpointStrategy.ALWAYS
        )
        
        wf.add_node("StartNode", start_node)
        wf.add_node("NextNode", next_node)
        wf.set_entry_point("StartNode")

        session_id = "next_crash_session"

        print(f"\n{'='*50}")
        print("ROUND 1: 首次运行 (预期崩溃)")
        print(f"{'='*50}")

        try:
            await wf.execute("Init", session_id=session_id)
        except WorkflowError as e:
            print(f"\n🔴 捕获到预期异常: {e}")

        print(f"\n{'='*50}")
        print("ROUND 2: 恢复运行 (预期跳过 StartNode)")
        print(f"{'='*50}")
        
        try:
            # 恢复执行
            # 期望：StartNode 不会被重新执行
            # 直接进入 NextNode，且能获取到 "Jumped Data"
            result = await wf.resume(session_id=session_id)
            print(f"\n🎉 恢复成功! 最终结果: {result}")
            
        except Exception as e:
            print(f"❌ 恢复失败: {e}")

    finally:
        # [v0.5 Best Practice] 必须关闭存储以释放文件锁 (SQLite WAL)
        await storage.shutdown()
        
    # 清理文件
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            if os.path.exists(db_file + "-wal"): os.remove(db_file + "-wal")
            if os.path.exists(db_file + "-shm"): os.remove(db_file + "-shm")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())