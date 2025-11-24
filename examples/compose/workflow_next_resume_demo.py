# examples/compose/workflow_next_resume_demo.py
"""
Workflow Next 指令断点恢复示例

演示 Gecko 如何处理 Next 指令的动态跳转持久化：
1. 节点 A 返回 Next("B", input="...")。
2. 系统在跳转后、B 执行前崩溃。
3. 系统恢复，直接从 next_pointer 指向的 B 继续执行，而不重复执行 A。
"""
import asyncio
import os
import sys

# 确保可以导入 gecko
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from gecko.compose.workflow import Workflow, WorkflowContext, CheckpointStrategy
from gecko.compose.nodes import step, Next
from gecko.plugins.storage.backends.sqlite import SQLiteStorage
from gecko.core.logging import setup_logging
from gecko.core.exceptions import WorkflowError

setup_logging(level="INFO")

# 全局标记，模拟第一次运行崩溃
CRASH_FLAG = True

@step("StartNode")
async def start_node(context: WorkflowContext):
    print("\n>>> [StartNode] 执行中...")
    # 动态跳转到 NextNode，并携带数据
    # 期望行为：StartNode 执行完后，Next 指令被持久化
    return Next(node="NextNode", input="Jumped Data")

@step("NextNode")
async def next_node(context: WorkflowContext):
    global CRASH_FLAG
    print("\n>>> [NextNode] 准备执行...")
    
    # 获取上一步传来的数据
    inp = context.get_last_output()
    print(f"    收到输入: {inp}")
    
    if CRASH_FLAG:
        print("    💀 [NextNode] 模拟系统崩溃! (Crash before logic)")
        CRASH_FLAG = False
        raise RuntimeError("System Crash in NextNode")
    
    print("    ✅ [NextNode] 执行成功")
    return f"Processed({inp})"

async def main():
    db_file = "next_resume.db"
    db_url = f"sqlite:///{db_file}"
    
    if os.path.exists(db_file):
        os.remove(db_file)

    storage = SQLiteStorage(db_url)
    await storage.initialize()

    wf = Workflow(
        name="NextResumeFlow", 
        storage=storage,
        checkpoint_strategy=CheckpointStrategy.ALWAYS
    )
    
    wf.add_node("StartNode", start_node)
    wf.add_node("NextNode", next_node)
    # 注意：这里没有显式添加 StartNode -> NextNode 的边
    # 完全依赖 Next 指令跳转
    wf.set_entry_point("StartNode")

    session_id = "next_crash_session"

    print(f"\n{'='*50}")
    print("ROUND 1: 首次运行 (预期在跳转后、NextNode 前崩溃)")
    print(f"{'='*50}")

    try:
        await wf.execute("Init", session_id=session_id)
    except WorkflowError as e:
        print(f"\n🔴 捕获到预期异常: {e}")

    print(f"\n{'='*50}")
    print("ROUND 2: 恢复运行 (预期直接从 NextNode 开始)")
    print(f"{'='*50}")
    
    # 重置 Workflow 实例模拟重启 (关键是 storage 和 session_id 一致)
    # 实际上用同一个 wf 实例也可以
    
    try:
        # 恢复执行
        # 期望：StartNode 不会被重新执行（没有 ">>> [StartNode] 执行中..." 输出）
        # 直接进入 NextNode，且能获取到 "Jumped Data"
        result = await wf.resume(session_id=session_id)
        print(f"\n🎉 恢复成功! 最终结果: {result}")
        
    except Exception as e:
        print(f"❌ 恢复失败: {e}")

    await storage.shutdown()
    if os.path.exists(db_file):
        os.remove(db_file)

if __name__ == "__main__":
    asyncio.run(main())