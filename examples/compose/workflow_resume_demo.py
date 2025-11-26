# examples/compose/workflow_resume_demo.py
"""
Workflow 断点恢复示例 (Resumability Demo)

演示 Gecko 如何处理系统崩溃和状态恢复：
1. 使用 SQLite 持久化状态 (基于重构后的 Storage 插件)
2. 模拟节点执行中的意外崩溃
3. 使用 resume() 接口从断点继续执行 (Phase 3 新特性)
"""
import asyncio
import os
import sys

# 确保可以导入 gecko
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from gecko.compose.workflow import Workflow, WorkflowContext, CheckpointStrategy
from gecko.compose.nodes import step, Next
# [Fix] Use create_storage factory
from gecko.plugins.storage.factory import create_storage
from gecko.core.logging import setup_logging
# [Fix] Import WorkflowError
from gecko.core.exceptions import WorkflowError

# 配置日志以便观察恢复过程
setup_logging(level="INFO")

# 全局标记，用于模拟“第一次必挂，第二次成功”
FAIL_FLAG = True

# ========================= 定义节点 =========================

@step("Step_A")
async def step_a(context: WorkflowContext):
    print("\n>>> 执行节点 A (初始化数据)...")
    # 模拟耗时操作
    await asyncio.sleep(0.5)
    return "Data A"

@step("Step_B")
async def step_b(context: WorkflowContext):
    global FAIL_FLAG
    print("\n>>> 执行节点 B (处理数据)...")
    
    # 获取上一步结果
    prev = context.get_last_output()
    print(f"    节点 B 收到: {prev}")
    
    if FAIL_FLAG:
        print("    💀 模拟系统崩溃! (System Crash)")
        FAIL_FLAG = False 
        raise RuntimeError("Unexpected System Failure in Node B")
    
    print("    ✅ 节点 B 执行成功")
    
    # [修改] 使用 Next 跳转，验证动态指针恢复
    return Next(node="C", input=f"Processed({prev})")

@step("Step_C")
async def step_c(context: WorkflowContext):
    print("\n>>> 执行节点 C (最终汇总)...")
    # [修改] 验证输入是否通过 Next 传递过来
    prev = context.get_last_output()
    print(f"    节点 C 收到: {prev}")
    return f"FinalResult -> {prev}"

# ========================= 主流程 =========================

async def main():
    # [Fix] Use explicit relative path
    db_file = "./resume_demo.db"
    db_url = f"sqlite:///{db_file}"
    
    # 清理旧数据确保 Demo 可重复
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except:
            pass

    print(f"🔌 初始化存储: {db_url}")
    # 1. 初始化存储
    # 断点恢复必须依赖持久化存储
    # [Fix] Use factory
    storage = await create_storage(db_url)

    # 2. 定义工作流
    wf = Workflow(
        name="ResumableFlow", 
        storage=storage, # type: ignore
        # [Phase 3 Feature] 策略: ALWAYS (每步保存)，这是 Resume 的前提
        checkpoint_strategy=CheckpointStrategy.ALWAYS
    )
    
    wf.add_node("A", step_a)
    wf.add_node("B", step_b)
    wf.add_node("C", step_c)
    
    wf.add_edge("A", "B")
    wf.add_edge("B", "C")
    wf.set_entry_point("A")

    session_id = "crash_test_session_001"

    print(f"\n{'='*50}")
    print("ROUND 1: 首次运行 (预期在 B 节点崩溃)")
    print(f"{'='*50}")

    try:
        # 正常执行
        await wf.execute("Start", session_id=session_id)
    except WorkflowError as e: # [Fix] Catch WorkflowError correctly
        print(f"\n🔴 捕获到预期异常: {e}")
        print("   工作流已中断。状态应已保存到 SQLite。")
    except Exception as e:
        print(f"\n🔴 捕获到其他异常: {e}")

    print(f"\n{'='*50}")
    print("ROUND 2: 恢复运行 (预期跳过 A，重试 B，完成 C)")
    print(f"{'='*50}")
    
    # 模拟重启系统：可以重新实例化 Workflow 对象，只要 session_id 和 storage 一样
    # wf_new = Workflow(..., storage=storage) 
    
    try:
        # [Phase 3 Feature] 调用 resume 而不是 execute
        # 引擎会自动加载上次的状态，发现 A 已完成，从 B 开始重试
        final_result = await wf.resume(session_id=session_id)
        
        print(f"\n🎉 工作流恢复并完成!")
        print(f"   最终结果: {final_result}")
        
    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 清理
    await storage.shutdown()
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            # SQLite WAL 模式可能会产生额外文件
            if os.path.exists(db_file + ".lock"): os.remove(db_file + ".lock")
            if os.path.exists(db_file + "-wal"): os.remove(db_file + "-wal")
            if os.path.exists(db_file + "-shm"): os.remove(db_file + "-shm")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())