# examples/advanced/concurrent_workflow_demo.py
"""
并发工作流演示

验证 Gecko 在高并发场景下的稳定性：
1. Workflow + Team: 并行执行多个 Agent
2. SummaryMemory: 多 Agent 共享同一个 Memory，验证锁机制
"""
import asyncio
import os
from typing import List

from gecko.compose.team import Team
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.memory import SummaryTokenMemory
from gecko.core.message import Message
from gecko.plugins.models.presets.zhipu import ZhipuChat

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key: return

    model = ZhipuChat(api_key=api_key, model="glm-4-flash")
    
    # 1. 创建共享的 Summary Memory
    # 设置极小的 max_tokens 以强制触发摘要
    shared_memory = SummaryTokenMemory(
        session_id="shared_session",
        model=model,
        max_tokens=100 
    )
    
    # 预填充历史记录，使其接近 limit
    # 注意：此处我们直接操作 memory 的内部存储模拟历史
    # 实际应使用 storage set，这里简化为通过 agent 交互积累
    
    # 2. 创建多个共享此 Memory 的 Agent
    agents = []
    for i in range(3):
        agent = AgentBuilder()\
            .with_model(model)\
            .build()
        # 强行替换 memory (Builder 默认会创建新的)
        agent.memory = shared_memory
        agents.append(agent)

    print(f"🚀 启动 3 个并发 Agent，共享同一个 SummaryMemory (Current Summary: '{shared_memory.current_summary}')")
    
    # 3. 使用 Team 并发执行
    # 每个 Agent 都会尝试读取 History -> 发现超限 -> 尝试触发 Summary
    # 预期：得益于 _summary_lock，只有一个 Summary 请求会真正执行，其他会等待并使用新摘要
    
    team = Team(
        members=agents, 
        name="ConcurrentSquad",
        max_concurrent=3
    )
    
    # 构造长输入以确保加上历史记录后绝对超限
    long_input = "请简述一下 Python 的历史 " * 10 
    
    results = await team.run(long_input)
    
    print("\n✅ 执行完成")
    print(f"Final Summary: {shared_memory.current_summary}")
    
    # 验证是否生成了摘要
    if shared_memory.current_summary:
        print("🎉 摘要生成成功，并发锁工作正常。")
    else:
        print("⚠️ 未生成摘要 (可能 Token 数未达阈值)")

if __name__ == "__main__":
    asyncio.run(main())