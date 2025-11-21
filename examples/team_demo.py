# examples/team_demo.py
"""
Team 多智能体协作示例

展示 Gecko Team 引擎的并行处理能力：
1. 专家评审团模式 (Panel of Experts)
2. 并发控制 (Rate Limiting)
3. 容错机制 (Partial Failure Handling)
4. 结果聚合 (Aggregation)

运行前提：
    export ZHIPU_API_KEY="your_api_key"
"""
from __future__ import annotations

import asyncio
import os
from typing import List

from gecko.compose.team import Team
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.logging import get_logger
from gecko.plugins.models.zhipu import glm_4_5_air

logger = get_logger(__name__)


# ========================= 1. 辅助函数 =========================

def create_expert(role: str, prompt: str, api_key: str) -> Agent:
    """
    创建一个特定角色的专家 Agent
    """
    model = glm_4_5_air(api_key=api_key, temperature=0.8)
    
    return (
        AgentBuilder()
        .with_model(model)
        .with_session_id(f"expert_{role}")
        .with_system_prompt(f"你是一位{role}。{prompt} 请简短回答（50字以内）。")
        .build()
    )


async def aggregate_results(results: List[str]) -> str:
    """
    聚合函数：将团队的意见汇总
    """
    summary = []
    for i, res in enumerate(results, 1):
        # 处理可能的错误信息（Team 的容错机制）
        if str(res).startswith("Error:"):
            summary.append(f"专家 {i}: [缺席] ({res})")
        else:
            summary.append(f"专家 {i}: {res}")
            
    return "\n".join(summary)


# ========================= 2. 主流程 =========================

async def main():
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 ZHIPU_API_KEY")
        return

    logger.info("🚀 初始化专家评审团...")

    # 1. 组建团队
    # 定义三个不同视角的专家
    optimist = create_expert(
        "乐观主义未来学家", 
        "请对未来的 AI 发展给出一个极其乐观的预测。", 
        api_key
    )
    
    pessimist = create_expert(
        "悲观主义安全专家", 
        "请警告人类 AI 可能带来的最大生存风险。", 
        api_key
    )
    
    realist = create_expert(
        "务实工程师", 
        "请从技术落地角度评估未来 5 年 AI 的实际应用。", 
        api_key
    )

    # 2. 创建 Team 引擎
    # 设置 max_concurrent=2，演示流量整形（虽然有3个专家，但同一时间只并发请求2个）
    team = Team(
        members=[optimist, pessimist, realist],
        name="AI_Review_Board",
        max_concurrent=2
    )

    topic = "我们应该如何看待 AGI 的到来？"
    print(f"\n🎙️ 议题: {topic}\n")

    # 3. 并行执行
    # Team.run 会自动处理并发、等待所有结果、并捕获单个 Agent 的异常
    raw_results = await team.run(topic)

    # 4. 结果展示
    print("-" * 20 + " 评审结果 " + "-" * 20)
    final_report = await aggregate_results(raw_results)
    print(final_report)
    print("-" * 50)


if __name__ == "__main__":
    # 配置日志级别以便观察 Team 的并发执行日志
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())