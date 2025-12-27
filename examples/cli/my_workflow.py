# my_workflow.py
"""
Gecko Workflow 定义文件
用于演示 `gecko run` 命令的动态加载能力。
"""
import os
from gecko import AgentBuilder, Workflow, step
from gecko.plugins.models import ZhipuChat

# 1. 获取 API Key (在 cli_demo.py 中会自动设置，或从系统环境变量读取)
# 注意：实际使用中建议优先从环境变量读取，这里为了演示方便直接取值
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 2. 定义节点逻辑
@step(name="zhipu_agent")
async def chat_with_zhipu(user_input: str):
    """
    这是一个 Workflow 节点。
    它初始化一个使用智谱 GLM-4 的 Agent，并处理输入。
    """
    if not ZHIPU_API_KEY:
        return "Error: ZHIPU_API_KEY not found in environment variables."

    # 使用 AgentBuilder 构建 Agent
    # 这里我们显式使用了 ZhipuChat 类，这是 gecko run 的强大之处：
    # 你可以在 Python 脚本中配置任何复杂的模型参数，而不仅受限于 CLI 参数
    agent = AgentBuilder()\
        .with_model(ZhipuChat(api_key=ZHIPU_API_KEY, model="glm-4-flash"))\
        .with_system_prompt("你是一个精通 Gecko 框架的 AI 助手，请用简洁的中文回答。")\
        .build()

    # 执行推理
    result = await agent.run(user_input)
    
    # 返回 AgentOutput 的文本内容
    return result.content # type: ignore

# 3. 构建工作流
# 必须定义一个名为 `workflow` 的变量，CLI 会寻找它
wf = Workflow(name="ZhipuDemoFlow")
wf.add_node("agent_node", chat_with_zhipu)
wf.set_entry_point("agent_node")

# 导出变量
workflow = wf