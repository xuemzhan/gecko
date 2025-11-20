# gecko/core/engine/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory

class CognitiveEngine(ABC):
    """
    认知引擎基类：定义 Agent 如何'思考'和'执行'
    """
    def __init__(self, model, toolbox: ToolBox, memory: TokenMemory):
        self.model = model
        self.toolbox = toolbox
        self.memory = memory

    @abstractmethod
    async def step(self, input_messages: List[Message]) -> AgentOutput:
        """
        执行单次或多轮推理步骤
        :param input_messages: 当前输入的消息（通常是 User Message）
        :return: 最终产出的 AgentOutput
        """
        pass