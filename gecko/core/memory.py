# gecko/core/memory.py
from typing import List, Optional
import tiktoken
from gecko.core.message import Message
from gecko.plugins.storage.interfaces import SessionInterface

class TokenMemory:
    """
    基于 Token 计数的记忆管理器
    替代原有的 naive sliding window (基于条数)
    """
    def __init__(
        self, 
        session_id: str,
        storage: Optional[SessionInterface] = None,
        max_tokens: int = 4000,
        model_name: str = "gpt-3.5-turbo"
    ):
        self.session_id = session_id
        self.storage = storage
        self.max_tokens = max_tokens
        self.model_name = model_name
        self._encoding = None

    @property
    def tokenizer(self):
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, message: Message) -> int:
        """计算单条消息的 Token 数 (简化估算)"""
        # 实际模型计算会更复杂，这里做基础估算
        content = str(message.content)
        return self.count_tokens(content) + 4  # +4 for role/structure overhead

    async def get_history(self, raw_messages: List[dict]) -> List[Message]:
        """
        从原始数据中加载并修剪消息，确保不超 Token 上限
        策略：始终保留 System Prompt，然后保留最近的对话
        """
        if not raw_messages:
            return []

        messages = [Message(**m) for m in raw_messages]
        if not messages:
            return []

        # 1. 提取 System Prompt
        system_msg = None
        if messages[0].role == "system":
            system_msg = messages.pop(0)

        current_tokens = 0
        selected_msgs: List[Message] = []

        # 计算 System Token
        if system_msg:
            sys_tokens = self.count_message_tokens(system_msg)
            current_tokens += sys_tokens
            # 如果 System Prompt 即使单独放都超标，强制保留但警告（或截断）
            # 这里简单处理：system 必须保留

        # 2. 倒序选取历史消息
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens > self.max_tokens:
                break
            selected_msgs.append(msg)
            current_tokens += msg_tokens

        # 恢复顺序
        selected_msgs.reverse()
        
        if system_msg:
            selected_msgs.insert(0, system_msg)

        return selected_msgs