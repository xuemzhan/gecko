"""Gecko 异常定义模块。

该模块包含 Gecko 项目内部及 agent 运行时常用的异常类。
主要原则：尽量使用小而明确的异常类层次，便于在上层捕获并根据 error_id/ type 做不同的处理。

优化建议（已在此文件中做最小改动实现）：
- 为类与方法添加中文注释，便于中文开发者理解。
- 为异常的关键属性（type / error_id / message / status_code）统一命名，便于上层序列化。
- 增加 __all__ 明确导出接口。

后续可选改进：将部分异常转换为 dataclass（如果需要自动序列化），或统一继承自一个带序列化方法的基类。
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from models.message import Message


class AgentRunException(Exception):
    """代理运行期间发生错误时抛出的基类异常。

    这个异常用于封装运行时（agent/tool）出现的错误，携带可展示给用户或 agent 的消息。

    Attributes:
        user_message: 可展示给最终用户的消息（str 或 Message）。
        agent_message: 可展示/记录给 agent 的消息（str 或 Message）。
        messages: 可选的历史消息列表（dict 或 Message）。
        stop_execution: 是否需要终止后续执行（True 表示停止）。
        type: 错误类型标识（字符串）。
        error_id: 更具体的错误 id，便于上层路由或统计。
    """

    def __init__(
        self,
        exc: Any,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
        stop_execution: bool = False,
    ):
        # 保留原始异常信息作为 Exception 的消息
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution
        # type 与 error_id 用于统一上层处理逻辑
        self.type = "agent_run_error"
        self.error_id = "agent_run_error"


class RetryAgentRun(AgentRunException):
    """当某个工具调用需要重试时抛出的异常。

    表示这是一个可重试的错误，上层调度器可以捕获并再次尝试执行。
    """

    def __init__(
        self,
        exc: Any,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=False
        )
        self.error_id = "retry_agent_run_error"


class StopAgentRun(AgentRunException):
    """当 agent 需要完全停止执行时抛出的异常。

    例如发生不可恢复的错误或用户要求中止执行时使用。
    """

    def __init__(
        self,
        exc: Any,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=True
        )
        self.error_id = "stop_agent_run_error"


class RunCancelledException(Exception):
    """当运行被取消（例如用户取消）时抛出的异常。

    该异常较为上层（与 agent-specific 无关），拥有统一的 type 与 error_id。
    """

    def __init__(self, message: str = "Operation cancelled by user"):
        super().__init__(message)
        self.type = "run_cancelled_error"
        self.error_id = "run_cancelled_error"


class AgnoError(Exception):
    """AGNO 框架内部错误基类。

    Attributes:
        message: 错误消息文本。
        status_code: 推荐的 HTTP 状态码（用于 API 层返回）。
        type / error_id: 统一的错误标识字段，便于上层序列化与统计。
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = "agno_error"
        self.error_id = "agno_error"

    def __str__(self) -> str:
        return str(self.message)


class ModelProviderError(AgnoError):
    """当模型提供方返回错误（或调用模型失败）时抛出的异常。

    可携带模型相关的额外信息（model_name / model_id）。
    """

    def __init__(
        self, message: str, status_code: int = 502, model_name: Optional[str] = None, model_id: Optional[str] = None
    ):
        super().__init__(message, status_code)
        self.model_name = model_name
        self.model_id = model_id

        self.type = "model_provider_error"
        self.error_id = "model_provider_error"


class ModelRateLimitError(ModelProviderError):
    """当模型返回速率限制（rate limit）错误时使用的异常类型。"""

    def __init__(
        self, message: str, status_code: int = 429, model_name: Optional[str] = None, model_id: Optional[str] = None
    ):
        super().__init__(message, status_code, model_name, model_id)
        self.error_id = "model_rate_limit_error"


class EvalError(Exception):
    """评估（evaluation）失败时抛出的异常基类（可扩展）。"""

    pass


class CheckTrigger(Enum):
    """守护规则（guardrail）触发器枚举。

    用于标识触发检查器（input/output）的具体原因。
    """

    OFF_TOPIC = "off_topic"
    INPUT_NOT_ALLOWED = "input_not_allowed"
    OUTPUT_NOT_ALLOWED = "output_not_allowed"
    VALIDATION_FAILED = "validation_failed"

    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"


class InputCheckError(Exception):
    """当输入检查失败时抛出的异常。

    Attributes:
        check_trigger: CheckTrigger，指示失败类型（会被用作 error_id）。
        additional_data: 可选的额外数据，便于上层记录或调试。
    """

    def __init__(
        self,
        message: str,
        check_trigger: CheckTrigger = CheckTrigger.INPUT_NOT_ALLOWED,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.type = "input_check_error"
        if isinstance(check_trigger, CheckTrigger):
            self.error_id = check_trigger.value
        else:
            self.error_id = str(check_trigger)

        self.message = message
        self.check_trigger = check_trigger
        self.additional_data = additional_data


class OutputCheckError(Exception):
    """当输出检查失败时抛出的异常（与 InputCheckError 结构相同）。"""

    def __init__(
        self,
        message: str,
        check_trigger: CheckTrigger = CheckTrigger.OUTPUT_NOT_ALLOWED,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.type = "output_check_error"
        if isinstance(check_trigger, CheckTrigger):
            self.error_id = check_trigger.value
        else:
            self.error_id = str(check_trigger)

        self.message = message
        self.check_trigger = check_trigger
        self.additional_data = additional_data


# 明确导出符号，便于 from agno.agno.exceptions import * 时的可控性
__all__ = [
    "AgentRunException",
    "RetryAgentRun",
    "StopAgentRun",
    "RunCancelledException",
    "AgnoError",
    "ModelProviderError",
    "ModelRateLimitError",
    "EvalError",
    "CheckTrigger",
    "InputCheckError",
    "OutputCheckError",
]
