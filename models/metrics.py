# agno/models/metrics.py

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from utils.time_utils import Timer

@dataclass
class Metrics:
    """
    封装与会话、运行或消息相关的性能和成本指标。

    该数据类旨在聚合所有相关的指标，包括：
    - Token 消耗（输入、输出、总计）
    - 音频和缓存相关的 Token 使用情况
    - 时间指标（如总时长、首个 Token 时间）
    - 特定于供应商的原始指标
    """

    # 主要 Token 消耗值
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # 音频 Token 使用情况
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    audio_total_tokens: int = 0

    # 缓存 Token 使用情况
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # 用于推理的 Token
    reasoning_tokens: int = 0

    # 时间指标
    timer: Optional[Timer] = field(default=None, repr=False, compare=False) # 内部计时器，不在表示和比较中出现
    time_to_first_token: Optional[float] = None
    duration: Optional[float] = None

    # 特定于供应商的指标
    provider_metrics: Optional[Dict[str, Any]] = None

    # 其他任何附加指标
    additional_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        [重构] 将指标对象序列化为字典，同时过滤掉空值或零值。
        
        此方法旨在生成一个干净的、仅包含有意义指标的字典，便于日志记录或API返回。
        原先的单行字典推导式虽然简洁，但逻辑复杂，可读性较差。
        重构后的版本通过一个循环和清晰的条件判断提高了可读性和可维护性。
        """
        metrics_dict = asdict(self)
        metrics_dict.pop("timer", None)  # 移除内部计时器工具

        clean_dict = {}
        for key, value in metrics_dict.items():
            if value is None:
                continue  # 忽略 None 值
            if isinstance(value, (int, float)) and value == 0:
                continue  # 忽略数值零
            if isinstance(value, dict) and not value:
                continue  # 忽略空字典
            
            clean_dict[key] = value
            
        return clean_dict

    def __add__(self, other: "Metrics") -> "Metrics":
        """
        重载加法运算符，以合并两个 Metrics 实例。
        
        这使得可以方便地累加多个操作（如多个消息或工具调用）的指标。
        例如: total_metrics = message1.metrics + tool_call1.metrics
        """
        if not isinstance(other, Metrics):
            return NotImplemented

        # 创建一个新的 Metrics 实例来存储合并后的结果
        return Metrics(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            audio_total_tokens=self.audio_total_tokens + other.audio_total_tokens,
            audio_input_tokens=self.audio_input_tokens + other.audio_input_tokens,
            audio_output_tokens=self.audio_output_tokens + other.audio_output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            
            # 安全地合并字典类型的指标
            provider_metrics={**(self.provider_metrics or {}), **(other.provider_metrics or {})},
            additional_metrics={**(self.additional_metrics or {}), **(other.additional_metrics or {})},
            
            # 合并时间指标
            duration=(self.duration or 0) + (other.duration or 0),
            time_to_first_token=(self.time_to_first_token or 0) + (other.time_to_first_token or 0),
        )

    def __radd__(self, other: Any) -> "Metrics":
        """
        实现反向加法，以支持 sum() 内置函数。
        
        当 sum() 从一个空列表开始计算时 (sum(metrics_list, start=0))，
        `other` 会是 0。此方法确保在这种情况下能正确处理。
        """
        if other == 0:
            return self
        return self.__add__(other)

    def start_timer(self):
        """启动或重置内部计时器。"""
        if self.timer is None:
            self.timer = Timer()
        self.timer.start()

    def stop_timer(self, set_duration: bool = True):
        """停止计时器并可选择性地更新 duration 字段。"""
        if self.timer is not None:
            self.timer.stop()
            if set_duration:
                self.duration = self.timer.elapsed

    def set_time_to_first_token(self):
        """记录从计时开始到当前的时间点，作为首个 Token 的生成时间。"""
        if self.timer is not None and self.time_to_first_token is None:
            self.time_to_first_token = self.timer.elapsed