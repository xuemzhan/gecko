# gecko/core/structure/repair_limiter.py
"""
LLM Repair 限流器

目的：
- 控制 StructureEngine 结构化修复（LLM repair）的调用频率
- 避免在生产环境出现“解析失败 -> repair 重试风暴 -> 成本失控”的问题

实现：
- 滑动窗口计数（window_seconds 内最多 max_calls 次）
- 超过后进入 cooldown_seconds 冷却期，冷却期内一律拒绝
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class RepairLimiterConfig:
    window_seconds: float = 60.0
    max_calls: int = 10
    cooldown_seconds: float = 30.0


class RepairLimiter:
    def __init__(self, cfg: RepairLimiterConfig) -> None:
        self.cfg = cfg
        # 记录每次 repair 调用的时间戳（滑动窗口用）
        self._calls: Deque[float] = deque()
        # 冷却截止时间戳（now < cooldown_until 时拒绝）
        self._cooldown_until: float = 0.0

    def allow(self, now: float | None = None) -> bool:
        """
        判断是否允许触发 repair。
        - True：允许
        - False：拒绝（可能处于冷却期，或窗口内次数已满）

        now 参数便于测试（可注入虚拟时间）。
        """
        if now is None:
            now = time.time()

        # 1) 冷却期直接拒绝
        if now < self._cooldown_until:
            return False

        # 2) 清理滑动窗口外的调用记录
        window = self.cfg.window_seconds
        while self._calls and (now - self._calls[0]) > window:
            self._calls.popleft()

        # 3) 超过窗口配额：进入冷却，并拒绝本次
        if len(self._calls) >= self.cfg.max_calls:
            self._cooldown_until = now + self.cfg.cooldown_seconds
            return False

        # 4) 允许：记录本次调用时间戳
        self._calls.append(now)
        return True

    def status(self, now: float | None = None) -> dict:
        """用于日志/调试的状态输出。"""
        if now is None:
            now = time.time()
        return {
            "window_seconds": self.cfg.window_seconds,
            "max_calls": self.cfg.max_calls,
            "cooldown_seconds": self.cfg.cooldown_seconds,
            "cooldown_remaining": max(0.0, self._cooldown_until - now),
            "calls_in_window": len(self._calls),
        }
