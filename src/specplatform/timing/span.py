from __future__ import annotations

"""最小计时 span 数据结构。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingSpan:
    """一次可测量操作的起止时间和关联上下文。"""

    span_id: str
    phase: str
    method: str
    plan_id: str
    start_ns: int
    end_ns: int | None = None
    run_id: str | None = None
    round_id: int | None = None
    request_id: str | None = None
    session_id: str | None = None
    worker_id: str | None = None
    batch_id: str | None = None
    proposal_id: str | None = None
    shared: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_duration_ms(self) -> float:
        """返回真实测量耗时；未结束或时间倒退时抛错。"""
        if self.end_ns is None:
            raise ValueError("TimingSpan has not ended.")
        if self.end_ns < self.start_ns:
            raise ValueError("TimingSpan end_ns cannot be earlier than start_ns.")
        return (self.end_ns - self.start_ns) / 1_000_000

    def finish(self, end_ns: int) -> "TimingSpan":
        """结束 span，并立即触发 duration 校验。"""
        self.end_ns = int(end_ns)
        _ = self.measured_duration_ms
        return self
