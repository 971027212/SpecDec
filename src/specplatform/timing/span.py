from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingSpan:
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
        if self.end_ns is None:
            raise ValueError("TimingSpan has not ended.")
        if self.end_ns < self.start_ns:
            raise ValueError("TimingSpan end_ns cannot be earlier than start_ns.")
        return (self.end_ns - self.start_ns) / 1_000_000

    def finish(self, end_ns: int) -> "TimingSpan":
        self.end_ns = int(end_ns)
        _ = self.measured_duration_ms
        return self
