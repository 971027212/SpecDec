from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

from specplatform.timing.events import event_from_span
from specplatform.timing.span import TimingSpan


Clock = Callable[[], int]


@dataclass
class TimingRecorder:
    clock: Clock = time.perf_counter_ns
    spans: list[TimingSpan] = field(default_factory=list)
    _span_counter: int = 0
    _event_counter: int = 0

    def next_span_id(self) -> str:
        self._span_counter += 1
        return f"span_{self._span_counter:06d}"

    def next_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    @contextmanager
    def span(
        self,
        *,
        phase: str,
        method: str,
        plan_id: str,
        run_id: str | None = None,
        round_id: int | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        worker_id: str | None = None,
        batch_id: str | None = None,
        proposal_id: str | None = None,
        shared: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TimingSpan]:
        timing_span = TimingSpan(
            span_id=self.next_span_id(),
            phase=phase,
            method=method,
            plan_id=plan_id,
            start_ns=self.clock(),
            run_id=run_id,
            round_id=round_id,
            request_id=request_id,
            session_id=session_id,
            worker_id=worker_id,
            batch_id=batch_id,
            proposal_id=proposal_id,
            shared=shared,
            metadata=dict(metadata or {}),
        )
        try:
            yield timing_span
        finally:
            timing_span.finish(self.clock())
            self.spans.append(timing_span)

    def record_completed(
        self,
        *,
        phase: str,
        method: str,
        plan_id: str,
        start_ns: int,
        end_ns: int,
        run_id: str | None = None,
        round_id: int | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        worker_id: str | None = None,
        batch_id: str | None = None,
        proposal_id: str | None = None,
        shared: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> TimingSpan:
        timing_span = TimingSpan(
            span_id=self.next_span_id(),
            phase=phase,
            method=method,
            plan_id=plan_id,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            run_id=run_id,
            round_id=round_id,
            request_id=request_id,
            session_id=session_id,
            worker_id=worker_id,
            batch_id=batch_id,
            proposal_id=proposal_id,
            shared=shared,
            metadata=dict(metadata or {}),
        )
        _ = timing_span.measured_duration_ms
        self.spans.append(timing_span)
        return timing_span

    def to_system_events(self, *, span_kind_by_phase: dict[str, str] | None = None):
        mapping = span_kind_by_phase or {}
        return [
            event_from_span(
                span,
                event_id_factory=self.next_event_id,
                span_kind=mapping.get(span.phase, "leaf"),
            )
            for span in self.spans
        ]
