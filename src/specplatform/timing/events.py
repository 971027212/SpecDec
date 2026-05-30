from __future__ import annotations

from collections.abc import Callable

from specplatform.core import PhaseEvent
from specplatform.timing.span import TimingSpan


def phase_category(phase: str) -> str:
    prefix = phase.split(".", maxsplit=1)[0]
    known = {
        "runtime",
        "scheduler",
        "draft",
        "verify",
        "accept",
        "session",
        "request",
        "artifact",
    }
    return prefix if prefix in known else "runtime"


def event_from_span(
    span: TimingSpan,
    *,
    event_id_factory: Callable[[], str],
    span_kind: str = "leaf",
    attribution: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    metadata: dict | None = None,
) -> PhaseEvent:
    measured = span.measured_duration_ms
    return PhaseEvent(
        event_id=event_id_factory(),
        span_id=span.span_id,
        parent_span_id=None,
        run_id=span.run_id or "",
        request_id=span.request_id or span.batch_id or "",
        session_id=span.session_id,
        method=span.method,
        plan_id=span.plan_id,
        phase=span.phase,
        phase_category=phase_category(span.phase),
        event_scope="system",
        span_kind=span_kind,
        duration_ms=measured,
        measured_duration_ms=measured,
        attributed_duration_ms=measured,
        worker_id=span.worker_id,
        batch_id=span.batch_id,
        proposal_id=span.proposal_id,
        shared=span.shared,
        attribution=attribution,
        round=span.round_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        start_ns=span.start_ns,
        end_ns=span.end_ns,
        metadata={**dict(span.metadata), **dict(metadata or {})},
    )


def attribution_event_from_span(
    span: TimingSpan,
    *,
    event_id_factory: Callable[[], str],
    request_id: str,
    proposal_id: str | None,
    attributed_duration_ms: float,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    metadata: dict | None = None,
) -> PhaseEvent:
    measured = span.measured_duration_ms
    return PhaseEvent(
        event_id=event_id_factory(),
        span_id=span.span_id,
        parent_span_id=span.span_id,
        run_id=span.run_id or "",
        request_id=request_id,
        session_id=request_id,
        method=span.method,
        plan_id=span.plan_id,
        phase="verify.request_attributed",
        phase_category="verify",
        event_scope="request",
        span_kind="attribution",
        duration_ms=attributed_duration_ms,
        measured_duration_ms=measured,
        attributed_duration_ms=attributed_duration_ms,
        worker_id=span.worker_id,
        batch_id=span.batch_id,
        proposal_id=proposal_id,
        shared=False,
        attribution="request_average",
        round=span.round_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        start_ns=span.start_ns,
        end_ns=span.end_ns,
        metadata={**dict(span.metadata), **dict(metadata or {})},
    )
