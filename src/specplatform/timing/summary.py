from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from specplatform.core import PhaseEvent


SUMMARY_VIEWS = (
    "system_leaf_summary",
    "system_aggregate_summary",
    "request_attributed_summary",
    "debug_summary",
)


@dataclass(frozen=True)
class TimingSummaryRow:
    summary_view: str
    method: str
    phase: str
    phase_category: str
    event_scope: str
    span_kind: str
    count: int
    total_measured_duration_ms: float
    total_attributed_duration_ms: float

    @property
    def mean_measured_duration_ms(self) -> float:
        return self.total_measured_duration_ms / self.count if self.count else 0.0

    @property
    def mean_attributed_duration_ms(self) -> float:
        return self.total_attributed_duration_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "summary_view": self.summary_view,
            "method": self.method,
            "phase": self.phase,
            "phase_category": self.phase_category,
            "event_scope": self.event_scope,
            "span_kind": self.span_kind,
            "count": self.count,
            "total_measured_duration_ms": self.total_measured_duration_ms,
            "total_attributed_duration_ms": self.total_attributed_duration_ms,
            "mean_measured_duration_ms": self.mean_measured_duration_ms,
            "mean_attributed_duration_ms": self.mean_attributed_duration_ms,
        }


def summarize_timing_events(events: list[PhaseEvent]) -> list[TimingSummaryRow]:
    rows: list[TimingSummaryRow] = []
    rows.extend(_summarize_view(events, "system_leaf_summary", event_scope="system", span_kind="leaf"))
    rows.extend(
        _summarize_view(
            events,
            "system_aggregate_summary",
            event_scope="system",
            span_kind="aggregate",
        )
    )
    rows.extend(
        _summarize_view(
            events,
            "request_attributed_summary",
            event_scope="request",
            span_kind="attribution",
        )
    )
    rows.extend(_summarize_debug(events))
    return rows


def _summarize_view(
    events: list[PhaseEvent],
    summary_view: str,
    *,
    event_scope: str,
    span_kind: str,
) -> list[TimingSummaryRow]:
    selected = [
        event
        for event in events
        if event.event_scope == event_scope and event.span_kind == span_kind
    ]
    return _group_events(selected, summary_view)


def _summarize_debug(events: list[PhaseEvent]) -> list[TimingSummaryRow]:
    return _group_events(events, "debug_summary")


def _group_events(events: list[PhaseEvent], summary_view: str) -> list[TimingSummaryRow]:
    grouped: dict[tuple[str, str, str, str, str], list[PhaseEvent]] = defaultdict(list)
    for event in events:
        grouped[
            (
                event.method,
                event.phase,
                event.phase_category,
                event.event_scope,
                event.span_kind,
            )
        ].append(event)
    rows: list[TimingSummaryRow] = []
    for (method, phase, category, scope, kind), group in sorted(grouped.items()):
        rows.append(
            TimingSummaryRow(
                summary_view=summary_view,
                method=method,
                phase=phase,
                phase_category=category,
                event_scope=scope,
                span_kind=kind,
                count=len(group),
                total_measured_duration_ms=sum(
                    float(event.measured_duration_ms or 0.0) for event in group
                ),
                total_attributed_duration_ms=sum(
                    float(event.attributed_duration_ms) for event in group
                ),
            )
        )
    return rows
