from __future__ import annotations

"""metrics summary 辅助函数。"""

from collections import defaultdict

from specplatform.core import PhaseEvent
from specplatform.timing.summary import TimingSummaryRow, summarize_timing_events


def summarize_events(events: list[PhaseEvent]) -> dict[str, dict[str, float]]:
    """按 method/phase 汇总 system leaf measured duration。"""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        if event.event_scope == "system" and event.span_kind == "leaf":
            totals[event.method][event.phase] += float(event.measured_duration_ms or 0.0)
    return {method: dict(phases) for method, phases in totals.items()}


def summarize_phase_views(events: list[PhaseEvent]) -> list[TimingSummaryRow]:
    """复用 timing summary 的多视图聚合。"""
    return summarize_timing_events(events)


def format_summary_markdown(summary: dict[str, dict[str, float]]) -> str:
    """把简单 summary 格式化成 Markdown 表格。"""
    lines = [
        "| method | phase | duration_ms |",
        "|---|---:|---:|",
    ]
    for method in sorted(summary):
        for phase, duration in sorted(summary[method].items()):
            lines.append(f"| {method} | {phase} | {duration:.3f} |")
    return "\n".join(lines)
