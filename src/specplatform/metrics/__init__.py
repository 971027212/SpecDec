"""metrics 出口。

metrics 层只记录事件和写 artifact，不参与 speculative decoding 的算法决策。
"""

from specplatform.metrics.artifacts import (
    write_phase_events_csv,
    write_phase_summary_csv,
    write_request_results_json,
    write_tree_snapshots_jsonl,
)
from specplatform.metrics.events import EventLogger
from specplatform.metrics.plots import (
    SINGLE_RESULT_CHART_NAMES,
    TIMING_CHART_MODES,
    build_timing_audit,
    write_timing_audit,
    write_timing_charts,
)
from specplatform.metrics.summary import format_summary_markdown, summarize_events, summarize_phase_views

__all__ = [
    "EventLogger",
    "SINGLE_RESULT_CHART_NAMES",
    "TIMING_CHART_MODES",
    "build_timing_audit",
    "format_summary_markdown",
    "summarize_events",
    "summarize_phase_views",
    "write_phase_events_csv",
    "write_phase_summary_csv",
    "write_request_results_json",
    "write_timing_audit",
    "write_timing_charts",
    "write_tree_snapshots_jsonl",
]
