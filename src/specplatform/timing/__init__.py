"""timing 出口。

timing 层负责真实耗时 span、事件转换和归因 summary，不参与 token 接受决策。
"""

from specplatform.timing.attribution import TimingAttributor
from specplatform.timing.events import attribution_event_from_span, event_from_span, phase_category
from specplatform.timing.recorder import TimingRecorder
from specplatform.timing.span import TimingSpan
from specplatform.timing.summary import SUMMARY_VIEWS, TimingSummaryRow, summarize_timing_events

__all__ = [
    "SUMMARY_VIEWS",
    "TimingAttributor",
    "TimingRecorder",
    "TimingSpan",
    "TimingSummaryRow",
    "attribution_event_from_span",
    "event_from_span",
    "phase_category",
    "summarize_timing_events",
]
