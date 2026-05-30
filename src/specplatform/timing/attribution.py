from __future__ import annotations

"""共享耗时的 request 归因。

例如 batch verifier 只真实测量一次，但 summary 需要知道每个 request 分到多少
耗时；这里负责生成 attribution event。
"""

from collections.abc import Callable
from dataclasses import dataclass

from specplatform.core import CandidateProposal, PhaseEvent
from specplatform.timing.events import attribution_event_from_span
from specplatform.timing.span import TimingSpan


@dataclass
class TimingAttributor:
    """把 batch span 的 measured duration 分配给 proposal/request。"""

    def attribute_batch_average(
        self,
        *,
        parent_span: TimingSpan,
        proposals: list[CandidateProposal],
        event_id_factory: Callable[[], str],
    ) -> list[PhaseEvent]:
        """按 request 数平均分摊 batch verifier 耗时。"""
        if not proposals:
            return []
        attributed_ms = parent_span.measured_duration_ms / len(proposals)
        return [
            attribution_event_from_span(
                parent_span,
                event_id_factory=event_id_factory,
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                attributed_duration_ms=attributed_ms,
                tokens_in=len(proposal.tokens),
                metadata={
                    "request_ids": [item.request_id for item in proposals],
                    "proposal_ids": [item.proposal_id for item in proposals],
                    "attribution_policy": "batch_average",
                },
            )
            for proposal in proposals
        ]
