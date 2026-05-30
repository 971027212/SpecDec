from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from specplatform.core import CandidateProposal, PhaseEvent
from specplatform.timing.events import attribution_event_from_span
from specplatform.timing.span import TimingSpan


@dataclass
class TimingAttributor:
    def attribute_batch_average(
        self,
        *,
        parent_span: TimingSpan,
        proposals: list[CandidateProposal],
        event_id_factory: Callable[[], str],
    ) -> list[PhaseEvent]:
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
