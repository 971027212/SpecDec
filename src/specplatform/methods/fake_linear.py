from __future__ import annotations

from dataclasses import dataclass

from specplatform.core import (
    AcceptResult,
    CandidateProposal,
    DraftBudget,
    RuntimeContext,
    VerificationResult,
)
from specplatform.methods.base import AcceptancePolicy, CandidateStrategy
from specplatform.model import ModelForwardInput


@dataclass
class FakeLinearCandidateStrategy(CandidateStrategy):
    method_name: str = "fake_linear"

    def propose(
        self,
        session,
        draft_runner,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        tokens: list[int] = []
        forward_ms: list[float] = []
        intervals: list[dict[str, int]] = []
        cursor = session.prefix_ids[-1]
        max_tokens = min(max(1, budget.max_tokens), session.remaining_tokens)
        for offset in range(max_tokens):
            output = draft_runner.forward(
                ModelForwardInput(
                    input_ids=[cursor],
                    position_ids=[len(session.prefix_ids) + offset - 1],
                    metadata={"method": self.method_name, "request_id": session.request_id},
                )
            )
            token_id = _argmax(output.logits[0])
            tokens.append(token_id)
            forward_ms.append(output.timing_ms)
            if output.start_ns is not None and output.end_ns is not None:
                intervals.append({"start_ns": output.start_ns, "end_ns": output.end_ns})
            cursor = token_id
        timing = {
            "draft_generate": sum(forward_ms),
            "forward": forward_ms,
        }
        if intervals:
            timing["forward_intervals_ns"] = intervals
            timing["start_ns"] = min(item["start_ns"] for item in intervals)
            timing["end_ns"] = max(item["end_ns"] for item in intervals)
        return CandidateProposal(
            proposal_id=f"{session.request_id}:step{session.step_idx}:draft0",
            request_id=session.request_id,
            worker_id=getattr(draft_runner, "runner_id", None),
            shape="linear",
            tokens=tokens,
            draft_length=len(tokens),
            timing=timing,
            metadata={
                "method": self.method_name,
                "prefix_ids": list(session.prefix_ids),
            },
        )


@dataclass
class LinearPrefixAcceptancePolicy(AcceptancePolicy):
    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        accepted_count = int(verification_result.accepted_prefix_len or 0)
        accepted = proposal.tokens[:accepted_count]
        rejected = proposal.tokens[accepted_count:]
        return AcceptResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            bonus_token=verification_result.bonus_token,
            stop_reason=None,
            timing=dict(verification_result.timing),
            metadata={
                "method": proposal.metadata.get("method", "fake_linear"),
                "accepted_prefix_len": accepted_count,
            },
        )


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])
