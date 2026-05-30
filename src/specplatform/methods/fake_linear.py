from __future__ import annotations

"""Phase 1 的 fake linear method。

这个 fake method 用线性 proposal 验证 runtime 边界：strategy 只生成候选，
acceptance policy 只消费验证结果，不直接访问验证后端。
"""

from dataclasses import dataclass

from specplatform.core import (
    AcceptResult,
    CandidateProposal,
    DraftBudget,
    RuntimeContext,
    VerificationResult,
)
from specplatform.methods.base import AcceptancePolicy, CandidateStrategy


@dataclass
class FakeLinearCandidateStrategy(CandidateStrategy):
    """使用 fake draft runner 生成线性 CandidateProposal。"""

    method_name: str = "fake_linear"

    def propose(
        self,
        session,
        draft_runner,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        """按 draft budget 向 draft runner 请求一条 linear proposal。"""
        max_tokens = min(max(1, budget.max_tokens), session.remaining_tokens)
        draft_generation = draft_runner.generate_tokens(
            prefix_ids=session.prefix_ids,
            max_tokens=max_tokens,
            request_id=session.request_id,
            metadata={"method": self.method_name},
        )
        timing = {
            "draft_generate": sum(draft_generation.forward_timing_ms),
            "forward": list(draft_generation.forward_timing_ms),
        }
        if draft_generation.forward_intervals_ns:
            timing["forward_intervals_ns"] = list(draft_generation.forward_intervals_ns)
            timing["start_ns"] = min(
                item["start_ns"] for item in draft_generation.forward_intervals_ns
            )
            timing["end_ns"] = max(
                item["end_ns"] for item in draft_generation.forward_intervals_ns
            )
        return CandidateProposal(
            proposal_id=f"{session.request_id}:step{session.step_idx}:draft0",
            request_id=session.request_id,
            worker_id=getattr(draft_runner, "runner_id", None),
            shape="linear",
            tokens=list(draft_generation.tokens),
            draft_length=len(draft_generation.tokens),
            timing=timing,
            metadata={
                "method": self.method_name,
                "prefix_ids": list(session.prefix_ids),
                "draft_budget_max_tokens": budget.max_tokens,
            },
        )


@dataclass
class LinearPrefixAcceptancePolicy(AcceptancePolicy):
    """根据 accepted_prefix_len 接受 proposal 的线性前缀。"""

    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        """只消费 VerificationResult，不访问验证后端或 runtime。"""
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
