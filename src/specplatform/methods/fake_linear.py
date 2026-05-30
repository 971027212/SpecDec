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
from specplatform.model import ModelForwardInput


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
        """按 draft budget 逐步调用 draft runner，拼出一条 linear proposal。"""
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


def _argmax(values: list[float]) -> int:
    """返回 logits 最大值所在的 token id。"""
    return max(range(len(values)), key=lambda index: values[index])
