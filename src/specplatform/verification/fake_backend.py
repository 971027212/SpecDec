from __future__ import annotations

"""Phase 1 fake verifier backend。

这个 verifier 只提供稳定的假验证结果，用来测试 runtime/acceptance 边界。
它不是真实 target model，也不决定最终 session append。
"""

from dataclasses import dataclass, field

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult
from specplatform.verification.base import VerifierBackend


@dataclass
class FakeProposalVerifier(VerifierBackend):
    """对 linear proposal 做最小假验证的 backend。"""

    backend_name: str = "fake_proposal"
    bonus_offset: int = 1
    vocab_size: int = 16
    timing: dict[str, float] = field(
        default_factory=lambda: {
            "target_forward": 0.2,
            "verify_total": 0.25,
            "server_forward": 0.2,
            "server_wait": 0.0,
            "grpc_call": 0.0,
            "upload": 0.0,
            "downlink": 0.0,
        }
    )

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """验证一个 linear proposal，并返回 accepted_prefix_len 与 bonus token。"""
        if proposal.shape != "linear":
            raise NotImplementedError("FakeProposalVerifier only supports linear proposals.")
        accepted_prefix_len = 1 if proposal.tokens else 0
        cursor = proposal.tokens[accepted_prefix_len - 1] if accepted_prefix_len else _last_prefix_token(proposal)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=accepted_prefix_len,
            verified_tokens=list(proposal.tokens[:accepted_prefix_len]),
            bonus_token=self._next_token(cursor),
            timing=dict(self.timing),
            payload={"mode": "fake_proposal", "accepted_prefix_len": accepted_prefix_len},
            metadata={"backend": self.backend_name},
        )

    def _next_token(self, token_id: int) -> int:
        """用简单偏移生成 fake bonus token。"""
        return (int(token_id) + self.bonus_offset) % self.vocab_size


def _last_prefix_token(proposal: CandidateProposal) -> int:
    """从 proposal metadata 中取 prefix 末尾 token，作为空 proposal 的 fallback。"""
    prefix = proposal.metadata.get("prefix_ids")
    if isinstance(prefix, list) and prefix:
        return int(prefix[-1])
    return 0
