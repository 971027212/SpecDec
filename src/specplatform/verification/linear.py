from __future__ import annotations

"""本地 linear target verifier。

verification 层只验证 proposal，并返回 VerificationResult。它不决定最终接受哪些 token，
也不修改 GenerationSession；这些动作留给 methods.acceptance 和 runtime。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult
from specplatform.model import CausalLMRunner
from specplatform.verification.base import VerifierBackend
from specplatform.verification.schema import LinearVerifyRequest, LinearVerifyResponse


@dataclass
class LinearVerifier(VerifierBackend):
    """用 target CausalLMRunner 对 linear draft tokens 做逐 token greedy 验证。"""

    model: CausalLMRunner
    backend_name: str = "linear_local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """验证单个 proposal；只返回 verifier 看到的事实。"""
        if proposal.shape != "linear":
            raise ValueError("LinearVerifier only supports linear proposals.")
        request = LinearVerifyRequest(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            prefix_ids=_proposal_prefix_ids(proposal),
            draft_tokens=list(proposal.tokens),
            eos_token_ids=_eos_token_ids(proposal, context),
            metadata=dict(proposal.metadata),
        )
        response = self.verify_request(request, context)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=response.accepted_prefix_len,
            verified_tokens=list(response.verified_tokens),
            bonus_token=response.bonus_token,
            payload=response.to_dict(),
            metadata={
                "backend_name": self.backend_name,
                **dict(self.metadata),
                **dict(response.metadata),
            },
        )

    def verify_request(
        self,
        request: LinearVerifyRequest,
        context: RuntimeContext | None = None,
    ) -> LinearVerifyResponse:
        """验证 HTTP/schema 层请求，供本地测试和 A100 service 共用。"""
        if not request.prefix_ids:
            raise ValueError("Linear verification requires a non-empty prefix.")

        accepted_prefix_len = 0
        verified_tokens: list[int] = []
        working_prefix = list(request.prefix_ids)
        eos_token_ids = set(request.eos_token_ids)

        for draft_token in request.draft_tokens:
            target_token = int(self.model.greedy_next_token(working_prefix))
            verified_tokens.append(target_token)
            if target_token != int(draft_token):
                # 第一个不匹配处：前缀之前都接受，target_token 作为纠偏 bonus 返回。
                return LinearVerifyResponse(
                    accepted_prefix_len=accepted_prefix_len,
                    verified_tokens=verified_tokens,
                    bonus_token=target_token,
                    metadata={"mismatch_at": accepted_prefix_len},
                )

            accepted_prefix_len += 1
            working_prefix.append(int(draft_token))
            if target_token in eos_token_ids:
                # draft 已经命中 EOS，不能再额外生成 bonus。
                return LinearVerifyResponse(
                    accepted_prefix_len=accepted_prefix_len,
                    verified_tokens=verified_tokens,
                    bonus_token=None,
                    metadata={"matched_eos": target_token},
                )

        bonus_token = int(self.model.greedy_next_token(working_prefix))
        if bonus_token in eos_token_ids:
            metadata = {"bonus_is_eos": bonus_token}
        else:
            metadata = {}
        return LinearVerifyResponse(
            accepted_prefix_len=accepted_prefix_len,
            verified_tokens=verified_tokens,
            bonus_token=bonus_token,
            metadata=metadata,
        )


def _proposal_prefix_ids(proposal: CandidateProposal) -> list[int]:
    """从 proposal metadata 取出 draft 前的 prefix。"""
    prefix_ids = proposal.metadata.get("prefix_ids")
    if prefix_ids is None:
        raise ValueError("Linear proposal metadata must include prefix_ids.")
    return [int(token_id) for token_id in prefix_ids]


def _eos_token_ids(proposal: CandidateProposal, context: RuntimeContext | None) -> list[int]:
    """按 proposal metadata -> context.method_config -> context.run_config 的优先级读取 EOS。"""
    raw = proposal.metadata.get("eos_token_ids")
    if raw is None and context is not None:
        raw = context.method_config.get("eos_token_ids") or context.run_config.get("eos_token_ids")
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    return [int(token_id) for token_id in raw]
