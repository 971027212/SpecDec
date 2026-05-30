from __future__ import annotations

"""verification 和 acceptance 的结果数据模型。

VerificationResult 表示 target/verifier 看到的事实；AcceptResult 表示
acceptance policy 对这些事实作出的接受/拒绝决定。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core.candidate import ProposalShape


@dataclass(frozen=True)
class VerificationResult:
    """verifier 对一个 CandidateProposal 的验证输出。

    verifier 只描述 accepted_prefix_len、verified_tokens、bonus_token 等信息，
    不直接修改 session，也不决定最终写回哪些 token。
    """

    request_id: str
    proposal_id: str
    shape: ProposalShape
    accepted_prefix_len: int | None = None
    verified_tokens: list[int] | None = None
    bonus_token: int | None = None
    logits: Any | None = None
    timing: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptResult:
    """acceptance policy 的输出。

    accepted_tokens 会写回 session；rejected_tokens 仅用于记录/分析；
    bonus_token 表示 target 在验证后额外给出的下一个 token。
    """

    request_id: str
    proposal_id: str
    accepted_tokens: list[int] = field(default_factory=list)
    rejected_tokens: list[int] = field(default_factory=list)
    bonus_token: int | None = None
    stop_reason: str | None = None
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_token_ids(self) -> list[int]:
        """按写回顺序合并 accepted tokens 和可选 bonus token。"""
        output = list(self.accepted_tokens)
        if self.bonus_token is not None:
            output.append(int(self.bonus_token))
        return output
