from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from specplatform.core.candidate import ProposalShape


@dataclass(frozen=True)
class VerificationResult:
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
        output = list(self.accepted_tokens)
        if self.bonus_token is not None:
            output.append(int(self.bonus_token))
        return output
