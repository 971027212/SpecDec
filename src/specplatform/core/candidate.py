from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from specplatform.core.types import CandidateTree


ProposalShape = Literal["linear", "tree"]


@dataclass(frozen=True)
class CandidateProposal:
    proposal_id: str
    request_id: str
    worker_id: str | None
    shape: ProposalShape
    tokens: list[int] = field(default_factory=list)
    tree: CandidateTree | None = None
    draft_length: int = 0
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape == "linear" and self.tree is not None:
            raise ValueError("linear proposals cannot carry a candidate tree.")
        if self.shape == "tree" and self.tree is None:
            raise ValueError("tree proposals require a candidate tree.")
