from __future__ import annotations

from abc import ABC, abstractmethod

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult


class VerifierBackend(ABC):
    backend_name: str

    @abstractmethod
    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        raise NotImplementedError

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        return [self.verify_proposal(proposal, context) for proposal in proposals]
