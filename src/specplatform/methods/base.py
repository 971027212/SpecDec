from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from specplatform.core import AcceptResult, CandidateProposal, DraftBudget, PlanHints, RuntimeContext, VerificationResult


class CandidateStrategy(ABC):
    @abstractmethod
    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        raise NotImplementedError


class AcceptancePolicy(ABC):
    @abstractmethod
    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        raise NotImplementedError


class PlanningPolicy(ABC):
    def plan(
        self,
        active_sessions: list[Any],
        resources: Any,
        history: Any,
        context: RuntimeContext,
    ) -> PlanHints:
        return PlanHints()
