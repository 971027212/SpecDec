from __future__ import annotations

"""method 层的策略接口。

这些接口让不同 speculative decoding 方法注入差异点，同时保持 runtime
不依赖具体方法名称。
"""

from abc import ABC, abstractmethod
from typing import Any

from dataclasses import dataclass, field

from specplatform.core import AcceptResult, CandidateProposal, DraftBudget, PlanHints, RuntimeContext, VerificationResult


class CandidateStrategy(ABC):
    """把 session + draft runner + budget 转成 CandidateProposal。"""

    @abstractmethod
    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        """生成候选 proposal；不调用 verifier，不写 session。"""
        raise NotImplementedError


class AcceptancePolicy(ABC):
    """把 VerificationResult 转成 AcceptResult 的策略接口。"""

    @abstractmethod
    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        """决定接受/拒绝哪些 token；不调用 verifier。"""
        raise NotImplementedError


class PlanningPolicy(ABC):
    """给 scheduler 提供 method-specific hint 的可选接口。"""

    def plan(
        self,
        active_sessions: list[Any],
        resources: Any,
        history: Any,
        context: RuntimeContext,
    ) -> PlanHints:
        """默认不给 scheduler 任何特殊提示。"""
        return PlanHints()


class ProactiveDraftPolicy(ABC):
    """verify in-flight 时可选的 proactive draft 策略。"""

    def propose_proactive(
        self,
        session: Any,
        proposal: CandidateProposal,
        draft_runner: Any,
        context: RuntimeContext,
    ) -> CandidateProposal | None:
        """默认不做 proactive draft。"""
        return None


@dataclass(frozen=True)
class ReconcileResult:
    """verify 返回后 proactive draft 的复用/丢弃结果。"""

    reused_proposal: CandidateProposal | None = None
    reused_token_count: int = 0
    discarded_token_count: int = 0
    aligned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ReconcilePolicy(ABC):
    """接受结果写回后，对 proactive draft 做对齐和复用。"""

    def reconcile(
        self,
        session: Any,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        accept_result: AcceptResult,
        proactive_proposal: CandidateProposal | None,
        context: RuntimeContext,
    ) -> ReconcileResult:
        """默认丢弃 proactive proposal。"""
        del session, proposal, verification_result, accept_result, context
        if proactive_proposal is None:
            return ReconcileResult()
        return ReconcileResult(
            discarded_token_count=int(proactive_proposal.draft_length),
            metadata={"reason": "no_reconcile_policy"},
        )
