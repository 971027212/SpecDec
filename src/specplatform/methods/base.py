from __future__ import annotations

"""method 层的策略接口。

这些接口让不同 speculative decoding 方法注入差异点，同时保持 runtime
不依赖具体方法名称。
"""

from abc import ABC, abstractmethod
from typing import Any

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
