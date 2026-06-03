from __future__ import annotations

"""Composable multi-request scheduling policies.

These policies are deliberately method-agnostic.  SpecEdge, DiP-SD and SLED
can provide hints, but the runtime-facing scheduler still only emits
``ExecutablePlan`` objects.
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import DraftBudget, PlanHints, RuntimeContext


@dataclass(frozen=True)
class RequestState:
    """Read-only scheduler view of one active request."""

    request_id: str
    remaining_tokens: int | None = None
    step_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestPool:
    """A stable per-round view over active sessions."""

    requests: tuple[RequestState, ...]
    sessions_by_id: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sessions(cls, sessions: list[Any]) -> "RequestPool":
        states: list[RequestState] = []
        sessions_by_id: dict[str, Any] = {}
        for session in sessions:
            request_id = str(session.request_id)
            sessions_by_id[request_id] = session
            states.append(
                RequestState(
                    request_id=request_id,
                    remaining_tokens=getattr(session, "remaining_tokens", None),
                    step_idx=getattr(session, "step_idx", None),
                    metadata={
                        "prompt_len": len(getattr(session, "prompt_ids", []) or []),
                        "generated_len": len(getattr(session, "generated_ids", []) or []),
                    },
                )
            )
        return cls(requests=tuple(states), sessions_by_id=sessions_by_id)

    @property
    def request_ids(self) -> list[str]:
        return [request.request_id for request in self.requests]

    def sessions(self) -> list[Any]:
        return [
            self.sessions_by_id[request.request_id]
            for request in self.requests
            if request.request_id in self.sessions_by_id
        ]

    def contains(self, request_id: str) -> bool:
        return request_id in self.sessions_by_id


class DraftLengthPolicy:
    """Choose per-request draft length for a scheduler round."""

    def draft_lengths(
        self,
        pool: RequestPool,
        *,
        default_budget: DraftBudget,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> dict[str, int]:
        raise NotImplementedError


@dataclass
class HintAwareDraftLengthPolicy(DraftLengthPolicy):
    """Use method hints when present, otherwise fall back to a fixed budget."""

    def draft_lengths(
        self,
        pool: RequestPool,
        *,
        default_budget: DraftBudget,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> dict[str, int]:
        del context
        lengths: dict[str, int] = {}
        for request in pool.requests:
            length = hints.draft_lengths.get(request.request_id, default_budget.max_tokens)
            if request.remaining_tokens is not None:
                length = min(int(length), max(0, int(request.remaining_tokens)))
            lengths[request.request_id] = max(0, int(length))
        return lengths


@dataclass
class FixedDraftLengthPolicy(DraftLengthPolicy):
    """Force one draft length for every active request."""

    max_tokens: int

    def draft_lengths(
        self,
        pool: RequestPool,
        *,
        default_budget: DraftBudget,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> dict[str, int]:
        del default_budget, hints, context
        return {
            request.request_id: max(0, min(int(self.max_tokens), int(request.remaining_tokens)))
            if request.remaining_tokens is not None
            else max(0, int(self.max_tokens))
            for request in pool.requests
        }


class BatchAssignmentPolicy:
    """Assign active requests into verify batches."""

    def assign_batches(
        self,
        pool: RequestPool,
        *,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> list[list[str]]:
        raise NotImplementedError


@dataclass
class PreferredBatchAssignmentPolicy(BatchAssignmentPolicy):
    """Honor preferred_batches when valid, otherwise chunk by batch_size."""

    batch_size: int | None = None

    def assign_batches(
        self,
        pool: RequestPool,
        *,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> list[list[str]]:
        del context
        preferred = _preferred_batches(pool, hints)
        if preferred:
            return preferred
        request_ids = pool.request_ids
        size = self.batch_size or max(1, len(request_ids))
        return [request_ids[index : index + size] for index in range(0, len(request_ids), size)]


def _preferred_batches(pool: RequestPool, hints: PlanHints) -> list[list[str]]:
    active = set(pool.request_ids)
    return [
        [request_id for request_id in batch if request_id in active]
        for batch in hints.preferred_batches
        if any(request_id in active for request_id in batch)
    ]
