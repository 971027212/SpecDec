from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DraftBudget:
    max_tokens: int
    max_branches: int = 1
    timeout_ms: float | None = None


@dataclass(frozen=True)
class DraftJob:
    request_id: str
    worker_id: str
    budget: DraftBudget
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyBatch:
    batch_id: str
    request_ids: list[str]
    proposal_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanHints:
    draft_lengths: dict[str, int] = field(default_factory=dict)
    preferred_batches: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutablePlan:
    draft_jobs: list[DraftJob]
    verify_batches: list[VerifyBatch]
    hints: PlanHints | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
