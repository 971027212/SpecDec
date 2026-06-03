from __future__ import annotations

"""request scheduler。

scheduler 只产出 ExecutablePlan：draft job 分配和 verify batch 切分。
它不调用 draft runner、不调用 verifier，也不修改 GenerationSession。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import DraftBudget, DraftJob, ExecutablePlan, PlanHints, RuntimeContext, VerifyBatch
from specplatform.schedulers.policies import (
    BatchAssignmentPolicy,
    DraftLengthPolicy,
    HintAwareDraftLengthPolicy,
    PreferredBatchAssignmentPolicy,
    RequestPool,
)


@dataclass(frozen=True)
class SchedulerResources:
    """runtime 提供给 scheduler 的可用资源视图。"""

    draft_worker_ids: list[str]
    draft_worker_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


class Scheduler:
    """scheduler 抽象接口。"""

    def plan(
        self,
        active_sessions: list[Any],
        resources: SchedulerResources,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> ExecutablePlan:
        """根据 active sessions 和资源生成一轮可执行计划。"""
        raise NotImplementedError


@dataclass
class RoundRobinRequestScheduler(Scheduler):
    """把 request 轮询分配到 draft workers，并按 batch_size 组成 verify batch。"""

    default_budget: DraftBudget = field(default_factory=lambda: DraftBudget(max_tokens=1))
    batch_size: int | None = None
    cursor: int = 0
    draft_length_policy: DraftLengthPolicy = field(default_factory=HintAwareDraftLengthPolicy)
    batch_assignment_policy: BatchAssignmentPolicy | None = None

    def __post_init__(self) -> None:
        if self.batch_assignment_policy is None:
            self.batch_assignment_policy = PreferredBatchAssignmentPolicy(batch_size=self.batch_size)

    def plan(
        self,
        active_sessions: list[Any],
        resources: SchedulerResources,
        hints: PlanHints,
        context: RuntimeContext,
    ) -> ExecutablePlan:
        """生成本轮 draft jobs 和 verify batches。"""
        worker_ids = list(resources.draft_worker_ids)
        if not worker_ids:
            raise ValueError("SchedulerResources requires at least one draft worker.")
        pool = RequestPool.from_sessions(active_sessions)
        lengths = self.draft_length_policy.draft_lengths(
            pool,
            default_budget=self.default_budget,
            hints=hints,
            context=context,
        )
        candidate_lengths = _candidate_draft_lengths(hints)
        jobs: list[DraftJob] = []
        for session in pool.sessions():
            candidate_workers = self._candidate_workers(session.request_id, hints, worker_ids)
            if not candidate_workers:
                candidate_workers = [self._single_worker(session.request_id, hints, worker_ids)]
            request_max_tokens = lengths.get(session.request_id, self.default_budget.max_tokens)
            candidate_count = len(candidate_workers)
            for candidate_index, worker_id in enumerate(candidate_workers):
                max_tokens = candidate_lengths.get(session.request_id, {}).get(worker_id, request_max_tokens)
                jobs.append(
                    DraftJob(
                        request_id=session.request_id,
                        worker_id=worker_id,
                        budget=DraftBudget(
                            max_tokens=max_tokens,
                            max_branches=self.default_budget.max_branches,
                            timeout_ms=self.default_budget.timeout_ms,
                        ),
                        metadata={
                            "scheduler": "round_robin_request",
                            "draft_length_policy": type(self.draft_length_policy).__name__,
                            "candidate_index": candidate_index,
                            "candidate_count": candidate_count,
                            "candidate_group_id": f"{session.request_id}:round-candidates",
                        },
                    )
                )
        assert self.batch_assignment_policy is not None
        batches = self.batch_assignment_policy.assign_batches(
            pool,
            hints=hints,
            context=context,
        )
        batch_metadata = _preferred_batch_metadata(hints)
        return ExecutablePlan(
            draft_jobs=jobs,
            verify_batches=[
                VerifyBatch(
                    batch_id=f"batch{index}",
                    request_ids=list(request_ids),
                    metadata=dict(batch_metadata[index]) if index < len(batch_metadata) else {},
                )
                for index, request_ids in enumerate(batches)
            ],
            hints=hints,
            metadata={
                "scheduler": "round_robin_request",
                "request_pool_size": len(pool.requests),
                "draft_length_policy": type(self.draft_length_policy).__name__,
                "batch_assignment_policy": type(self.batch_assignment_policy).__name__,
            },
        )

    def _single_worker(self, request_id: str, hints: PlanHints, worker_ids: list[str]) -> str:
        preferred_worker = hints.worker_preferences.get(request_id)
        if preferred_worker in worker_ids:
            return preferred_worker
        worker_id = worker_ids[self.cursor % len(worker_ids)]
        self.cursor += 1
        return worker_id

    @staticmethod
    def _candidate_workers(request_id: str, hints: PlanHints, worker_ids: list[str]) -> list[str]:
        requested = hints.candidate_worker_preferences.get(request_id, [])
        seen: set[str] = set()
        candidates: list[str] = []
        for worker_id in requested:
            worker_id = str(worker_id)
            if worker_id in worker_ids and worker_id not in seen:
                seen.add(worker_id)
                candidates.append(worker_id)
        return candidates


def _candidate_draft_lengths(hints: PlanHints) -> dict[str, dict[str, int]]:
    return {
        str(request_id): {
            str(worker_id): max(0, int(length))
            for worker_id, length in dict(lengths or {}).items()
        }
        for request_id, lengths in dict(getattr(hints, "candidate_draft_lengths", {}) or {}).items()
    }


def _preferred_batch_metadata(hints: PlanHints) -> list[dict[str, Any]]:
    raw = dict(getattr(hints, "metadata", {}) or {}).get("preferred_batch_metadata", [])
    if not isinstance(raw, list):
        return []
    return [
        dict(item)
        for item in raw
        if isinstance(item, dict)
    ]
