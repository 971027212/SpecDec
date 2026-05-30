from __future__ import annotations

"""request scheduler。

scheduler 只产出 ExecutablePlan：draft job 分配和 verify batch 切分。
它不调用 draft runner、不调用 verifier，也不修改 GenerationSession。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import DraftBudget, DraftJob, ExecutablePlan, PlanHints, RuntimeContext, VerifyBatch


@dataclass(frozen=True)
class SchedulerResources:
    """runtime 提供给 scheduler 的可用资源视图。"""

    draft_worker_ids: list[str]


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
        jobs: list[DraftJob] = []
        for session in active_sessions:
            worker_id = worker_ids[self.cursor % len(worker_ids)]
            self.cursor += 1
            max_tokens = hints.draft_lengths.get(session.request_id, self.default_budget.max_tokens)
            jobs.append(
                DraftJob(
                    request_id=session.request_id,
                    worker_id=worker_id,
                    budget=DraftBudget(
                        max_tokens=max_tokens,
                        max_branches=self.default_budget.max_branches,
                        timeout_ms=self.default_budget.timeout_ms,
                    ),
                )
            )
        batches = _preferred_batches(active_sessions, hints)
        if not batches:
            size = self.batch_size or max(1, len(active_sessions))
            request_ids = [session.request_id for session in active_sessions]
            batches = [request_ids[index : index + size] for index in range(0, len(request_ids), size)]
        return ExecutablePlan(
            draft_jobs=jobs,
            verify_batches=[
                VerifyBatch(batch_id=f"batch{index}", request_ids=list(request_ids))
                for index, request_ids in enumerate(batches)
            ],
            hints=hints,
            metadata={"scheduler": "round_robin_request"},
        )


def _preferred_batches(active_sessions: list[Any], hints: PlanHints) -> list[list[str]]:
    """过滤 planning hints 中已不活跃的 request。"""
    active = {session.request_id for session in active_sessions}
    return [
        [request_id for request_id in batch if request_id in active]
        for batch in hints.preferred_batches
        if any(request_id in active for request_id in batch)
    ]
