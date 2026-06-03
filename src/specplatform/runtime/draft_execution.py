from __future__ import annotations

"""Draft job execution helpers shared by sync and async runtimes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DraftExecutionResult:
    """Completed draft job with externally measured timing."""

    job: Any
    session: Any
    proposal: Any
    start_ns: int
    end_ns: int
    parallelism: int


@dataclass(frozen=True)
class ProactiveExecutionResult:
    """Completed proactive draft attempt with externally measured timing."""

    proposal: Any
    session: Any
    runner_id: str
    proactive: Any
    start_ns: int
    end_ns: int
    parallelism: int


def execute_draft_jobs(
    *,
    jobs: list[Any],
    sessions_by_id: dict[str, Any],
    draft_runners: dict[str, Any],
    candidate_strategy: Any,
    context: Any,
    clock: Callable[[], int],
    max_workers: int | None = None,
) -> list[DraftExecutionResult]:
    """Execute independent draft jobs, preserving scheduler order in results."""
    if not jobs:
        return []
    batch_proposer = getattr(candidate_strategy, "propose_batch", None)
    if callable(batch_proposer) and max_workers is None:
        parallelism = _resolved_draft_parallelism(
            draft_runners=draft_runners,
            jobs=jobs,
            context=context,
            requested=max_workers,
        )
        start_ns = clock()
        proposals = batch_proposer(
            jobs=list(jobs),
            sessions_by_id=sessions_by_id,
            draft_runners=draft_runners,
            context=context,
        )
        end_ns = clock()
        if len(proposals) != len(jobs):
            raise ValueError("CandidateStrategy.propose_batch returned a different number of proposals.")
        return [
            DraftExecutionResult(
                job=job,
                session=sessions_by_id[job.request_id],
                proposal=proposal,
                start_ns=start_ns,
                end_ns=end_ns,
                parallelism=parallelism,
            )
            for job, proposal in zip(jobs, proposals)
        ]
    parallelism = _resolved_draft_parallelism(
        draft_runners=draft_runners,
        jobs=jobs,
        context=context,
        requested=max_workers,
    )

    def run_one(job: Any) -> DraftExecutionResult:
        session = sessions_by_id[job.request_id]
        runner = draft_runners[job.worker_id]
        start_ns = clock()
        proposal = candidate_strategy.propose(session, runner, job.budget, context)
        end_ns = clock()
        return DraftExecutionResult(
            job=job,
            session=session,
            proposal=proposal,
            start_ns=start_ns,
            end_ns=end_ns,
            parallelism=parallelism,
        )

    if parallelism <= 1 or len(jobs) <= 1:
        return [run_one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        return [future.result() for future in futures]


def execute_proactive_jobs(
    *,
    proposals: list[Any],
    sessions_by_id: dict[str, Any],
    draft_runners: dict[str, Any],
    proactive_policy: Any,
    context: Any,
    clock: Callable[[], int],
    max_workers: int | None = None,
) -> list[ProactiveExecutionResult]:
    """Execute independent proactive draft attempts, preserving proposal order."""
    if not proposals:
        return []
    parallelism = _resolved_proactive_parallelism(
        draft_runners=draft_runners,
        proposals=proposals,
        context=context,
        requested=max_workers,
    )

    def run_one(proposal: Any) -> ProactiveExecutionResult:
        session = sessions_by_id[proposal.request_id]
        runner_id = str(proposal.worker_id or next(iter(draft_runners)))
        runner = draft_runners[runner_id]
        start_ns = clock()
        proactive = proactive_policy.propose_proactive(session, proposal, runner, context)
        end_ns = clock()
        return ProactiveExecutionResult(
            proposal=proposal,
            session=session,
            runner_id=runner_id,
            proactive=proactive,
            start_ns=start_ns,
            end_ns=end_ns,
            parallelism=parallelism,
        )

    if parallelism <= 1 or len(proposals) <= 1:
        return [run_one(proposal) for proposal in proposals]
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_one, proposal) for proposal in proposals]
        return [future.result() for future in futures]


def draft_parallelism_for(
    *,
    draft_runners: dict[str, Any],
    jobs: list[Any],
    context: Any,
    requested: int | None = None,
) -> int:
    """Expose the auto parallelism decision for tests and metadata."""
    return _resolved_draft_parallelism(
        draft_runners=draft_runners,
        jobs=jobs,
        context=context,
        requested=requested,
    )


def _resolved_draft_parallelism(
    *,
    draft_runners: dict[str, Any],
    jobs: list[Any],
    context: Any,
    requested: int | None,
) -> int:
    if not jobs:
        return 1
    configured = requested
    for source_name in ("method_config", "run_config"):
        source = getattr(context, source_name, {}) or {}
        if source.get("draft_parallelism") is not None:
            configured = int(source["draft_parallelism"])
            break
    if configured is not None and int(configured) > 0:
        return max(1, min(len(jobs), int(configured)))
    model_id_count = len(_distinct_runner_model_ids(draft_runners, jobs))
    return max(1, min(len(jobs), model_id_count))


def _resolved_proactive_parallelism(
    *,
    draft_runners: dict[str, Any],
    proposals: list[Any],
    context: Any,
    requested: int | None,
) -> int:
    if not proposals:
        return 1
    configured = requested
    for source_name in ("method_config", "run_config"):
        source = getattr(context, source_name, {}) or {}
        if source.get("proactive_parallelism") is not None:
            configured = int(source["proactive_parallelism"])
            break
        if source.get("draft_parallelism") is not None:
            configured = int(source["draft_parallelism"])
            break
    if configured is not None and int(configured) > 0:
        return max(1, min(len(proposals), int(configured)))
    model_id_count = len(_distinct_proposal_runner_model_ids(draft_runners, proposals))
    return max(1, min(len(proposals), model_id_count))


def _distinct_runner_model_ids(draft_runners: dict[str, Any], jobs: list[Any]) -> set[int]:
    model_ids: set[int] = set()
    for job in jobs:
        runner = draft_runners.get(job.worker_id)
        if runner is None:
            continue
        model = getattr(runner, "model", runner)
        model_ids.add(id(model))
    return model_ids


def _distinct_proposal_runner_model_ids(draft_runners: dict[str, Any], proposals: list[Any]) -> set[int]:
    model_ids: set[int] = set()
    for proposal in proposals:
        runner_id = str(proposal.worker_id or next(iter(draft_runners)))
        runner = draft_runners.get(runner_id)
        if runner is None:
            continue
        model = getattr(runner, "model", runner)
        model_ids.add(id(model))
    return model_ids
