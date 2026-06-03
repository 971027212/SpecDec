from __future__ import annotations

"""Generic distributed draft / central batch-verify pipeline runtime.

The engine models the execution pattern used by edge methods such as DiP-SD:
draft jobs run independently on edge devices, while a shared verifier processes
planned batches in stage order.  It is deliberately method-agnostic; method
behavior enters through planning, candidate and acceptance policies.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from specplatform.core import CandidateProposal, DraftBudget, DraftJob, PlanHints, RuntimeContext
from specplatform.methods.base import AcceptancePolicy, CandidateStrategy, PlanningPolicy
from specplatform.metrics import EventLogger
from specplatform.runtime.draft_execution import DraftExecutionResult, draft_parallelism_for
from specplatform.runtime.engine import (
    RuntimeRequestResult,
    RuntimeRunResult,
    _commit_acceptance_if_supported,
    _draft_runner_metadata,
    _group_proposals_by_request,
    _max_end_ns,
    _min_start_ns,
    _plan_metadata,
    _select_accept_record,
    _validate_verification_results,
)
from specplatform.runtime.session import GenerationSession
from specplatform.schedulers import Scheduler, SchedulerResources
from specplatform.timing import TimingAttributor, TimingRecorder, event_from_span
from specplatform.timing.span import TimingSpan
from specplatform.verification import VerifierBackend


@dataclass(frozen=True)
class _PrefetchedDraft:
    """A next-round draft already launched after its previous verify stage."""

    request_id: str
    worker_id: str
    budget_tokens: int
    prefix_ids: tuple[int, ...]
    step_idx: int
    future: Future[DraftExecutionResult]
    submitted_ns: int
    source_round: int
    source_stage: int
    source_batch_id: str | None


@dataclass
class DistributedBatchPipelineRuntimeEngine:
    """Run draft jobs in parallel and verify planned batches in stage order."""

    candidate_strategy: CandidateStrategy
    acceptance_policy: AcceptancePolicy
    scheduler: Scheduler
    verifier: VerifierBackend
    planning_policy: PlanningPolicy | None = None
    timing_recorder: TimingRecorder | None = None
    timing_attributor: TimingAttributor = field(default_factory=TimingAttributor)
    max_draft_workers: int | None = None

    def run(
        self,
        *,
        run_id: str,
        sessions: list[GenerationSession],
        draft_runners: dict[str, Any],
        context: RuntimeContext | None = None,
        max_rounds: int | None = None,
    ) -> RuntimeRunResult:
        context = context or RuntimeContext()
        logger = EventLogger()
        recorder = self.timing_recorder or TimingRecorder(clock=context.clock)
        if self.timing_recorder is None:
            self.timing_recorder = recorder
        method = str(context.run_config.get("method", "distributed_batch_pipeline"))
        request_results = {
            session.request_id: RuntimeRequestResult(request_id=session.request_id)
            for session in sessions
        }
        stalled: set[str] = set()
        prefetched_drafts: dict[str, _PrefetchedDraft] = {}
        planning_history: dict[str, Any] = {}
        round_index = 0
        while True:
            active_sessions = [
                session
                for session in sessions
                if not session.is_finished and session.request_id not in stalled
            ]
            if not active_sessions:
                break
            if max_rounds is not None and round_index >= max_rounds:
                break

            plan_id = f"{run_id}:round{round_index}"
            with recorder.span(
                phase="runtime.round_total",
                method=method,
                plan_id=plan_id,
                run_id=run_id,
                round_id=round_index,
                shared=True,
                metadata={"runtime_engine": "distributed_batch_pipeline"},
            ) as round_span:
                with recorder.span(
                    phase="planner.hints",
                    method=method,
                    plan_id=plan_id,
                    run_id=run_id,
                    round_id=round_index,
                    shared=True,
                    metadata={"runtime_engine": "distributed_batch_pipeline"},
                ) as planner_span:
                    planning_history["dip_sd_prefetch_by_request"] = _prefetched_draft_history(prefetched_drafts)
                    hints = self._plan_hints(active_sessions, draft_runners, context, history=planning_history)
                    planner_span.phase = _planner_phase(hints)
                    planner_span.metadata.update(_planner_metadata(hints))
                _record_span_event(logger, recorder, planner_span, span_kind="leaf", attribution="system")
                with recorder.span(
                    phase="scheduler.plan",
                    method=method,
                    plan_id=plan_id,
                    run_id=run_id,
                    round_id=round_index,
                    shared=True,
                ) as scheduler_span:
                    plan = self.scheduler.plan(
                        active_sessions=active_sessions,
                        resources=SchedulerResources(
                            draft_worker_ids=list(draft_runners),
                            draft_worker_metadata=_draft_runner_metadata(draft_runners),
                        ),
                        hints=hints,
                        context=context,
                    )
                    scheduler_span.metadata.update(_plan_metadata(plan, hints))
                _record_span_event(logger, recorder, scheduler_span, span_kind="leaf", attribution="system")
                planner_wait_end_ns = int(scheduler_span.end_ns or recorder.clock())
                planner_wait_span = recorder.record_completed(
                    phase="pipeline.planner_wait",
                    method=method,
                    plan_id=plan_id,
                    run_id=run_id,
                    round_id=round_index,
                    shared=True,
                    start_ns=int(round_span.start_ns),
                    end_ns=planner_wait_end_ns,
                    metadata={
                        "runtime_engine": "distributed_batch_pipeline",
                        "planner_phase": planner_span.phase,
                        "planner_ms": planner_span.measured_duration_ms,
                        "scheduler_ms": scheduler_span.measured_duration_ms,
                    },
                )
                _record_span_event(
                    logger,
                    recorder,
                    planner_wait_span,
                    span_kind="leaf",
                    attribution="system",
                    metadata=dict(planner_wait_span.metadata),
                )

                sessions_by_id = {session.request_id: session for session in active_sessions}
                proposals_by_request: dict[str, list[CandidateProposal]] = {}
                proposals_by_id: dict[str, CandidateProposal] = {}
                draft_spans_by_proposal: dict[str, TimingSpan] = {}
                futures_by_job_index, prefetch_hit_indices = self._submit_draft_jobs_with_prefetch(
                    plan.draft_jobs,
                    sessions_by_id=sessions_by_id,
                    draft_runners=draft_runners,
                    context=context,
                    clock=recorder.clock,
                    prefetched_drafts=prefetched_drafts,
                )
                draft_submit_end_ns = recorder.clock()
                if prefetch_hit_indices:
                    prefetch_hit_span = recorder.record_completed(
                        phase="pipeline.steady_state_prefetch_reuse",
                        method=method,
                        plan_id=plan_id,
                        run_id=run_id,
                        round_id=round_index,
                        shared=True,
                        start_ns=draft_submit_end_ns,
                        end_ns=recorder.clock(),
                        metadata={
                            "runtime_engine": "distributed_batch_pipeline",
                            "steady_state_prefetch": True,
                            "prefetch_hit_count": len(prefetch_hit_indices),
                            "job_indices": list(prefetch_hit_indices),
                        },
                    )
                    _record_span_event(
                        logger,
                        recorder,
                        prefetch_hit_span,
                        span_kind="leaf",
                        attribution="system",
                        metadata=dict(prefetch_hit_span.metadata),
                    )
                job_indices_by_request: dict[str, list[int]] = {}
                jobs_by_request: dict[str, list[DraftJob]] = {}
                for job_index, job in enumerate(plan.draft_jobs):
                    request_id = str(job.request_id)
                    job_indices_by_request.setdefault(request_id, []).append(job_index)
                    jobs_by_request.setdefault(request_id, []).append(job)

                consumed_job_indices: set[int] = set()
                server_available_ns = int(draft_submit_end_ns)
                for stage_index, batch in enumerate(plan.verify_batches):
                    batch_wait_start_ns = recorder.clock()
                    stage_job_indices = [
                        job_index
                        for request_id in batch.request_ids
                        for job_index in job_indices_by_request.get(str(request_id), [])
                    ]
                    stage_results: list[DraftExecutionResult] = []
                    for job_index in stage_job_indices:
                        if job_index in consumed_job_indices:
                            continue
                        stage_results.append(futures_by_job_index[job_index].result())
                        consumed_job_indices.add(job_index)
                    for draft_result in stage_results:
                        proposal, draft_span = self._record_draft_result(
                            draft_result,
                            logger=logger,
                            recorder=recorder,
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                        )
                        proposals_by_request.setdefault(draft_result.job.request_id, []).append(proposal)
                        proposals_by_id[proposal.proposal_id] = proposal
                        draft_spans_by_proposal[proposal.proposal_id] = draft_span
                        request_results[draft_result.job.request_id].proposals.append(proposal.proposal_id)

                    proposal_ids = [
                        proposal.proposal_id
                        for request_id in batch.request_ids
                        for proposal in proposals_by_request.get(str(request_id), [])
                    ]
                    batch.proposal_ids = proposal_ids
                    proposals = [
                        proposals_by_id[proposal_id]
                        for proposal_id in proposal_ids
                        if proposal_id in proposals_by_id
                    ]
                    if not proposals:
                        continue

                    draft_ready_ns = max(
                        int(draft_spans_by_proposal[proposal.proposal_id].end_ns or batch_wait_start_ns)
                        for proposal in proposals
                    )
                    if draft_ready_ns > server_available_ns:
                        idle_span = recorder.record_completed(
                            phase="pipeline.draft_ready_wait",
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            batch_id=batch.batch_id,
                            shared=True,
                            start_ns=server_available_ns,
                            end_ns=draft_ready_ns,
                            metadata={
                                "stage_index": stage_index,
                                "batch_id": batch.batch_id,
                                "request_ids": [proposal.request_id for proposal in proposals],
                                "runtime_engine": "distributed_batch_pipeline",
                                "idle_kind": "draft_ready_wait",
                            },
                        )
                        _record_span_event(
                            logger,
                            recorder,
                            idle_span,
                            span_kind="leaf",
                            attribution="system",
                            metadata=dict(idle_span.metadata),
                        )

                    with recorder.span(
                        phase="verify.batch_total",
                        method=method,
                        plan_id=plan_id,
                        run_id=run_id,
                        round_id=round_index,
                        request_id=batch.batch_id,
                        batch_id=batch.batch_id,
                        shared=True,
                        metadata={
                            **dict(batch.metadata or {}),
                            "stage_index": int(dict(batch.metadata or {}).get("stage_index", stage_index)),
                            "request_ids": [proposal.request_id for proposal in proposals],
                            "proposal_ids": [proposal.proposal_id for proposal in proposals],
                            "runtime_engine": "distributed_batch_pipeline",
                        },
                    ) as verify_span:
                        verification_results = self.verifier.verify_batch(proposals, context)
                    server_available_ns = int(verify_span.end_ns or recorder.clock())
                    stage_span = recorder.record_completed(
                        phase="pipeline.stage",
                        method=method,
                        plan_id=plan_id,
                        run_id=run_id,
                        round_id=round_index,
                        batch_id=batch.batch_id,
                        shared=True,
                        start_ns=batch_wait_start_ns,
                        end_ns=server_available_ns,
                        metadata={
                            **dict(batch.metadata or {}),
                            "stage_index": int(dict(batch.metadata or {}).get("stage_index", stage_index)),
                            "request_ids": [proposal.request_id for proposal in proposals],
                            "proposal_ids": [proposal.proposal_id for proposal in proposals],
                            "runtime_engine": "distributed_batch_pipeline",
                        },
                    )
                    _record_span_event(
                        logger,
                        recorder,
                        stage_span,
                        span_kind="aggregate",
                        attribution="batch",
                        metadata=dict(stage_span.metadata),
                    )
                    verification_results_by_id = _validate_verification_results(proposals, verification_results)
                    for verification_result in verification_results:
                        response_timing = dict(verification_result.timing.get("response_timing") or {})
                        _record_detail_event_specs(
                            logger,
                            recorder,
                            verification_result.timing.get("client_events", []),
                            default_phase=None,
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            request_id=verification_result.request_id,
                            session_id=verification_result.request_id,
                            batch_id=batch.batch_id,
                            proposal_id=verification_result.proposal_id,
                            attribution="request",
                            metadata_base={
                                "response_timing": response_timing,
                                "client_serialize_ms": verification_result.timing.get("client_serialize_ms"),
                                "client_deserialize_ms": verification_result.timing.get("client_deserialize_ms"),
                                "client_http_total_ms": verification_result.timing.get("client_http_total_ms"),
                                "request_bytes": verification_result.timing.get("request_bytes"),
                                "response_bytes": verification_result.timing.get("response_bytes"),
                                "modeled_upload_ms": verification_result.timing.get("modeled_upload_ms"),
                                "modeled_downlink_ms": verification_result.timing.get("modeled_downlink_ms"),
                                "network_or_queue_residual_ms": verification_result.timing.get(
                                    "network_or_queue_residual_ms"
                                ),
                                "backend_name": verification_result.metadata.get("backend_name"),
                            },
                        )
                    _record_span_event(
                        logger,
                        recorder,
                        verify_span,
                        span_kind="leaf",
                        attribution="batch",
                        tokens_in=sum(len(proposal.tokens) for proposal in proposals),
                        metadata=dict(verify_span.metadata),
                    )
                    for event in self.timing_attributor.attribute_batch_average(
                        parent_span=verify_span,
                        proposals=proposals,
                        event_id_factory=recorder.next_event_id,
                    ):
                        logger.record(event)

                    for request_id, request_proposals in _group_proposals_by_request(proposals).items():
                        accept_records = []
                        for proposal in request_proposals:
                            verification_result = verification_results_by_id[proposal.proposal_id]
                            with recorder.span(
                                phase="accept.apply",
                                method=method,
                                plan_id=plan_id,
                                run_id=run_id,
                                round_id=round_index,
                                request_id=proposal.request_id,
                                session_id=proposal.request_id,
                                worker_id=proposal.worker_id,
                                batch_id=batch.batch_id,
                                proposal_id=proposal.proposal_id,
                                metadata={
                                    "candidate_count": len(request_proposals),
                                    "candidate_group_id": f"{proposal.request_id}:round{round_index}",
                                },
                            ) as accept_span:
                                accept_result = self.acceptance_policy.accept(proposal, verification_result, context)
                            accept_records.append((proposal, verification_result, accept_result, accept_span))

                        winner = _commit_acceptance_if_supported(
                            _select_accept_record(accept_records),
                            acceptance_policy=self.acceptance_policy,
                            context=context,
                            draft_runners=draft_runners,
                        )
                        for proposal, _verification_result, accept_result, accept_span in accept_records:
                            if proposal.proposal_id == winner[0].proposal_id:
                                accept_result = winner[2]
                            accept_span.metadata.update(
                                {
                                    **dict(accept_result.metadata),
                                    "candidate_winner": proposal.proposal_id == winner[0].proposal_id,
                                    "candidate_output_len": len(accept_result.output_token_ids),
                                }
                            )
                            _record_span_event(
                                logger,
                                recorder,
                                accept_span,
                                span_kind="leaf",
                                attribution="request",
                                metadata=dict(accept_span.metadata),
                            )

                        proposal, _verification_result, accept_result, accept_span = winner
                        _update_acceptance_history(
                            planning_history,
                            proposal=proposal,
                            accept_result=accept_result,
                        )
                        session = sessions_by_id[request_id]
                        with recorder.span(
                            phase="pipeline.sync",
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            request_id=proposal.request_id,
                            session_id=proposal.request_id,
                            worker_id=proposal.worker_id,
                            batch_id=batch.batch_id,
                            proposal_id=proposal.proposal_id,
                            metadata={"runtime_engine": "distributed_batch_pipeline"},
                        ) as sync_span:
                            pass
                        _record_span_event(
                            logger,
                            recorder,
                            sync_span,
                            span_kind="detail",
                            attribution="request",
                            metadata=dict(sync_span.metadata),
                        )
                        with recorder.span(
                            phase="session.append",
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            request_id=proposal.request_id,
                            session_id=proposal.request_id,
                            worker_id=proposal.worker_id,
                            batch_id=batch.batch_id,
                            proposal_id=proposal.proposal_id,
                            metadata={"state": "PIPELINE_STATE_UPDATE"},
                        ) as append_span:
                            emitted = session.append_tokens(accept_result.output_token_ids)
                        _record_span_event(
                            logger,
                            recorder,
                            append_span,
                            span_kind="leaf",
                            attribution="request",
                            tokens_out=len(emitted),
                        )
                        if not emitted:
                            stalled.add(session.request_id)
                        request_results[session.request_id].output_token_ids = list(session.generated_ids)
                        request_results[session.request_id].stop_reason = accept_result.stop_reason
                        draft_span = draft_spans_by_proposal[proposal.proposal_id]
                        generation_span = recorder.record_completed(
                            phase="request.generation_total",
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            request_id=proposal.request_id,
                            session_id=proposal.request_id,
                            worker_id=proposal.worker_id,
                            batch_id=batch.batch_id,
                            proposal_id=proposal.proposal_id,
                            start_ns=_min_start_ns(draft_span, verify_span, accept_span, append_span),
                            end_ns=_max_end_ns(draft_span, verify_span, accept_span, append_span),
                        )
                        _record_span_event(
                            logger,
                            recorder,
                            generation_span,
                            span_kind="aggregate",
                            attribution="request",
                            tokens_out=len(accept_result.output_token_ids),
                            metadata=dict(accept_result.metadata),
                        )
                        if emitted and not session.is_finished:
                            source_job = _job_for_prefetch(jobs_by_request.get(session.request_id, []), proposal)
                            if source_job is not None:
                                self._submit_steady_state_prefetch(
                                    source_job,
                                    session=session,
                                    accept_result=accept_result,
                                    draft_runners=draft_runners,
                                    context=context,
                                    clock=recorder.clock,
                                    prefetched_drafts=prefetched_drafts,
                                    logger=logger,
                                    recorder=recorder,
                                    method=method,
                                    plan_id=plan_id,
                                    run_id=run_id,
                                    round_id=round_index,
                                    stage_index=stage_index,
                                    batch_id=batch.batch_id,
                                )

                for job_index, future in futures_by_job_index.items():
                    if job_index not in consumed_job_indices:
                        future.result()
            _record_span_event(logger, recorder, round_span, span_kind="aggregate", attribution="system")
            round_index += 1
        self._discard_prefetched_drafts(
            prefetched_drafts,
            logger=logger,
            recorder=recorder,
            method=method,
            run_id=run_id,
            reason="run_finished",
        )
        return RuntimeRunResult(request_results=list(request_results.values()), events=logger)

    def _submit_draft_jobs(
        self,
        jobs: list[Any],
        *,
        sessions_by_id: dict[str, Any],
        draft_runners: dict[str, Any],
        context: RuntimeContext,
        clock: Any,
    ) -> dict[int, Future[DraftExecutionResult]]:
        parallelism = draft_parallelism_for(
            draft_runners=draft_runners,
            jobs=jobs,
            context=context,
            requested=self.max_draft_workers,
        )

        def run_one(job: Any) -> DraftExecutionResult:
            session = sessions_by_id[job.request_id]
            runner = draft_runners[job.worker_id]
            start_ns = clock()
            proposal = self.candidate_strategy.propose(session, runner, job.budget, context)
            end_ns = clock()
            return DraftExecutionResult(
                job=job,
                session=session,
                proposal=proposal,
                start_ns=start_ns,
                end_ns=end_ns,
                parallelism=parallelism,
            )

        executor = ThreadPoolExecutor(max_workers=max(1, parallelism))
        futures = {
            index: executor.submit(run_one, job)
            for index, job in enumerate(jobs)
        }
        # Shutdown after all submitted work completes; futures remain usable.
        executor.shutdown(wait=False)
        return futures

    def _submit_draft_jobs_with_prefetch(
        self,
        jobs: list[DraftJob],
        *,
        sessions_by_id: dict[str, Any],
        draft_runners: dict[str, Any],
        context: RuntimeContext,
        clock: Any,
        prefetched_drafts: dict[str, _PrefetchedDraft],
    ) -> tuple[dict[int, Future[DraftExecutionResult]], list[int]]:
        futures: dict[int, Future[DraftExecutionResult]] = {}
        prefetch_hit_indices: list[int] = []
        pending_jobs: list[DraftJob] = []
        pending_indices: list[int] = []
        if _steady_state_enabled(context):
            for job_index, job in enumerate(jobs):
                prefetch = prefetched_drafts.get(str(job.request_id))
                session = sessions_by_id.get(str(job.request_id))
                if _prefetch_matches_job(prefetch, job, session):
                    prefetched_drafts.pop(str(job.request_id), None)
                    futures[job_index] = _adapt_prefetched_future(prefetch.future, job, session)
                    prefetch_hit_indices.append(job_index)
                else:
                    if prefetch is not None:
                        prefetched_drafts.pop(str(job.request_id), None)
                        prefetch.future.result()
                    pending_jobs.append(job)
                    pending_indices.append(job_index)
        else:
            pending_jobs = list(jobs)
            pending_indices = list(range(len(jobs)))
        pending_futures = self._submit_draft_jobs(
            pending_jobs,
            sessions_by_id=sessions_by_id,
            draft_runners=draft_runners,
            context=context,
            clock=clock,
        )
        for local_index, future in pending_futures.items():
            futures[pending_indices[local_index]] = future
        return futures, prefetch_hit_indices

    def _submit_steady_state_prefetch(
        self,
        job: DraftJob,
        *,
        session: GenerationSession,
        accept_result: Any,
        draft_runners: dict[str, Any],
        context: RuntimeContext,
        clock: Any,
        prefetched_drafts: dict[str, _PrefetchedDraft],
        logger: EventLogger,
        recorder: TimingRecorder,
        method: str,
        plan_id: str,
        run_id: str,
        round_id: int,
        stage_index: int,
        batch_id: str | None,
    ) -> None:
        if not _steady_state_enabled(context):
            return
        request_id = str(job.request_id)
        if request_id in prefetched_drafts or job.worker_id not in draft_runners:
            return
        session_snapshot = _snapshot_session(session)
        prefetch_prefix_ids = tuple(int(token_id) for token_id in session_snapshot.prefix_ids)
        prefetch_tokens, prefetch_metadata = _steady_state_prefetch_budget(
            job,
            session_snapshot,
            accept_result,
            context,
        )
        if prefetch_tokens <= 0:
            return
        prefetch_job = DraftJob(
            request_id=job.request_id,
            worker_id=job.worker_id,
            budget=DraftBudget(
                max_tokens=prefetch_tokens,
                max_branches=job.budget.max_branches,
                timeout_ms=job.budget.timeout_ms,
            ),
            metadata={
                **dict(job.metadata or {}),
                "steady_state_prefetch": True,
                "source_budget_max_tokens": int(job.budget.max_tokens),
                **prefetch_metadata,
            },
        )
        submit_ns = clock()
        parallelism = draft_parallelism_for(
            draft_runners=draft_runners,
            jobs=[prefetch_job],
            context=context,
            requested=self.max_draft_workers,
        )

        def run_one() -> DraftExecutionResult:
            runner = draft_runners[prefetch_job.worker_id]
            start_ns = clock()
            proposal = self.candidate_strategy.propose(session_snapshot, runner, prefetch_job.budget, context)
            end_ns = clock()
            proposal.metadata.update(
                {
                    "steady_state_prefetch": True,
                    "prefetch_source_round": round_id,
                    "prefetch_source_stage": stage_index,
                    "prefetch_source_batch_id": batch_id,
                    "prefetch_submit_ns": submit_ns,
                    "prefetch_prefix_ids": list(prefetch_prefix_ids),
                    "prefetch_step_idx": int(session_snapshot.step_idx),
                    "source_budget_max_tokens": int(job.budget.max_tokens),
                    **prefetch_metadata,
                }
            )
            return DraftExecutionResult(
                job=prefetch_job,
                session=session_snapshot,
                proposal=proposal,
                start_ns=start_ns,
                end_ns=end_ns,
                parallelism=parallelism,
            )

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_one)
        executor.shutdown(wait=False)
        prefetched_drafts[request_id] = _PrefetchedDraft(
            request_id=request_id,
            worker_id=str(prefetch_job.worker_id),
            budget_tokens=int(prefetch_job.budget.max_tokens),
            prefix_ids=prefetch_prefix_ids,
            step_idx=int(session_snapshot.step_idx),
            future=future,
            submitted_ns=submit_ns,
            source_round=round_id,
            source_stage=stage_index,
            source_batch_id=batch_id,
        )
        submit_span = recorder.record_completed(
            phase="pipeline.steady_state_prefetch_submit",
            method=method,
            plan_id=plan_id,
            run_id=run_id,
            round_id=round_id,
            request_id=request_id,
            session_id=request_id,
            worker_id=prefetch_job.worker_id,
            batch_id=batch_id,
            shared=False,
            start_ns=submit_ns,
            end_ns=clock(),
            metadata={
                "runtime_engine": "distributed_batch_pipeline",
                "steady_state_prefetch": True,
                "source_round": round_id,
                "source_stage": stage_index,
                "source_batch_id": batch_id,
                "max_tokens": int(prefetch_job.budget.max_tokens),
                "source_budget_max_tokens": int(job.budget.max_tokens),
                **prefetch_metadata,
                "prefix_len": len(prefetch_prefix_ids),
                "step_idx": int(session_snapshot.step_idx),
            },
        )
        _record_span_event(
            logger,
            recorder,
            submit_span,
            span_kind="leaf",
            attribution="request",
            metadata=dict(submit_span.metadata),
        )

    def _discard_prefetched_drafts(
        self,
        prefetched_drafts: dict[str, _PrefetchedDraft],
        *,
        logger: EventLogger,
        recorder: TimingRecorder,
        method: str,
        run_id: str,
        reason: str,
    ) -> None:
        while prefetched_drafts:
            _request_id, prefetch = prefetched_drafts.popitem()
            draft_result = prefetch.future.result()
            proposal = draft_result.proposal
            discard_span = recorder.record_completed(
                phase="draft.prefetch_discard",
                method=method,
                plan_id=f"{run_id}:prefetch-discard",
                run_id=run_id,
                request_id=draft_result.session.request_id,
                session_id=draft_result.session.request_id,
                worker_id=draft_result.job.worker_id,
                proposal_id=proposal.proposal_id,
                start_ns=draft_result.start_ns,
                end_ns=draft_result.end_ns,
                metadata={
                    **dict(proposal.metadata or {}),
                    "steady_state_prefetch": True,
                    "discard_reason": reason,
                    "source_round": prefetch.source_round,
                    "source_stage": prefetch.source_stage,
                    "source_batch_id": prefetch.source_batch_id,
                },
            )
            _record_span_event(
                logger,
                recorder,
                discard_span,
                span_kind="leaf",
                attribution="request",
                tokens_out=len(proposal.tokens),
                metadata=dict(discard_span.metadata),
            )

    def _record_draft_result(
        self,
        draft_result: DraftExecutionResult,
        *,
        logger: EventLogger,
        recorder: TimingRecorder,
        method: str,
        plan_id: str,
        run_id: str,
        round_id: int,
    ) -> tuple[CandidateProposal, TimingSpan]:
        job = draft_result.job
        session = draft_result.session
        proposal = draft_result.proposal
        draft_span = recorder.record_completed(
            phase="draft.generate",
            method=method,
            plan_id=plan_id,
            run_id=run_id,
            round_id=round_id,
            request_id=session.request_id,
            session_id=session.request_id,
            worker_id=job.worker_id,
            proposal_id=proposal.proposal_id,
            metadata={
                "state": "READY_DRAFT",
                "draft_parallelism": draft_result.parallelism,
                "parallel_draft": draft_result.parallelism > 1,
                "runtime_engine": "distributed_batch_pipeline",
            },
            start_ns=draft_result.start_ns,
            end_ns=draft_result.end_ns,
        )
        _record_span_event(
            logger,
            recorder,
            draft_span,
            span_kind="leaf",
            attribution="request",
            tokens_out=len(proposal.tokens),
            metadata=dict(proposal.metadata),
        )
        _record_detail_event_specs(
            logger,
            recorder,
            proposal.metadata.get("draft_token_forward_events", []),
            default_phase="draft.token_forward",
            method=method,
            plan_id=plan_id,
            run_id=run_id,
            round_id=round_id,
            request_id=session.request_id,
            session_id=session.request_id,
            worker_id=job.worker_id,
            proposal_id=proposal.proposal_id,
            attribution="request",
            tokens_out=1,
        )
        return proposal, draft_span

    def _plan_hints(
        self,
        active_sessions: list[GenerationSession],
        draft_runners: dict[str, Any],
        context: RuntimeContext,
        *,
        history: dict[str, Any] | None = None,
    ) -> PlanHints:
        if self.planning_policy is None:
            return PlanHints()
        return self.planning_policy.plan(
            active_sessions=active_sessions,
            resources={
                "draft_worker_ids": list(draft_runners),
                "draft_worker_metadata": _draft_runner_metadata(draft_runners),
            },
            history=history or {},
            context=context,
        )


def _update_acceptance_history(
    history: dict[str, Any],
    *,
    proposal: CandidateProposal,
    accept_result: Any,
) -> None:
    request_id = str(proposal.request_id)
    draft_token_count = max(0, len(proposal.tokens))
    if draft_token_count <= 0:
        return
    metadata = dict(getattr(accept_result, "metadata", {}) or {})
    accepted_count = int(metadata.get("accepted_count", len(getattr(accept_result, "accepted_tokens", []) or [])) or 0)
    accepted_count = max(0, min(draft_token_count, accepted_count))
    bonus_count = 1 if getattr(accept_result, "bonus_token", None) is not None else 0
    all_stats = history.setdefault("dip_sd_acceptance_stats", {})
    stats = all_stats.setdefault(
        request_id,
        {
            "proposal_count": 0,
            "draft_token_count": 0,
            "accepted_draft_count": 0,
            "bonus_count": 0,
            "output_token_count": 0,
            "last_observed_acceptance": None,
        },
    )
    stats["proposal_count"] = int(stats.get("proposal_count") or 0) + 1
    stats["draft_token_count"] = int(stats.get("draft_token_count") or 0) + draft_token_count
    stats["accepted_draft_count"] = int(stats.get("accepted_draft_count") or 0) + accepted_count
    stats["bonus_count"] = int(stats.get("bonus_count") or 0) + bonus_count
    stats["output_token_count"] = int(stats.get("output_token_count") or 0) + len(getattr(accept_result, "output_token_ids", []) or [])
    stats["last_observed_acceptance"] = accepted_count / max(1, draft_token_count)
    stats["observed_acceptance"] = (
        int(stats["accepted_draft_count"]) / max(1, int(stats["draft_token_count"]))
    )


def _planner_phase(hints: PlanHints) -> str:
    metadata = dict(getattr(hints, "metadata", {}) or {})
    if metadata.get("method_family") == "dip_sd":
        return "dip_sd.solver"
    return "planner.hints"


def _planner_metadata(hints: PlanHints) -> dict[str, Any]:
    metadata = dict(getattr(hints, "metadata", {}) or {})
    keys = (
        "method_family",
        "solver_active",
        "requested_solver_mode",
        "solver_mode",
        "solver_backend_name",
        "paper_solver_complete",
        "solver_backend_fallback_used",
        "solver_backend_fallback_reason",
        "solver_backend_info",
        "solver_cache_hit",
        "solver_cache_key",
        "online_solver_enabled",
        "offline_plan_table_hit",
        "offline_plan_key",
        "offline_plan_shape_key",
        "offline_plan_source",
        "shape_cache_key",
        "shape_cache_source_key",
        "planned_batch_count",
        "estimated_throughput_tokens_per_ms",
        "estimated_throughput_tokens_per_s",
        "estimated_expected_tokens",
        "estimated_pipeline_span_ms",
        "estimated_verification_sum_ms",
        "estimated_max_draft_verify_ms",
        "solver_planned_batch_count",
        "hybrid_single_batch_threshold",
        "hybrid_single_batch_applied",
        "hybrid_single_batch_reason",
        "dip_sd_model_config",
        "latency_calibration_profile",
        "latency_calibration_enabled",
        "latency_calibration_applied",
        "latency_calibration_overrides",
        "adaptive_draft_length_enabled",
        "adaptive_draft_length_applied",
        "adaptive_draft_length_changes",
        "adaptive_draft_length_min_tokens",
        "adaptive_draft_length_target_acceptance",
        "adaptive_draft_length_min_factor",
        "adaptive_draft_length_fastest_beta",
        "ready_aware_rebatch_enabled",
        "ready_aware_rebatch_applied",
        "ready_aware_ready_time_ms",
        "ready_aware_rebatch_spread_ms",
        "ready_aware_rebatch_min_spread_ms",
        "ready_aware_rebatch_reason",
        "ready_aware_original_batches",
        "ready_aware_rebatched_batches",
        "acceptance_feedback_enabled",
        "acceptance_feedback_by_request",
        "acceptance_feedback_applied_count",
        "acceptance_cache_bucket",
        "solver_cache_shape_level",
    )
    return {
        "runtime_engine": "distributed_batch_pipeline",
        "hints_metadata": {key: metadata.get(key) for key in keys if key in metadata},
    }


def _steady_state_enabled(context: RuntimeContext) -> bool:
    raw = context.method_config.get("dip_sd_steady_state_enabled")
    if raw is not None:
        return _config_bool(raw)
    return bool(context.method_config.get("dip_sd_solver") or context.run_config.get("method") == "dip_sd")


def _snapshot_session(session: GenerationSession) -> GenerationSession:
    snapshot = GenerationSession(
        request_id=str(session.request_id),
        prompt_ids=[int(token_id) for token_id in session.prompt_ids],
        max_new_tokens=int(session.max_new_tokens),
        max_len=int(session.max_len),
        eos_token_id=session.eos_token_id,
        eos_token_ids=[int(token_id) for token_id in session.eos_token_ids],
        generated_ids=[int(token_id) for token_id in session.generated_ids],
        step_idx=int(session.step_idx),
    )
    return snapshot


def _steady_state_prefetch_budget(
    job: DraftJob,
    session: GenerationSession,
    accept_result: Any,
    context: RuntimeContext,
) -> tuple[int, dict[str, Any]]:
    source_budget = max(1, int(job.budget.max_tokens))
    remaining = max(0, int(session.remaining_tokens))
    budget = min(source_budget, remaining)
    adaptive_enabled = _config_bool(
        context.method_config.get("dip_sd_prefetch_adaptive_length_enabled", True)
    )
    metadata: dict[str, Any] = {
        "prefetch_adaptive_length_enabled": adaptive_enabled,
        "prefetch_length_reason": "source_budget",
    }
    if remaining <= 0:
        return 0, metadata
    if adaptive_enabled:
        output_len = len(getattr(accept_result, "output_token_ids", []) or [])
        accept_metadata = dict(getattr(accept_result, "metadata", {}) or {})
        accepted_count = int(accept_metadata.get("accepted_count") or len(getattr(accept_result, "accepted_tokens", []) or []))
        lookahead = max(
            0,
            int(context.method_config.get("dip_sd_prefetch_acceptance_lookahead_tokens", 1) or 0),
        )
        min_tokens = max(
            1,
            int(context.method_config.get("dip_sd_prefetch_min_tokens", 1) or 1),
        )
        observed_target = max(min_tokens, output_len + lookahead)
        source_budget_floor = _config_bool(
            context.method_config.get("dip_sd_prefetch_use_source_budget_floor", False)
        )
        if source_budget_floor:
            observed_target = max(observed_target, source_budget)
        budget = min(budget, observed_target)
        metadata.update(
            {
                "prefetch_length_reason": "acceptance_output_length",
                "prefetch_observed_output_tokens": int(output_len),
                "prefetch_observed_accepted_tokens": int(accepted_count),
                "prefetch_acceptance_lookahead_tokens": int(lookahead),
                "prefetch_min_tokens": int(min_tokens),
                "prefetch_use_source_budget_floor": source_budget_floor,
            }
        )
    max_tokens = int(context.method_config.get("dip_sd_prefetch_max_tokens", 0) or 0)
    if max_tokens > 0:
        budget = min(budget, max_tokens)
        metadata["prefetch_max_tokens"] = int(max_tokens)
    budget = max(1, min(int(budget), remaining))
    metadata["prefetch_budget_tokens"] = int(budget)
    return budget, metadata


def _prefetch_matches_job(
    prefetch: _PrefetchedDraft | None,
    job: DraftJob,
    session: GenerationSession | None,
) -> bool:
    if prefetch is None:
        return False
    if session is None:
        return False
    return (
        str(prefetch.request_id) == str(job.request_id)
        and str(prefetch.worker_id) == str(job.worker_id)
        and int(prefetch.budget_tokens) >= int(job.budget.max_tokens)
        and tuple(int(token_id) for token_id in session.prefix_ids) == prefetch.prefix_ids
        and int(session.step_idx) == int(prefetch.step_idx)
    )


def _prefetched_draft_history(prefetched_drafts: dict[str, _PrefetchedDraft]) -> dict[str, dict[str, Any]]:
    return {
        str(request_id): {
            "worker_id": str(prefetch.worker_id),
            "budget_tokens": int(prefetch.budget_tokens),
            "source_round": int(prefetch.source_round),
            "source_stage": int(prefetch.source_stage),
            "source_batch_id": prefetch.source_batch_id,
        }
        for request_id, prefetch in prefetched_drafts.items()
    }


def _adapt_prefetched_future(
    future: Future[DraftExecutionResult],
    job: DraftJob,
    session: GenerationSession,
) -> Future[DraftExecutionResult]:
    adapted: Future[DraftExecutionResult] = Future()

    def _complete(source: Future[DraftExecutionResult]) -> None:
        try:
            adapted.set_result(_adapt_prefetched_draft_result(source.result(), job, session))
        except BaseException as exc:  # pragma: no cover - mirrors Future exception propagation.
            adapted.set_exception(exc)

    future.add_done_callback(_complete)
    return adapted


def _adapt_prefetched_draft_result(
    draft_result: DraftExecutionResult,
    job: DraftJob,
    session: GenerationSession,
) -> DraftExecutionResult:
    proposal = draft_result.proposal
    max_tokens = max(0, min(int(job.budget.max_tokens), int(session.remaining_tokens)))
    original_tokens = list(proposal.tokens)
    reused_tokens = original_tokens[:max_tokens]
    metadata = dict(proposal.metadata or {})
    original_budget = int(
        metadata.get("source_budget_max_tokens")
        or metadata.get("prefetch_original_budget_tokens")
        or metadata.get("max_tokens")
        or len(original_tokens)
    )
    metadata.update(
        {
            "steady_state_prefetch": True,
            "prefetch_reused": True,
            "prefetch_truncated": len(reused_tokens) < len(original_tokens),
            "prefetch_original_draft_length": len(original_tokens),
            "prefetch_original_budget_tokens": original_budget,
            "prefetch_reused_budget_tokens": int(job.budget.max_tokens),
            "remaining_tokens": int(session.remaining_tokens),
            "allow_bonus": len(reused_tokens) < int(session.remaining_tokens),
        }
    )
    draft_budget = dict(metadata.get("draft_budget") or {})
    draft_budget.update(
        {
            "max_tokens": int(job.budget.max_tokens),
            "max_branches": int(job.budget.max_branches),
            "timeout_ms": job.budget.timeout_ms,
        }
    )
    metadata["draft_budget"] = draft_budget
    adapted_proposal = replace(
        proposal,
        tokens=reused_tokens,
        draft_length=len(reused_tokens),
        metadata=metadata,
    )
    return replace(draft_result, job=job, proposal=adapted_proposal)


def _job_for_prefetch(jobs: list[DraftJob], proposal: CandidateProposal) -> DraftJob | None:
    if not jobs:
        return None
    worker_id = str(proposal.worker_id or "")
    for job in jobs:
        if str(job.worker_id) == worker_id:
            return job
    return jobs[0]


def _config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _record_span_event(
    logger: EventLogger,
    recorder: TimingRecorder,
    span: TimingSpan,
    *,
    span_kind: str,
    attribution: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    logger.record(
        event_from_span(
            span,
            event_id_factory=recorder.next_event_id,
            span_kind=span_kind,
            attribution=attribution,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=metadata,
        )
    )


def _record_detail_event_specs(
    logger: EventLogger,
    recorder: TimingRecorder,
    event_specs: Any,
    *,
    default_phase: str | None,
    method: str,
    plan_id: str,
    run_id: str,
    round_id: int,
    request_id: str | None = None,
    session_id: str | None = None,
    worker_id: str | None = None,
    batch_id: str | None = None,
    proposal_id: str | None = None,
    attribution: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    metadata_base: dict[str, Any] | None = None,
) -> None:
    if not isinstance(event_specs, list):
        return
    for event_spec in event_specs:
        if not isinstance(event_spec, dict):
            continue
        start_ns = event_spec.get("start_ns")
        end_ns = event_spec.get("end_ns")
        if start_ns is None or end_ns is None:
            continue
        phase = str(event_spec.get("phase") or default_phase or "")
        if not phase:
            continue
        base_metadata = dict(metadata_base or {})
        if phase != "verify.http_total":
            base_metadata.pop("response_timing", None)
            base_metadata.pop("network_or_queue_residual_ms", None)
        metadata = {
            **base_metadata,
            **dict(event_spec.get("metadata") or {}),
            **{
                key: value
                for key, value in event_spec.items()
                if key not in {"start_ns", "end_ns", "metadata"}
            },
        }
        detail_span = recorder.record_completed(
            phase=phase,
            method=method,
            plan_id=plan_id,
            run_id=run_id,
            round_id=round_id,
            request_id=request_id,
            session_id=session_id,
            worker_id=worker_id,
            batch_id=batch_id,
            proposal_id=proposal_id,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            metadata=metadata,
        )
        _record_span_event(
            logger,
            recorder,
            detail_span,
            span_kind="detail",
            attribution=attribution,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=metadata,
        )
