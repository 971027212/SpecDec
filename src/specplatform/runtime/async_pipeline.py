from __future__ import annotations

"""异步 edge-cloud speculative runtime。

该 runtime 把 verify batch 放到后台 future 中执行，并在 verify in-flight
期间运行 proactive draft。method-specific 行为通过 proactive/reconcile 策略注入。
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import Any

from specplatform.core import CandidateProposal, PlanHints, RuntimeContext, VerificationResult, VerifyBatch
from specplatform.methods.base import (
    AcceptancePolicy,
    CandidateStrategy,
    PlanningPolicy,
    ProactiveDraftPolicy,
    ReconcilePolicy,
)
from specplatform.metrics import EventLogger
from specplatform.runtime.draft_execution import execute_draft_jobs, execute_proactive_jobs
from specplatform.runtime.engine import (
    RuntimeRequestResult,
    RuntimeRunResult,
    _commit_acceptance_if_supported,
    _group_proposals_by_request,
    _select_accept_record,
    _validate_verification_results,
)
from specplatform.runtime.session import GenerationSession
from specplatform.schedulers import Scheduler, SchedulerResources, StaticQueueBatchPlanner, VerificationArrival, summarize_queue_batches
from specplatform.schedulers.batch_planner import attach_proposals_to_batches
from specplatform.timing import TimingAttributor, TimingRecorder, event_from_span
from specplatform.timing.span import TimingSpan
from specplatform.verification import VerifierBackend


@dataclass
class AsyncPipelineRuntimeEngine:
    """通用异步 pipeline runtime。"""

    candidate_strategy: CandidateStrategy
    acceptance_policy: AcceptancePolicy
    scheduler: Scheduler
    verifier: VerifierBackend
    proactive_policy: ProactiveDraftPolicy
    reconcile_policy: ReconcilePolicy
    planning_policy: PlanningPolicy | None = None
    timing_recorder: TimingRecorder | None = None
    timing_attributor: TimingAttributor = field(default_factory=TimingAttributor)
    max_verify_workers: int = 1

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
        method = str(context.run_config.get("method", "async_pipeline_runtime"))
        request_results = {
            session.request_id: RuntimeRequestResult(request_id=session.request_id)
            for session in sessions
        }
        cached_proposals: dict[str, CandidateProposal] = {}
        stalled: set[str] = set()
        round_index = 0
        with ThreadPoolExecutor(max_workers=max(1, int(self.max_verify_workers))) as executor:
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
                ) as round_span:
                    hints = self._plan_hints(active_sessions, draft_runners, context)
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

                    sessions_by_id = {session.request_id: session for session in active_sessions}
                    proposals_by_request: dict[str, list[CandidateProposal]] = {}
                    draft_spans_by_proposal: dict[str, TimingSpan] = {}
                    pending_jobs: list[Any] = []
                    cached_results_by_index: dict[int, tuple[Any, Any, CandidateProposal, TimingSpan]] = {}
                    for job_index, job in enumerate(plan.draft_jobs):
                        session = sessions_by_id[job.request_id]
                        cached = _pop_aligned_cached_proposal(cached_proposals, session)
                        if cached is not None:
                            with recorder.span(
                                phase="draft.reuse_proactive",
                                method=method,
                                plan_id=plan_id,
                                run_id=run_id,
                                round_id=round_index,
                                request_id=session.request_id,
                                session_id=session.request_id,
                                worker_id=job.worker_id,
                                metadata={
                                    "state": "READY_DRAFT",
                                    "source_proposal_id": cached.proposal_id,
                                    "tree_node_count": cached.metadata.get("tree_node_count"),
                                },
                            ) as draft_span:
                                proposal = cached
                                draft_span.proposal_id = proposal.proposal_id
                            cached_results_by_index[job_index] = (job, session, proposal, draft_span)
                        else:
                            pending_jobs.append(job)
                    pending_results = iter(
                        execute_draft_jobs(
                            jobs=pending_jobs,
                            sessions_by_id=sessions_by_id,
                            draft_runners=draft_runners,
                            candidate_strategy=self.candidate_strategy,
                            context=context,
                            clock=recorder.clock,
                        )
                    )
                    for job_index, job in enumerate(plan.draft_jobs):
                        if job_index in cached_results_by_index:
                            job, session, proposal, draft_span = cached_results_by_index[job_index]
                        else:
                            draft_result = next(pending_results)
                            session = draft_result.session
                            proposal = draft_result.proposal
                            draft_span = recorder.record_completed(
                                phase="draft.generate",
                                method=method,
                                plan_id=plan_id,
                                run_id=run_id,
                                round_id=round_index,
                                request_id=session.request_id,
                                session_id=session.request_id,
                                worker_id=job.worker_id,
                                proposal_id=proposal.proposal_id,
                                metadata={
                                    "state": "READY_DRAFT",
                                    "draft_parallelism": draft_result.parallelism,
                                    "parallel_draft": draft_result.parallelism > 1,
                                },
                                start_ns=draft_result.start_ns,
                                end_ns=draft_result.end_ns,
                            )
                        proposals_by_request.setdefault(job.request_id, []).append(proposal)
                        draft_spans_by_proposal[proposal.proposal_id] = draft_span
                        request_results[job.request_id].proposals.append(proposal.proposal_id)
                        _record_span_event(
                            logger,
                            recorder,
                            draft_span,
                            span_kind="leaf",
                            attribution="request",
                            tokens_out=proposal.draft_length or len(proposal.tokens),
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
                            round_id=round_index,
                            request_id=session.request_id,
                            session_id=session.request_id,
                            worker_id=job.worker_id,
                            proposal_id=proposal.proposal_id,
                            attribution="request",
                        )

                    attach_proposals_to_batches(
                        plan.verify_batches,
                        {
                            request_id: [proposal.proposal_id for proposal in proposals]
                            for request_id, proposals in proposals_by_request.items()
                        },
                    )
                    proposals_by_id = {
                        proposal.proposal_id: proposal
                        for proposals in proposals_by_request.values()
                        for proposal in proposals
                    }

                    verify_batches = _maybe_apply_sled_static_queue_batches(
                        plan.verify_batches,
                        proposals_by_id=proposals_by_id,
                        draft_spans_by_proposal=draft_spans_by_proposal,
                        round_start_ns=int(round_span.start_ns),
                        context=context,
                        logger=logger,
                        recorder=recorder,
                        method=method,
                        plan_id=plan_id,
                        run_id=run_id,
                        round_id=round_index,
                    )

                    pending_verify_batches: list[tuple[VerifyBatch, list[CandidateProposal], int, Any]] = []
                    for batch in verify_batches:
                        proposals = [
                            proposals_by_id[proposal_id]
                            for proposal_id in batch.proposal_ids
                            if proposal_id in proposals_by_id
                        ]
                        if not proposals:
                            continue

                        verify_submit_ns = perf_counter_ns()
                        future = executor.submit(self.verifier.verify_batch, proposals, context)
                        pending_verify_batches.append((batch, proposals, verify_submit_ns, future))

                    for batch, proposals, verify_submit_ns, future in pending_verify_batches:
                        proactive_by_proposal: dict[str, CandidateProposal | None] = {}
                        proactive_spans: list[TimingSpan] = []
                        proactive_results = execute_proactive_jobs(
                            proposals=proposals,
                            sessions_by_id=sessions_by_id,
                            draft_runners=draft_runners,
                            proactive_policy=self.proactive_policy,
                            context=context,
                            clock=recorder.clock,
                        )
                        for proactive_result in proactive_results:
                            proposal = proactive_result.proposal
                            session = proactive_result.session
                            runner_id = proactive_result.runner_id
                            proactive = proactive_result.proactive
                            proactive_span = recorder.record_completed(
                                phase="draft.proactive",
                                method=method,
                                plan_id=plan_id,
                                run_id=run_id,
                                round_id=round_index,
                                request_id=session.request_id,
                                session_id=session.request_id,
                                worker_id=runner_id,
                                batch_id=batch.batch_id,
                                proposal_id=proposal.proposal_id,
                                start_ns=proactive_result.start_ns,
                                end_ns=proactive_result.end_ns,
                                metadata={
                                    "state": "PROACTIVE_DRAFTING",
                                    "proactive_parallelism": proactive_result.parallelism,
                                    "parallel_proactive": proactive_result.parallelism > 1,
                                    "proactive_proposal_id": None if proactive is None else proactive.proposal_id,
                                    "tree_node_count": 0
                                    if proactive is None
                                    else proactive.metadata.get("tree_node_count", 0),
                                },
                            )
                            proactive_by_proposal[proposal.proposal_id] = proactive
                            proactive_spans.append(proactive_span)
                            _record_span_event(
                                logger,
                                recorder,
                                proactive_span,
                                span_kind="leaf",
                                attribution="request",
                                tokens_out=0 if proactive is None else proactive.draft_length,
                                metadata=dict(proactive_span.metadata),
                            )
                            if proactive is not None:
                                _record_detail_event_specs(
                                    logger,
                                    recorder,
                                    proactive.metadata.get("draft_token_forward_events", []),
                                    default_phase="draft.proactive_token_forward",
                                    method=method,
                                    plan_id=plan_id,
                                    run_id=run_id,
                                    round_id=round_index,
                                    request_id=session.request_id,
                                    session_id=session.request_id,
                                    worker_id=runner_id,
                                    proposal_id=proactive.proposal_id,
                                    attribution="request",
                                )

                        verification_attempt = _await_verification_results(
                            future,
                            executor=executor,
                            verifier=self.verifier,
                            logger=logger,
                            recorder=recorder,
                            context=context,
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            batch_id=batch.batch_id,
                            proposals=proposals,
                            proactive_by_proposal=proactive_by_proposal,
                            verify_submit_ns=verify_submit_ns,
                        )
                        proposals = verification_attempt.proposals
                        proactive_by_proposal = verification_attempt.proactive_by_proposal
                        verification_results = verification_attempt.verification_results
                        verify_span = _record_verify_batch_span(
                            logger,
                            recorder,
                            verification_results,
                            fallback_start_ns=verify_submit_ns,
                            method=method,
                            plan_id=plan_id,
                            run_id=run_id,
                            round_id=round_index,
                            batch_id=batch.batch_id,
                            proposals=proposals,
                            batch_metadata=dict(batch.metadata or {}),
                        )
                        verification_results_by_id = _validate_verification_results(
                            proposals,
                            verification_results,
                        )
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
                        for event in self.timing_attributor.attribute_batch_average(
                            parent_span=verify_span,
                            proposals=proposals,
                            event_id_factory=recorder.next_event_id,
                        ):
                            logger.record(event)
                        _observe_planning_policy(
                            self.planning_policy,
                            draft_spans_by_proposal,
                            proactive_spans,
                            verification_results,
                        )

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
                                        "state": "VERIFY_DONE",
                                        "candidate_count": len(request_proposals),
                                        "candidate_group_id": f"{proposal.request_id}:round{round_index}",
                                    },
                                ) as accept_span:
                                    accept_result = self.acceptance_policy.accept(
                                        proposal,
                                        verification_result,
                                        context,
                                    )
                                accept_records.append((proposal, verification_result, accept_result, accept_span))

                            winner = _commit_acceptance_if_supported(
                                _select_accept_record(accept_records),
                                acceptance_policy=self.acceptance_policy,
                                context=context,
                                draft_runners=draft_runners,
                            )
                            accept_records = [
                                winner
                                if record[0].proposal_id == winner[0].proposal_id
                                else record
                                for record in accept_records
                            ]
                            for proposal, _verification_result, accept_result, accept_span in accept_records:
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

                            proposal, verification_result, accept_result, accept_span = winner
                            session = sessions_by_id[request_id]
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
                                metadata={"state": "RECONCILE_AND_APPEND"},
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
                            proactive = proactive_by_proposal.get(proposal.proposal_id)
                            with recorder.span(
                                phase="pipeline.reconcile",
                                method=method,
                                plan_id=plan_id,
                                run_id=run_id,
                                round_id=round_index,
                                request_id=proposal.request_id,
                                session_id=proposal.request_id,
                                worker_id=proposal.worker_id,
                                batch_id=batch.batch_id,
                                proposal_id=proposal.proposal_id,
                            ) as reconcile_span:
                                reconcile_result = self.reconcile_policy.reconcile(
                                    session,
                                    proposal,
                                    verification_result,
                                    accept_result,
                                    proactive,
                                    context,
                                )
                                reconcile_span.metadata.update(
                                    {
                                        **dict(reconcile_result.metadata),
                                        "aligned": reconcile_result.aligned,
                                        "reused_token_count": reconcile_result.reused_token_count,
                                        "discarded_token_count": reconcile_result.discarded_token_count,
                                        "reused_proposal_id": None
                                        if reconcile_result.reused_proposal is None
                                        else reconcile_result.reused_proposal.proposal_id,
                                    }
                                )
                            _record_span_event(
                                logger,
                                recorder,
                                reconcile_span,
                                span_kind="leaf",
                                attribution="request",
                                metadata=dict(reconcile_span.metadata),
                            )
                            if reconcile_result.reused_proposal is not None:
                                cached_proposals[proposal.request_id] = reconcile_result.reused_proposal
                            if not emitted:
                                stalled.add(session.request_id)
                            request_results[session.request_id].output_token_ids = list(session.generated_ids)
                            request_results[session.request_id].stop_reason = accept_result.stop_reason

                            request_draft_spans = [
                                draft_spans_by_proposal[item.proposal_id]
                                for item in request_proposals
                                if item.proposal_id in draft_spans_by_proposal
                            ]
                            generation_start = min(
                                *[span.start_ns for span in request_draft_spans],
                                verify_span.start_ns,
                                accept_span.start_ns,
                                append_span.start_ns,
                            )
                            generation_end = max(
                                *[int(span.end_ns or 0) for span in request_draft_spans],
                                int(verify_span.end_ns or 0),
                                int(accept_span.end_ns or 0),
                                int(append_span.end_ns or 0),
                            )
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
                                start_ns=generation_start,
                                end_ns=generation_end,
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
                _record_span_event(logger, recorder, round_span, span_kind="aggregate", attribution="system")
                round_index += 1
        return RuntimeRunResult(request_results=list(request_results.values()), events=logger)

    def _plan_hints(
        self,
        active_sessions: list[GenerationSession],
        draft_runners: dict[str, Any],
        context: RuntimeContext,
    ) -> PlanHints:
        if self.planning_policy is None:
            return PlanHints()
        return self.planning_policy.plan(
            active_sessions=active_sessions,
            resources={
                "draft_worker_ids": list(draft_runners),
                "draft_worker_metadata": _draft_runner_metadata(draft_runners),
            },
            history={},
            context=context,
        )


def _pop_aligned_cached_proposal(
    cached_proposals: dict[str, CandidateProposal],
    session: GenerationSession,
) -> CandidateProposal | None:
    cached = cached_proposals.pop(session.request_id, None)
    if cached is None:
        return None
    if [int(token_id) for token_id in cached.metadata.get("prefix_ids", [])] != list(session.prefix_ids):
        return None
    return cached


def _maybe_apply_sled_static_queue_batches(
    verify_batches: list[VerifyBatch],
    *,
    proposals_by_id: dict[str, CandidateProposal],
    draft_spans_by_proposal: dict[str, TimingSpan],
    round_start_ns: int,
    context: RuntimeContext,
    logger: EventLogger,
    recorder: TimingRecorder,
    method: str,
    plan_id: str,
    run_id: str,
    round_id: int,
) -> list[VerifyBatch]:
    """Use SLED's central static queue planner on real per-worker draft arrivals."""
    if not _config_bool(context.method_config.get("sled_static_queue_enabled"), False):
        return verify_batches

    proposal_order = [
        proposal_id
        for batch in verify_batches
        for proposal_id in batch.proposal_ids
        if proposal_id in proposals_by_id
    ]
    if not proposal_order:
        return verify_batches

    batch_size = _optional_int(context.method_config.get("sled_batch_size"), default=len(proposal_order))
    if batch_size is None or batch_size <= 0:
        batch_size = len(proposal_order)
    batch_size = max(1, int(batch_size))
    max_wait_ms = _optional_float(context.method_config.get("sled_queue_max_wait_ms"))
    pad_to_max_length = _config_bool(context.method_config.get("sled_queue_pad_to_max_length"), True)

    arrivals: list[VerificationArrival] = []
    for order_index, proposal_id in enumerate(proposal_order):
        proposal = proposals_by_id[proposal_id]
        draft_span = draft_spans_by_proposal.get(proposal_id)
        arrival_ns = int(draft_span.end_ns) if draft_span is not None and draft_span.end_ns is not None else round_start_ns
        arrival_ms = max(0.0, (arrival_ns - int(round_start_ns)) / 1_000_000)
        prefix_ids = proposal.metadata.get("prefix_ids") or []
        arrivals.append(
            VerificationArrival(
                arrival_id=f"{order_index}:{proposal.proposal_id}",
                request_id=proposal.request_id,
                device_id=str(proposal.worker_id or proposal.metadata.get("edge_device_worker_id") or "edge-unknown"),
                arrival_ms=arrival_ms,
                draft_length=len(proposal.tokens),
                prompt_tokens=len(prefix_ids) if isinstance(prefix_ids, list) else 0,
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "worker_id": proposal.worker_id,
                    "arrival_source": "runtime_draft_completion",
                },
            )
        )

    queue_batches = StaticQueueBatchPlanner(
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        pad_to_max_length=pad_to_max_length,
    ).plan(arrivals)
    summary = summarize_queue_batches(queue_batches)

    by_request = {arrival.request_id: arrival for arrival in arrivals}
    replanned: list[VerifyBatch] = []
    for batch_index, queue_batch in enumerate(queue_batches):
        request_ids = [arrival.request_id for arrival in queue_batch.arrivals]
        proposal_ids = [
            str(arrival.metadata["proposal_id"])
            for arrival in queue_batch.arrivals
            if arrival.metadata.get("proposal_id") in proposals_by_id
        ]
        metadata = {
            "scheduler": "sled_static_queue",
            "sled_static_queue": True,
            "dispatch_reason": queue_batch.metadata.get("dispatch_reason"),
            "dispatch_ms": queue_batch.dispatch_ms,
            "arrival_ms_by_request": {
                request_id: by_request[request_id].arrival_ms
                for request_id in request_ids
                if request_id in by_request
            },
            "queue_wait_ms_by_request": dict(queue_batch.queue_wait_ms_by_request),
            "avg_queue_wait_ms": (
                sum(queue_batch.queue_wait_ms_by_request.values()) / len(queue_batch.queue_wait_ms_by_request)
                if queue_batch.queue_wait_ms_by_request
                else 0.0
            ),
            "max_queue_wait_ms": max(queue_batch.queue_wait_ms_by_request.values(), default=0.0),
            "padded_draft_length": queue_batch.padded_draft_length,
            "token_slots": queue_batch.token_slots,
            "padding_token_count": queue_batch.padding_token_count,
            "target_batch_size": batch_size,
            "pad_to_max_length": pad_to_max_length,
            "max_wait_ms": max_wait_ms,
        }
        replanned.append(
            VerifyBatch(
                batch_id=f"{plan_id}:sled-static-batch{batch_index}",
                request_ids=request_ids,
                proposal_ids=proposal_ids,
                metadata=metadata,
            )
        )

    planning_start_ns = recorder.clock()
    planning_span = recorder.record_completed(
        phase="scheduler.sled_static_queue",
        method=method,
        plan_id=plan_id,
        run_id=run_id,
        round_id=round_id,
        shared=True,
        start_ns=planning_start_ns,
        end_ns=recorder.clock(),
        metadata={
            "sled_static_queue": True,
            "batch_size": batch_size,
            "max_wait_ms": max_wait_ms,
            "pad_to_max_length": pad_to_max_length,
            "arrival_count": len(arrivals),
            "batch_count": len(replanned),
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "request_ids": list(batch.request_ids),
                    "proposal_ids": list(batch.proposal_ids),
                    "metadata": dict(batch.metadata or {}),
                }
                for batch in replanned
            ],
            "queue_summary": summary,
        },
    )
    _record_span_event(
        logger,
        recorder,
        planning_span,
        span_kind="leaf",
        attribution="system",
        metadata=dict(planning_span.metadata),
    )
    return replanned


@dataclass(frozen=True)
class VerificationAttemptOutcome:
    verification_results: list[Any]
    proposals: list[CandidateProposal]
    proactive_by_proposal: dict[str, CandidateProposal | None]


def _await_verification_results(
    future: Any,
    *,
    executor: ThreadPoolExecutor,
    verifier: VerifierBackend,
    logger: EventLogger,
    recorder: TimingRecorder,
    context: RuntimeContext,
    method: str,
    plan_id: str,
    run_id: str,
    round_id: int,
    batch_id: str,
    proposals: list[CandidateProposal],
    proactive_by_proposal: dict[str, CandidateProposal | None],
    verify_submit_ns: int,
) -> VerificationAttemptOutcome:
    timeout_ms = _optional_float(context.method_config.get("sled_verify_timeout_ms"))
    if timeout_ms is None:
        timeout_ms = _optional_float(context.method_config.get("verify_timeout_ms"))
    if timeout_ms is None or timeout_ms <= 0:
        return VerificationAttemptOutcome(
            verification_results=future.result(),
            proposals=proposals,
            proactive_by_proposal=proactive_by_proposal,
        )

    retry_limit = max(
        0,
        _optional_int(
            context.method_config.get("sled_retry_count"),
            default=_optional_int(context.method_config.get("verify_retry_count"), default=0),
        ),
    )
    fallback_threshold = max(
        0,
        _optional_int(
            context.method_config.get("sled_fallback_failure_threshold"),
            default=_optional_int(context.method_config.get("fallback_failure_threshold"), default=0),
        ),
    )
    fallback_enabled = _config_bool(context.method_config.get("sled_enable_fallback_release"), False)
    timeout_ns = int(round(timeout_ms * 1_000_000))
    attempt = 0
    attempt_start_ns = int(verify_submit_ns)
    consecutive_timeouts = 0
    current_future = future
    current_proposals = list(proposals)
    current_proactive_by_proposal = dict(proactive_by_proposal)
    while True:
        now_ns = recorder.clock()
        remaining_ns = max(0, attempt_start_ns + timeout_ns - now_ns)
        try:
            return VerificationAttemptOutcome(
                verification_results=current_future.result(timeout=remaining_ns / 1_000_000_000),
                proposals=current_proposals,
                proactive_by_proposal=current_proactive_by_proposal,
            )
        except TimeoutError:
            consecutive_timeouts += 1
            timeout_end_ns = max(recorder.clock(), attempt_start_ns + timeout_ns)
            _record_control_event(
                logger,
                recorder,
                phase="verify.timeout",
                method=method,
                plan_id=plan_id,
                run_id=run_id,
                round_id=round_id,
                batch_id=batch_id,
                start_ns=attempt_start_ns,
                end_ns=timeout_end_ns,
                metadata={
                    "attempt": attempt,
                    "timeout_ms": timeout_ms,
                    "consecutive_timeouts": consecutive_timeouts,
                    "request_ids": [proposal.request_id for proposal in proposals],
                    "proposal_ids": [proposal.proposal_id for proposal in proposals],
                },
            )
            if fallback_enabled and fallback_threshold > 0 and consecutive_timeouts >= fallback_threshold:
                fallback_results = _fallback_verification_results(
                    current_proposals,
                    proactive_by_proposal=current_proactive_by_proposal,
                    consecutive_timeouts=consecutive_timeouts,
                )
                _record_control_event(
                    logger,
                    recorder,
                    phase="verify.fallback_release",
                    method=method,
                    plan_id=plan_id,
                    run_id=run_id,
                    round_id=round_id,
                    batch_id=batch_id,
                    start_ns=timeout_end_ns,
                    end_ns=timeout_end_ns,
                    metadata={
                        "fallback_reason": "consecutive_verify_timeouts",
                        "consecutive_timeouts": consecutive_timeouts,
                        "fallback_failure_threshold": fallback_threshold,
                        "request_ids": [proposal.request_id for proposal in current_proposals],
                        "proposal_ids": [proposal.proposal_id for proposal in current_proposals],
                    },
                )
                current_future.cancel()
                return VerificationAttemptOutcome(
                    verification_results=fallback_results,
                    proposals=current_proposals,
                    proactive_by_proposal=current_proactive_by_proposal,
                )
            if attempt >= retry_limit:
                _record_control_event(
                    logger,
                    recorder,
                    phase="verify.retry_exhausted",
                    method=method,
                    plan_id=plan_id,
                    run_id=run_id,
                    round_id=round_id,
                    batch_id=batch_id,
                    start_ns=timeout_end_ns,
                    end_ns=timeout_end_ns,
                    metadata={
                        "retry_limit": retry_limit,
                        "consecutive_timeouts": consecutive_timeouts,
                        "fallback_enabled": fallback_enabled,
                    },
                )
                return VerificationAttemptOutcome(
                    verification_results=current_future.result(),
                    proposals=current_proposals,
                    proactive_by_proposal=current_proactive_by_proposal,
                )
            attempt += 1
            retry_proposals = _retry_proposals_with_proactive_tokens(
                current_proposals,
                proactive_by_proposal=current_proactive_by_proposal,
            )
            _record_control_event(
                logger,
                recorder,
                phase="verify.retry_enqueue",
                method=method,
                plan_id=plan_id,
                run_id=run_id,
                round_id=round_id,
                batch_id=batch_id,
                start_ns=timeout_end_ns,
                end_ns=timeout_end_ns,
                metadata={
                    "attempt": attempt,
                    "retry_limit": retry_limit,
                    "consecutive_timeouts": consecutive_timeouts,
                    "request_ids": [proposal.request_id for proposal in retry_proposals],
                    "proposal_ids": [proposal.proposal_id for proposal in retry_proposals],
                    "extended_with_proactive_count": sum(
                        1 for proposal in retry_proposals if proposal.metadata.get("retry_extended_with_proactive")
                    ),
                },
            )
            current_proposals = retry_proposals
            current_proactive_by_proposal = {
                proposal.proposal_id: None if proposal.metadata.get("retry_extended_with_proactive") else current_proactive_by_proposal.get(proposal.proposal_id)
                for proposal in current_proposals
            }
            current_future = executor.submit(verifier.verify_batch, current_proposals, context)
            attempt_start_ns = timeout_end_ns


def _fallback_verification_results(
    proposals: list[CandidateProposal],
    *,
    proactive_by_proposal: dict[str, CandidateProposal | None],
    consecutive_timeouts: int,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    for proposal in proposals:
        proactive = proactive_by_proposal.get(proposal.proposal_id)
        fallback_tokens = list(proposal.tokens)
        results.append(
            VerificationResult(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                shape=proposal.shape,
                accepted_prefix_len=len(fallback_tokens),
                verified_tokens=list(fallback_tokens),
                bonus_token=None,
                timing={
                    "response_timing": {
                        "backend_name": "sled_local_fallback",
                        "batch_size": len(proposals),
                        "fallback_release": True,
                    }
                },
                payload={
                    "accepted_prefix_len": len(fallback_tokens),
                    "verified_tokens": list(fallback_tokens),
                    "bonus_token": None,
                    "fallback_release": True,
                },
                metadata={
                    "backend_name": "sled_local_fallback",
                    "fallback_release": True,
                    "fallback_released_token_count": len(fallback_tokens),
                    "fallback_had_proactive_tokens": proactive is not None,
                    "fallback_proactive_token_count": 0 if proactive is None else len(proactive.tokens),
                    "consecutive_timeouts": consecutive_timeouts,
                },
            )
        )
    return results


def _retry_proposals_with_proactive_tokens(
    proposals: list[CandidateProposal],
    *,
    proactive_by_proposal: dict[str, CandidateProposal | None],
) -> list[CandidateProposal]:
    retry_proposals: list[CandidateProposal] = []
    for proposal in proposals:
        proactive = proactive_by_proposal.get(proposal.proposal_id)
        if proposal.shape != "linear" or proactive is None:
            retry_proposals.append(proposal)
            continue
        proposal_prefix = [int(token_id) for token_id in proposal.metadata.get("prefix_ids", [])]
        expected_proactive_prefix = [*proposal_prefix, *[int(token_id) for token_id in proposal.tokens]]
        proactive_prefix = [int(token_id) for token_id in proactive.metadata.get("prefix_ids", [])]
        if proactive_prefix != expected_proactive_prefix:
            retry_proposals.append(proposal)
            continue
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "retry_extended_with_proactive": True,
                "retry_original_draft_length": len(proposal.tokens),
                "retry_proactive_token_count": len(proactive.tokens),
                "retry_parent_proposal_id": proposal.proposal_id,
                "allow_bonus": bool(metadata.get("allow_bonus", False)),
            }
        )
        retry_proposals.append(
            replace(
                proposal,
                tokens=[*proposal.tokens, *[int(token_id) for token_id in proactive.tokens]],
                draft_length=len(proposal.tokens) + len(proactive.tokens),
                metadata=metadata,
            )
        )
    return retry_proposals


def _record_control_event(
    logger: EventLogger,
    recorder: TimingRecorder,
    *,
    phase: str,
    method: str,
    plan_id: str,
    run_id: str,
    round_id: int,
    batch_id: str,
    start_ns: int,
    end_ns: int,
    metadata: dict[str, Any],
) -> None:
    span = recorder.record_completed(
        phase=phase,
        method=method,
        plan_id=plan_id,
        run_id=run_id,
        round_id=round_id,
        request_id=batch_id,
        batch_id=batch_id,
        shared=True,
        start_ns=start_ns,
        end_ns=end_ns,
        metadata=metadata,
    )
    _record_span_event(
        logger,
        recorder,
        span,
        span_kind="leaf",
        attribution="batch",
        metadata=dict(span.metadata),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return int(default)
    return int(value)


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _record_verify_batch_span(
    logger: EventLogger,
    recorder: TimingRecorder,
    verification_results: list[Any],
    *,
    fallback_start_ns: int,
    method: str,
    plan_id: str,
    run_id: str,
    round_id: int,
    batch_id: str,
    proposals: list[CandidateProposal],
    batch_metadata: dict[str, Any] | None = None,
) -> TimingSpan:
    starts: list[int] = []
    ends: list[int] = []
    metadata: dict[str, Any] = {
        "request_ids": [proposal.request_id for proposal in proposals],
        "proposal_ids": [proposal.proposal_id for proposal in proposals],
        **dict(batch_metadata or {}),
    }
    for result in verification_results:
        result_metadata = dict(getattr(result, "metadata", {}) or {})
        if result_metadata.get("fallback_release"):
            metadata["fallback_release"] = True
            metadata["fallback_release_count"] = int(metadata.get("fallback_release_count") or 0) + 1
            metadata["fallback_released_token_count"] = int(metadata.get("fallback_released_token_count") or 0) + int(
                result_metadata.get("fallback_released_token_count") or 0
            )
        for event in result.timing.get("client_events", []):
            if event.get("phase") == "verify.http_total":
                starts.append(int(event["start_ns"]))
                ends.append(int(event["end_ns"]))
                metadata.update(dict(event.get("metadata") or {}))
        response_timing = dict(result.timing.get("response_timing") or {})
        if response_timing:
            metadata["response_timing"] = response_timing
    start_ns = min(starts) if starts else int(fallback_start_ns)
    end_ns = max(ends) if ends else perf_counter_ns()
    span = recorder.record_completed(
        phase="verify.batch_total",
        method=method,
        plan_id=plan_id,
        run_id=run_id,
        round_id=round_id,
        request_id=batch_id,
        batch_id=batch_id,
        shared=True,
        start_ns=start_ns,
        end_ns=end_ns,
        metadata=metadata,
    )
    _record_span_event(
        logger,
        recorder,
        span,
        span_kind="leaf",
        attribution="batch",
        tokens_in=sum(len(proposal.tokens) for proposal in proposals),
    )
    return span


def _observe_planning_policy(
    planning_policy: PlanningPolicy | None,
    draft_spans_by_request: dict[str, TimingSpan],
    proactive_spans: list[TimingSpan],
    verification_results: list[Any],
) -> None:
    observe = getattr(planning_policy, "observe", None)
    if observe is None:
        return
    draft_ms = sum(float(span.measured_duration_ms) for span in draft_spans_by_request.values() if span.end_ns is not None)
    draft_ms += sum(float(span.measured_duration_ms) for span in proactive_spans if span.end_ns is not None)
    draft_tokens = max(1, len(draft_spans_by_request) + len(proactive_spans))
    server_ms_values: list[float] = []
    residual_values: list[float] = []
    for result in verification_results:
        timing = dict(result.timing.get("response_timing") or {})
        server_ms = timing.get("server_batch_total_ms", timing.get("server_total_ms"))
        if server_ms is not None:
            server_ms_values.append(float(server_ms))
        residual = result.timing.get("network_or_queue_residual_ms")
        if residual is not None:
            residual_values.append(float(residual))
    observe(
        draft_ms_per_token=draft_ms / draft_tokens if draft_tokens else None,
        server_verify_ms=max(server_ms_values) if server_ms_values else None,
        network_residual_ms=max(residual_values) if residual_values else None,
    )


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


def _draft_runner_metadata(draft_runners: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        worker_id: dict(getattr(runner, "metadata", {}) or {})
        for worker_id, runner in draft_runners.items()
    }


def _plan_metadata(plan: Any, hints: PlanHints) -> dict[str, Any]:
    return {
        "plan_metadata": dict(getattr(plan, "metadata", {}) or {}),
        "hints_metadata": dict(getattr(hints, "metadata", {}) or {}),
        "planning_hints": dict(getattr(hints, "metadata", {}) or {}),
        "draft_jobs": [
            {
                "request_id": job.request_id,
                "worker_id": job.worker_id,
                "max_tokens": job.budget.max_tokens,
                "max_branches": job.budget.max_branches,
                "metadata": dict(job.metadata or {}),
            }
            for job in getattr(plan, "draft_jobs", [])
        ],
        "verify_batches": [
            {
                "batch_id": batch.batch_id,
                "request_ids": list(batch.request_ids),
                "metadata": dict(batch.metadata or {}),
            }
            for batch in getattr(plan, "verify_batches", [])
        ],
        "worker_preferences": dict(getattr(hints, "worker_preferences", {}) or {}),
        "candidate_worker_preferences": {
            request_id: list(worker_ids)
            for request_id, worker_ids in dict(getattr(hints, "candidate_worker_preferences", {}) or {}).items()
        },
        "candidate_draft_lengths": {
            request_id: dict(worker_lengths or {})
            for request_id, worker_lengths in dict(getattr(hints, "candidate_draft_lengths", {}) or {}).items()
        },
        "draft_lengths": dict(getattr(hints, "draft_lengths", {}) or {}),
        "preferred_batches": [
            list(batch)
            for batch in getattr(hints, "preferred_batches", [])
        ],
    }


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
        span = recorder.record_completed(
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
            span,
            span_kind="detail",
            attribution=attribution,
            metadata=metadata,
        )
