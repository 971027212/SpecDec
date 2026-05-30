from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import CandidateProposal, PlanHints, RuntimeContext
from specplatform.methods.base import AcceptancePolicy, CandidateStrategy, PlanningPolicy
from specplatform.metrics import EventLogger
from specplatform.runtime.session import GenerationSession
from specplatform.schedulers import Scheduler, SchedulerResources
from specplatform.schedulers.batch_planner import attach_proposals_to_batches
from specplatform.timing import TimingAttributor, TimingRecorder, event_from_span
from specplatform.timing.span import TimingSpan
from specplatform.verification import VerifierBackend


@dataclass
class RuntimeRequestResult:
    request_id: str
    output_token_ids: list[int] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass
class RuntimeRunResult:
    request_results: list[RuntimeRequestResult]
    events: EventLogger


@dataclass
class RuntimeEngine:
    candidate_strategy: CandidateStrategy
    acceptance_policy: AcceptancePolicy
    scheduler: Scheduler
    verifier: VerifierBackend
    planning_policy: PlanningPolicy | None = None
    timing_recorder: TimingRecorder | None = None
    timing_attributor: TimingAttributor = field(default_factory=TimingAttributor)

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
        method = _method_label(context)
        request_results = {
            session.request_id: RuntimeRequestResult(request_id=session.request_id)
            for session in sessions
        }
        stalled: set[str] = set()
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
                        resources=SchedulerResources(draft_worker_ids=list(draft_runners)),
                        hints=hints,
                        context=context,
                    )
                self._record_span_event(
                    logger,
                    recorder,
                    scheduler_span,
                    span_kind="leaf",
                    attribution="system",
                )
                sessions_by_id = {session.request_id: session for session in active_sessions}
                proposals_by_request: dict[str, CandidateProposal] = {}
                draft_spans_by_request: dict[str, TimingSpan] = {}
                for job in plan.draft_jobs:
                    session = sessions_by_id[job.request_id]
                    runner = draft_runners[job.worker_id]
                    with recorder.span(
                        phase="draft.generate",
                        method=method,
                        plan_id=plan_id,
                        run_id=run_id,
                        round_id=round_index,
                        request_id=session.request_id,
                        session_id=session.request_id,
                        worker_id=job.worker_id,
                    ) as draft_span:
                        proposal = self.candidate_strategy.propose(session, runner, job.budget, context)
                        draft_span.proposal_id = proposal.proposal_id
                    proposals_by_request[job.request_id] = proposal
                    draft_spans_by_request[job.request_id] = draft_span
                    request_results[job.request_id].proposals.append(proposal.proposal_id)
                    self._record_span_event(
                        logger,
                        recorder,
                        draft_span,
                        span_kind="leaf",
                        attribution="request",
                        tokens_out=len(proposal.tokens),
                        metadata=dict(proposal.metadata),
                    )

                attach_proposals_to_batches(
                    plan.verify_batches,
                    {request_id: proposal.proposal_id for request_id, proposal in proposals_by_request.items()},
                )
                proposals_by_id = {
                    proposal.proposal_id: proposal
                    for proposal in proposals_by_request.values()
                }
                for batch in plan.verify_batches:
                    proposals = [
                        proposals_by_id[proposal_id]
                        for proposal_id in batch.proposal_ids
                        if proposal_id in proposals_by_id
                    ]
                    if not proposals:
                        continue
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
                            "request_ids": [proposal.request_id for proposal in proposals],
                            "proposal_ids": [proposal.proposal_id for proposal in proposals],
                        },
                    ) as verify_span:
                        verification_results = self.verifier.verify_batch(proposals, context)
                    self._record_span_event(
                        logger,
                        recorder,
                        verify_span,
                        span_kind="leaf",
                        attribution="batch",
                        tokens_in=sum(len(proposal.tokens) for proposal in proposals),
                    )
                    for event in self.timing_attributor.attribute_batch_average(
                        parent_span=verify_span,
                        proposals=proposals,
                        event_id_factory=recorder.next_event_id,
                    ):
                        logger.record(event)
                    for verification_result in verification_results:
                        proposal = proposals_by_id[verification_result.proposal_id]
                        session = sessions_by_id[proposal.request_id]
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
                        ) as accept_span:
                            accept_result = self.acceptance_policy.accept(proposal, verification_result, context)
                        self._record_span_event(
                            logger,
                            recorder,
                            accept_span,
                            span_kind="leaf",
                            attribution="request",
                            metadata=dict(accept_result.metadata),
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
                        ) as append_span:
                            emitted = session.append_tokens(accept_result.output_token_ids)
                        self._record_span_event(
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
                        draft_span = draft_spans_by_request[proposal.request_id]
                        generation_start = _min_start_ns(draft_span, verify_span, accept_span, append_span)
                        generation_end = _max_end_ns(draft_span, verify_span, accept_span, append_span)
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
                        self._record_span_event(
                            logger,
                            recorder,
                            generation_span,
                            span_kind="aggregate",
                            attribution="request",
                            tokens_out=len(accept_result.output_token_ids),
                            metadata=dict(accept_result.metadata),
                        )
            self._record_span_event(
                logger,
                recorder,
                round_span,
                span_kind="aggregate",
                attribution="system",
            )
            round_index += 1
        return RuntimeRunResult(
            request_results=list(request_results.values()),
            events=logger,
        )

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
            resources={"draft_worker_ids": list(draft_runners)},
            history={},
            context=context,
        )

    def _record_span_event(
        self,
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


def _method_label(context: RuntimeContext) -> str:
    return str(context.run_config.get("method", "unified_runtime"))


def _min_start_ns(*spans: TimingSpan) -> int:
    return min(span.start_ns for span in spans)


def _max_end_ns(*spans: TimingSpan) -> int:
    ends = [span.end_ns for span in spans if span.end_ns is not None]
    if len(ends) != len(spans):
        raise ValueError("Cannot aggregate unfinished TimingSpan.")
    return max(int(end) for end in ends)
