from __future__ import annotations

"""统一 speculative runtime 引擎。

RuntimeEngine 只编排通用流程：调 scheduler、执行 draft job、批量 verifier、
调用 acceptance policy、写回 session、记录 timing/metrics。它不应该根据
method 名称写分支。
"""

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
    """单个 request 的 runtime 输出摘要。"""

    request_id: str
    output_token_ids: list[int] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass
class RuntimeRunResult:
    """一次 runtime.run 的聚合结果。"""

    request_results: list[RuntimeRequestResult]
    events: EventLogger


@dataclass
class RuntimeEngine:
    """统一运行引擎，通过注入策略对象保持 method 无关。"""

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
        """执行多 request 的统一 draft/verify/accept/append 循环。"""
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
                    verification_results_by_id = _validate_verification_results(
                        proposals,
                        verification_results,
                    )
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
                    for proposal in proposals:
                        verification_result = verification_results_by_id[proposal.proposal_id]
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
        """从可选 planning policy 读取 scheduler hint。"""
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
        """把 TimingSpan 转成 PhaseEvent 并写入 logger。"""
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
    """从 run_config 中取展示用 method 名称；不用于 runtime 分支。"""
    return str(context.run_config.get("method", "unified_runtime"))


def _min_start_ns(*spans: TimingSpan) -> int:
    """聚合多个 span 的最早开始时间。"""
    return min(span.start_ns for span in spans)


def _max_end_ns(*spans: TimingSpan) -> int:
    """聚合多个已完成 span 的最晚结束时间。"""
    ends = [span.end_ns for span in spans if span.end_ns is not None]
    if len(ends) != len(spans):
        raise ValueError("Cannot aggregate unfinished TimingSpan.")
    return max(int(end) for end in ends)


def _validate_verification_results(
    proposals: list[CandidateProposal],
    verification_results: list[Any],
) -> dict[str, Any]:
    """确保 verifier 对每个 proposal 恰好返回一个对应结果。

    真实 HTTP/batch verifier 可能出现部分失败、重复响应或未知 proposal_id。
    runtime 在进入 acceptance 前 fail fast，避免某个 request 永远不 append 而反复进入下一轮。
    """
    expected_ids = [proposal.proposal_id for proposal in proposals]
    expected = set(expected_ids)
    results_by_id: dict[str, Any] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for result in verification_results:
        proposal_id = str(result.proposal_id)
        if proposal_id not in expected:
            unknown.append(proposal_id)
            continue
        if proposal_id in results_by_id:
            duplicates.append(proposal_id)
            continue
        results_by_id[proposal_id] = result

    missing = [proposal_id for proposal_id in expected_ids if proposal_id not in results_by_id]
    if missing or duplicates or unknown:
        raise ValueError(
            "Verifier returned invalid result set: "
            f"missing={missing}, duplicates={duplicates}, unknown={unknown}."
        )
    return results_by_id
