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
from specplatform.runtime.draft_execution import execute_draft_jobs
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
                        resources=SchedulerResources(
                            draft_worker_ids=list(draft_runners),
                            draft_worker_metadata=_draft_runner_metadata(draft_runners),
                        ),
                        hints=hints,
                        context=context,
                    )
                    scheduler_span.metadata.update(_plan_metadata(plan, hints))
                self._record_span_event(
                    logger,
                    recorder,
                    scheduler_span,
                    span_kind="leaf",
                    attribution="system",
                )
                sessions_by_id = {session.request_id: session for session in active_sessions}
                proposals_by_request: dict[str, list[CandidateProposal]] = {}
                draft_spans_by_proposal: dict[str, TimingSpan] = {}
                draft_results = execute_draft_jobs(
                    jobs=list(plan.draft_jobs),
                    sessions_by_id=sessions_by_id,
                    draft_runners=draft_runners,
                    candidate_strategy=self.candidate_strategy,
                    context=context,
                    clock=recorder.clock,
                )
                for draft_result in draft_results:
                    job = draft_result.job
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
                    self._record_span_event(
                        logger,
                        recorder,
                        draft_span,
                        span_kind="leaf",
                        attribution="request",
                        tokens_out=len(proposal.tokens),
                        metadata=dict(proposal.metadata),
                    )
                    self._record_detail_event_specs(
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
                        tokens_out=1,
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
                    for verification_result in verification_results:
                        response_timing = dict(verification_result.timing.get("response_timing") or {})
                        self._record_detail_event_specs(
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
                    for request_id, request_proposals in _group_proposals_by_request(proposals).items():
                        accept_records = []
                        for proposal in request_proposals:
                            verification_result = verification_results_by_id[proposal.proposal_id]
                            proposal = proposals_by_id[verification_result.proposal_id]
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
                            self._record_span_event(
                                logger,
                                recorder,
                                accept_span,
                                span_kind="leaf",
                                attribution="request",
                                metadata=dict(accept_span.metadata),
                            )

                        proposal, _verification_result, accept_result, accept_span = winner
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
                        draft_span = draft_spans_by_proposal[proposal.proposal_id]
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
            resources={
                "draft_worker_ids": list(draft_runners),
                "draft_worker_metadata": _draft_runner_metadata(draft_runners),
            },
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

    def _record_detail_event_specs(
        self,
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
        """把 draft/http 内部细粒度计时转成 non-leaf detail events。"""
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
            self._record_span_event(
                logger,
                recorder,
                detail_span,
                span_kind="detail",
                attribution=attribution,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                metadata=metadata,
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


def _group_proposals_by_request(proposals: list[CandidateProposal]) -> dict[str, list[CandidateProposal]]:
    grouped: dict[str, list[CandidateProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.request_id, []).append(proposal)
    return grouped


def _select_accept_record(records: list[tuple[Any, Any, Any, TimingSpan]]) -> tuple[Any, Any, Any, TimingSpan]:
    if not records:
        raise ValueError("Cannot select from empty accept records.")
    return max(records, key=_accept_record_score)


def _commit_acceptance_if_supported(
    record: tuple[Any, Any, Any, TimingSpan],
    *,
    acceptance_policy: Any,
    context: RuntimeContext,
    draft_runners: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, TimingSpan]:
    """Let stateful methods commit only the selected proposal."""
    commit = getattr(acceptance_policy, "commit_acceptance", None)
    if not callable(commit):
        return record
    proposal, verification_result, accept_result, accept_span = record
    committed = commit(
        proposal,
        verification_result,
        accept_result,
        context,
        draft_runners=draft_runners,
    )
    return proposal, verification_result, committed, accept_span


def _accept_record_score(record: tuple[Any, Any, Any, TimingSpan]) -> tuple[int, int, int, int]:
    proposal, _verification_result, accept_result, _accept_span = record
    accepted_count = int(accept_result.metadata.get("accepted_count", len(accept_result.accepted_tokens)) or 0)
    rejected_count = int(accept_result.metadata.get("rejected_count", len(accept_result.rejected_tokens)) or 0)
    return (
        len(accept_result.output_token_ids),
        accepted_count,
        -rejected_count,
        -int(proposal.metadata.get("candidate_index", 0) or 0),
    )
