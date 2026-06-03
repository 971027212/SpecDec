from __future__ import annotations

"""SLED primitives faithful to the edge-device / server-batch paper flow."""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import AcceptResult, CandidateProposal, DraftBudget, PlanHints, RuntimeContext, VerificationResult
from specplatform.draft import DraftGeneration
from specplatform.methods.base import CandidateStrategy, PlanningPolicy, ProactiveDraftPolicy, ReconcilePolicy, ReconcileResult
from specplatform.methods.planning_math import (
    _batch_size,
    _batches_by_ready_time,
    _clamp_depth,
    _resource_worker_ids,
    _resource_worker_metadata,
    _speed_profile,
    _worker_latency_ms,
    _worker_ms_per_token,
)


@dataclass
class SLEDDynamicCandidateStrategy(CandidateStrategy):
    """Generate one edge-device draft sequence with confidence-triggered stop."""

    proposal_prefix: str = "sled-linear"
    confidence_threshold: float = 0.5

    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        max_tokens = min(int(budget.max_tokens), int(session.remaining_tokens))
        threshold = _confidence_threshold(context, self.confidence_threshold)
        if not hasattr(draft_runner, "generate_tokens_until_confidence_drop"):
            raise TypeError("SLED requires a draft runner with confidence-triggered dynamic drafting.")
        generation: DraftGeneration = draft_runner.generate_tokens_until_confidence_drop(
            prefix_ids=session.prefix_ids,
            max_tokens=max_tokens,
            confidence_threshold=threshold,
            request_id=session.request_id,
            metadata={
                "draft_budget": {
                    "max_tokens": budget.max_tokens,
                    "max_branches": budget.max_branches,
                    "timeout_ms": budget.timeout_ms,
                }
            },
        )

        metadata = dict(generation.metadata)
        metadata["prefix_ids"] = list(session.prefix_ids)
        metadata["remaining_tokens"] = session.remaining_tokens
        metadata["allow_bonus"] = (
            len(generation.tokens) < session.remaining_tokens
            and _config_bool(context, "sled_allow_bonus", True)
            and not _config_bool(context, "disable_bonus", False)
        )
        metadata["method"] = "sled_dynamic"
        metadata["confidence_threshold"] = threshold
        metadata["edge_device_worker_id"] = metadata.get("runner_id")
        runner_id = str(metadata.get("runner_id") or "draft")
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{runner_id}"

        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=session.request_id,
            worker_id=metadata.get("runner_id"),
            shape="linear",
            tokens=list(generation.tokens),
            draft_length=len(generation.tokens),
            timing=dict(generation.timing),
            metadata=metadata,
        )


@dataclass
class SLEDAsyncDraftPolicy(ProactiveDraftPolicy):
    """Continue edge-device drafting while server verification is in flight."""

    proposal_prefix: str = "sled-proactive"
    default_max_tokens: int = 8
    confidence_threshold: float = 0.5

    def propose_proactive(
        self,
        session: Any,
        proposal: CandidateProposal,
        draft_runner: Any,
        context: RuntimeContext,
    ) -> CandidateProposal | None:
        if proposal.shape != "linear":
            return None
        if not hasattr(draft_runner, "generate_tokens_until_confidence_drop"):
            raise TypeError("SLED async requires confidence-triggered draft runner support.")

        prefix_ids = [int(token_id) for token_id in proposal.metadata.get("prefix_ids", [])]
        if not prefix_ids:
            return None
        proactive_prefix = [*prefix_ids, *[int(token_id) for token_id in proposal.tokens]]
        remaining_tokens = int(session.max_new_tokens) - (len(proactive_prefix) - len(session.prompt_ids))
        if remaining_tokens <= 0:
            return None

        max_tokens = min(
            _config_int(context, "sled_async_proactive_tokens", self.default_max_tokens),
            remaining_tokens,
        )
        if max_tokens <= 0:
            return None
        threshold = _confidence_threshold(context, self.confidence_threshold)
        generation: DraftGeneration = draft_runner.generate_tokens_until_confidence_drop(
            prefix_ids=proactive_prefix,
            max_tokens=max_tokens,
            confidence_threshold=threshold,
            request_id=session.request_id,
            metadata={
                "proactive": True,
                "parent_proposal_id": proposal.proposal_id,
                "draft_budget": {
                    "max_tokens": max_tokens,
                    "max_branches": 1,
                    "timeout_ms": None,
                },
            },
        )
        if not generation.tokens:
            return None

        metadata = dict(generation.metadata)
        metadata.update(
            {
                "prefix_ids": proactive_prefix,
                "remaining_tokens": remaining_tokens,
                "allow_bonus": False,
                "method": "sled_async_proactive",
                "confidence_threshold": threshold,
                "edge_device_worker_id": metadata.get("runner_id"),
                "parent_proposal_id": proposal.proposal_id,
            }
        )
        runner_id = str(metadata.get("runner_id") or proposal.worker_id or "draft")
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{proposal.proposal_id}"
        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=session.request_id,
            worker_id=runner_id,
            shape="linear",
            tokens=list(generation.tokens),
            draft_length=len(generation.tokens),
            timing=dict(generation.timing),
            metadata=metadata,
        )


@dataclass
class SLEDAsyncReconcilePolicy(ReconcilePolicy):
    """Reuse locally generated async draft tokens only when prefixes align."""

    def reconcile(
        self,
        session: Any,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        accept_result: AcceptResult,
        proactive_proposal: CandidateProposal | None,
        context: RuntimeContext,
    ) -> ReconcileResult:
        del verification_result, context
        if proactive_proposal is None:
            return ReconcileResult(metadata={"reason": "no_proactive_proposal"})
        proactive_prefix = [int(token_id) for token_id in proactive_proposal.metadata.get("prefix_ids", [])]
        committed_prefix = list(session.prefix_ids)
        accepted_all_parent = (
            len(accept_result.accepted_tokens) == len(proposal.tokens)
            and not accept_result.rejected_tokens
            and accept_result.bonus_token is None
        )
        if proactive_prefix == committed_prefix and accepted_all_parent:
            return ReconcileResult(
                reused_proposal=proactive_proposal,
                reused_token_count=int(proactive_proposal.draft_length),
                aligned=True,
                metadata={
                    "reason": "prefix_aligned_after_full_parent_accept",
                    "accepted_output_tokens": list(accept_result.output_token_ids),
                    "proactive_prefix_len": len(proactive_prefix),
                    "sled_async_reuse": True,
                },
            )
        return ReconcileResult(
            discarded_token_count=int(proactive_proposal.draft_length),
            aligned=False,
            metadata={
                "reason": "prefix_mismatch_or_parent_rejected",
                "accepted_output_tokens": list(accept_result.output_token_ids),
                "parent_accepted_all": accepted_all_parent,
                "proactive_prefix_len": len(proactive_prefix),
                "committed_prefix_len": len(committed_prefix),
                "sled_async_reuse": False,
            },
        )


@dataclass
class SLEDPlanningPolicy(PlanningPolicy):
    """Assign each request to one edge device and batch server verification.

    The paper's heterogeneity comes from many edge devices, each with its own
    local draft model.  A request stays on one device; the server batches
    verification requests from many devices.  This policy therefore never emits
    candidate_worker_preferences for SLED.
    """

    min_depth: int = 1
    max_depth: int = 8
    max_speculation_tokens: int = 8
    target_batch_size: int | None = None
    confidence_threshold: float = 0.5
    default_draft_ms_per_token: float = 12.0
    _request_worker_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _assignment_cursor: int = field(default=0, init=False, repr=False)

    def plan(
        self,
        active_sessions: list[Any],
        resources: Any,
        history: Any,
        context: RuntimeContext,
    ) -> PlanHints:
        del history
        request_ids = [str(session.request_id) for session in active_sessions]
        worker_ids = _resource_worker_ids(resources)
        worker_metadata = _resource_worker_metadata(resources)
        assignment = self._edge_device_assignment(active_sessions, worker_ids, worker_metadata)
        batch_size = _batch_size(
            context,
            "sled_batch_size",
            self.target_batch_size,
            request_count=len(request_ids),
            worker_count=len(worker_ids),
        )
        preferred_batches = _batches_by_ready_time(
            request_ids,
            batch_size=batch_size,
            ready_time_ms=assignment["ready_time_ms"],
        )
        threshold = _confidence_threshold(context, self.confidence_threshold)
        return PlanHints(
            draft_lengths=assignment["draft_lengths"],
            worker_preferences=assignment["worker_preferences"],
            preferred_batches=preferred_batches,
            metadata={
                "method_family": "sled",
                "edge_device_worker_assignment": True,
                "heterogeneous_worker_assignment": True,
                "single_edge_device_per_request": True,
                "dynamic_drafting": True,
                "confidence_threshold": threshold,
                "max_speculation_tokens": int(self.max_speculation_tokens),
                "shared_server_batch_verification": True,
                "static_batch_size": batch_size,
                "worker_count": len(worker_ids),
                "estimated_worker_load_ms": dict(assignment["worker_load_ms"]),
                "estimated_ready_time_ms": dict(assignment["ready_time_ms"]),
                "request_worker_assignment": dict(assignment["worker_preferences"]),
                "assignment_trace": list(assignment["assignment_trace"]),
                "assignment_objective": "stable_edge_device_assignment_with_confidence_triggered_verification",
            },
        )

    def _edge_device_assignment(
        self,
        active_sessions: list[Any],
        worker_ids: list[str],
        worker_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not worker_ids:
            return {
                "worker_preferences": {},
                "draft_lengths": {
                    str(session.request_id): _clamp_depth(
                        self.max_speculation_tokens,
                        min_depth=self.min_depth,
                        max_depth=self.max_depth,
                        remaining=getattr(session, "remaining_tokens", None),
                    )
                    for session in active_sessions
                },
                "worker_load_ms": {},
                "ready_time_ms": {str(session.request_id): 0.0 for session in active_sessions},
                "assignment_trace": [],
            }

        live_workers = set(worker_ids)
        worker_load_ms = {worker_id: 0.0 for worker_id in worker_ids}
        worker_preferences: dict[str, str] = {}
        draft_lengths: dict[str, int] = {}
        ready_time_ms: dict[str, float] = {}
        assignment_trace: list[dict[str, Any]] = []
        weighted_workers = _weighted_worker_cycle(worker_ids, worker_metadata)
        for session in active_sessions:
            request_id = str(session.request_id)
            assigned_worker = self._request_worker_map.get(request_id)
            reused_assignment = assigned_worker in live_workers
            if not reused_assignment:
                assigned_worker = self._next_edge_worker(weighted_workers)
                self._request_worker_map[request_id] = assigned_worker
            assert assigned_worker is not None
            depth_cap = _clamp_depth(
                self.max_speculation_tokens,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                remaining=getattr(session, "remaining_tokens", None),
            )
            metadata = worker_metadata.get(assigned_worker, {})
            request_time_ms = _worker_latency_ms(metadata) + depth_cap * _worker_ms_per_token(
                metadata,
                default_ms=self.default_draft_ms_per_token,
            )
            worker_load_ms[assigned_worker] += request_time_ms
            worker_preferences[request_id] = assigned_worker
            draft_lengths[request_id] = depth_cap
            ready_time_ms[request_id] = worker_load_ms[assigned_worker]
            assignment_trace.append(
                {
                    "request_id": request_id,
                    "worker_id": assigned_worker,
                    "reused_assignment": bool(reused_assignment),
                    "max_speculation_tokens": depth_cap,
                    "estimated_ready_ms": ready_time_ms[request_id],
                }
            )
        return {
            "worker_preferences": worker_preferences,
            "draft_lengths": draft_lengths,
            "worker_load_ms": worker_load_ms,
            "ready_time_ms": ready_time_ms,
            "assignment_trace": assignment_trace,
        }

    def _next_edge_worker(self, weighted_workers: list[str]) -> str:
        if not weighted_workers:
            raise ValueError("SLEDPlanningPolicy requires at least one draft worker.")
        worker_id = weighted_workers[self._assignment_cursor % len(weighted_workers)]
        self._assignment_cursor += 1
        return worker_id


def _confidence_threshold(context: RuntimeContext, default: float) -> float:
    raw = context.method_config.get("sled_confidence_threshold")
    if raw is None:
        raw = context.method_config.get("confidence_threshold")
    threshold = float(default if raw is None else raw)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("SLED confidence_threshold must be in [0, 1].")
    return threshold


def _config_bool(context: RuntimeContext, key: str, default: bool) -> bool:
    raw = context.method_config.get(key)
    if raw is None:
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _config_int(context: RuntimeContext, key: str, default: int) -> int:
    raw = context.method_config.get(key)
    return int(default if raw is None else raw)


def _weighted_worker_cycle(worker_ids: list[str], worker_metadata: dict[str, dict[str, Any]]) -> list[str]:
    cycle: list[str] = []
    for worker_id in worker_ids:
        metadata = worker_metadata.get(worker_id, {})
        speed = _relative_speed(metadata)
        repeats = max(1, int(round(speed)))
        cycle.extend([worker_id] * repeats)
    return cycle or list(worker_ids)


def _relative_speed(metadata: dict[str, Any]) -> float:
    profile = _speed_profile(metadata)
    if profile:
        return float(profile.get("relative_speed") or 1.0)
    return 1.0
