from __future__ import annotations

"""DiP-SD planning policy that maps the paper solver into platform hints."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any

from specplatform.core import PlanHints, RuntimeContext
from specplatform.methods.base import PlanningPolicy
from specplatform.methods.dip_sd.model import (
    DiPSDModelConfig,
    DiPSDPaperDefaults,
    DiPSDScheduleEvaluation,
    DiPSDUserParams,
    draft_latency_ms,
    evaluate_schedule,
)
from specplatform.methods.dip_sd.solver import DiPSDSolver
from specplatform.methods.planning_math import (
    resource_worker_ids,
    resource_worker_metadata,
    speed_profile,
    worker_latency_ms,
    worker_ms_per_token,
)


@dataclass
class DiPSDPlanningPolicy(PlanningPolicy):
    """Paper-style DiP-SD planner.

    This policy owns only DiP-SD method decisions: batch-count scan,
    user-to-stage association and per-user integer draft length.  It returns
    ordinary ``PlanHints`` for the shared scheduler/runtime.
    """

    max_draft_length: int = 20
    initial_draft_length: int = 7
    min_batch_count: int = 2
    max_batch_count: int | None = None
    default_acceptance: float = 0.78
    default_comm_latency_ms: float = 3.0
    default_draft_c: float = 4.0305e-11
    default_draft_beta: float = 33.8151
    model_config: DiPSDModelConfig | None = None
    solver_mode: str = "enumerate"
    plan_cache_enabled: bool = True
    single_batch_small_request_threshold: int = 0
    _plan_cache: dict[tuple[Any, ...], PlanHints] = field(default_factory=dict, init=False, repr=False)
    _shape_plan_cache: dict[tuple[Any, ...], PlanHints] = field(default_factory=dict, init=False, repr=False)

    def plan(
        self,
        active_sessions: list[Any],
        resources: Any,
        history: Any,
        context: RuntimeContext,
    ) -> PlanHints:
        if not active_sessions:
            return PlanHints(metadata={"method_family": "dip_sd", "solver_active": False})
        worker_ids = resource_worker_ids(resources)
        worker_metadata = resource_worker_metadata(resources)
        speed_aware_assignment = _config_bool(
            context.method_config.get("dip_sd_speed_aware_worker_assignment", True)
        )
        worker_assignment_order = _worker_assignment_order(
            worker_ids,
            worker_metadata,
            speed_aware=speed_aware_assignment,
            default_ms_per_token=self.default_draft_beta,
        )
        worker_preferences = _assign_user_devices(
            active_sessions,
            worker_assignment_order,
            history=history,
            context=context,
        )
        users = [
            self._user_params(
                session,
                worker_preferences.get(str(session.request_id)),
                worker_metadata,
                history,
                context,
            )
            for session in active_sessions
        ]
        acceptance_feedback = {
            user.request_id: dict(user.metadata.get("acceptance_feedback") or {})
            for user in users
            if user.metadata.get("acceptance_feedback")
        }
        acceptance_feedback_enabled = _config_bool(
            context.method_config.get("dip_sd_acceptance_feedback_enabled", True)
        )
        model_config = self._model_config(context)
        cache_enabled = bool(context.method_config.get("dip_sd_plan_cache_enabled", self.plan_cache_enabled))
        cache_key = self._cache_key(active_sessions, worker_ids, worker_preferences, users, model_config, context)
        requested_solver_mode = str(context.method_config.get("dip_sd_solver", self.solver_mode))
        shape_cache_enabled = _config_bool(
            context.method_config.get("dip_sd_shape_plan_cache_enabled", True)
        )
        shape_cache_key = self._shape_cache_key(
            active_sessions,
            worker_ids,
            users,
            model_config,
            context,
        )
        no_online_solver = _config_bool(context.method_config.get("dip_sd_no_online_solver", False)) or (
            requested_solver_mode in {"offline_table", "offline_heuristic", "no_online_heuristic"}
        )
        offline_shape_key, offline_entry = _offline_plan_table_entry(
            active_sessions,
            worker_ids,
            context,
        )
        if cache_enabled and cache_key in self._plan_cache:
            cached = _clone_hints(self._plan_cache[cache_key])
            cached.metadata["solver_cache_hit"] = True
            cached.metadata["solver_cache_key"] = _cache_key_label(cache_key)
            cached.metadata["online_solver_enabled"] = not no_online_solver
            cached.metadata["offline_plan_table_hit"] = False
            cached.metadata["offline_plan_shape_key"] = offline_shape_key
            cached.metadata["acceptance_feedback_enabled"] = acceptance_feedback_enabled
            cached.metadata["acceptance_feedback_by_request"] = acceptance_feedback
            cached.metadata["acceptance_feedback_applied_count"] = len(acceptance_feedback)
            cached.metadata["solver_cache_shape_level"] = False
            return cached
        if cache_enabled and shape_cache_enabled and shape_cache_key in self._shape_plan_cache:
            cached = _rebind_shape_cached_hints(
                self._shape_plan_cache[shape_cache_key],
                active_sessions=active_sessions,
                users=users,
                worker_preferences=worker_preferences,
                worker_assignment_order=worker_assignment_order,
                model_config=model_config,
                context=context,
                cache_key=shape_cache_key,
                offline_shape_key=offline_shape_key,
                requested_solver_mode=requested_solver_mode,
                no_online_solver=no_online_solver,
                acceptance_feedback_enabled=acceptance_feedback_enabled,
                acceptance_feedback=acceptance_feedback,
                speed_aware_assignment=speed_aware_assignment,
            )
            if cached is not None:
                return cached
        if no_online_solver:
            if offline_entry is not None:
                return _offline_plan_hints(
                    active_sessions=active_sessions,
                    users=users,
                    worker_preferences=worker_preferences,
                    worker_assignment_order=worker_assignment_order,
                    model_config=model_config,
                    context=context,
                    entry=offline_entry,
                    shape_key=offline_shape_key,
                    acceptance_feedback_enabled=acceptance_feedback_enabled,
                    acceptance_feedback=acceptance_feedback,
                    speed_aware_assignment=speed_aware_assignment,
                )
            fallback_mode = _no_online_solver_fallback_mode(context, requested_solver_mode)
            if fallback_mode == "error":
                raise ValueError(
                    f"DiP-SD no-online-solver mode requires an offline plan table entry for {offline_shape_key!r}."
                )
            return _heuristic_no_online_plan_hints(
                active_sessions=active_sessions,
                users=users,
                worker_preferences=worker_preferences,
                worker_assignment_order=worker_assignment_order,
                model_config=model_config,
                context=context,
                shape_key=offline_shape_key,
                requested_solver_mode=requested_solver_mode,
                acceptance_feedback_enabled=acceptance_feedback_enabled,
                acceptance_feedback=acceptance_feedback,
                speed_aware_assignment=speed_aware_assignment,
                source=fallback_mode,
            )
        solver = DiPSDSolver(
            max_draft_length=int(context.method_config.get("dip_sd_max_draft_length", self.max_draft_length)),
            initial_draft_length=int(context.method_config.get("dip_sd_initial_draft_length", self.initial_draft_length)),
            min_batch_count=int(context.method_config.get("dip_sd_min_batch_count", self.min_batch_count)),
            max_batch_count=_optional_int(context.method_config.get("dip_sd_max_batch_count"), self.max_batch_count),
            solver_mode=requested_solver_mode,
        )
        solution = solver.solve(users, model_config)
        draft_lengths, adaptive_length_metadata = _maybe_adapt_draft_lengths(
            users,
            dict(solution.draft_lengths),
            context,
        )
        effective_batches = [list(batch) for batch in solution.batches]
        effective_evaluation = evaluate_schedule(
            users=users,
            batches=effective_batches,
            draft_lengths=draft_lengths,
            config=model_config,
        )
        single_batch_threshold = int(
            context.method_config.get(
                "dip_sd_single_batch_small_request_threshold",
                self.single_batch_small_request_threshold,
            )
            or 0
        )
        hybrid_single_batch_applied = False
        hybrid_single_batch_reason: str | None = None
        if 1 < len(users) <= single_batch_threshold:
            single_batch = [[user.request_id for user in users]]
            single_batch_evaluation = evaluate_schedule(
                users=users,
                batches=single_batch,
                draft_lengths=draft_lengths,
                config=model_config,
            )
            if single_batch_evaluation.feasible:
                effective_batches = single_batch
                effective_evaluation = single_batch_evaluation
                hybrid_single_batch_applied = True
                hybrid_single_batch_reason = "small_request_batching"
            else:
                hybrid_single_batch_reason = single_batch_evaluation.reason or "single_batch_infeasible"
        effective_batches, ready_rebatch_metadata = _maybe_ready_aware_rebatch(
            users=users,
            batches=effective_batches,
            draft_lengths=draft_lengths,
            config=model_config,
            context=context,
        )
        if ready_rebatch_metadata["ready_aware_rebatch_applied"]:
            effective_evaluation = evaluate_schedule(
                users=users,
                batches=effective_batches,
                draft_lengths=draft_lengths,
                config=model_config,
            )
        preferred_batch_metadata = [
            {
                "stage_index": metric.stage_index,
                "planned_batch_count": len(effective_batches),
                "max_draft_len": metric.max_draft_len,
                "max_prefix_len": metric.max_prefix_len,
                "estimated_verify_ms": metric.verify_ms,
                "estimated_memory_bytes": metric.memory_bytes,
                "estimated_draft_complete_ms": metric.draft_complete_ms,
                "estimated_stage_duration_ms": metric.stage_duration_ms,
                "request_ids": list(metric.request_ids),
            }
            for metric in effective_evaluation.batch_metrics
        ]
        hints = PlanHints(
            draft_lengths=dict(draft_lengths),
            worker_preferences=worker_preferences,
            preferred_batches=[list(batch) for batch in effective_batches],
            metadata={
                "method_family": "dip_sd",
                "solver_active": True,
                "online_solver_enabled": True,
                "offline_plan_table_hit": False,
                "offline_plan_shape_key": offline_shape_key,
                "solver_mode": solution.solver_mode,
                "requested_solver_mode": solution.requested_solver_mode,
                "solver_backend_name": solution.backend_name,
                "paper_solver_complete": solution.paper_solver_complete,
                "solver_backend_fallback_used": solution.backend_fallback_used,
                "solver_backend_fallback_reason": solution.backend_fallback_reason,
                "solver_backend_info": dict(solution.backend_info),
                "joint_batch_assignment": True,
                "joint_draft_length": True,
                "phase_level_pipeline_required": True,
                "distributed_local_drafting": True,
                "central_batch_verification": True,
                "planned_batch_count": len(effective_batches),
                "solver_planned_batch_count": solution.batch_count,
                "hybrid_single_batch_threshold": single_batch_threshold,
                "hybrid_single_batch_applied": hybrid_single_batch_applied,
                "hybrid_single_batch_reason": hybrid_single_batch_reason,
                "estimated_throughput_tokens_per_ms": effective_evaluation.throughput_tokens_per_ms,
                "estimated_throughput_tokens_per_s": effective_evaluation.throughput_tokens_per_ms * 1000.0,
                "estimated_expected_tokens": effective_evaluation.expected_tokens,
                "estimated_pipeline_span_ms": effective_evaluation.pipeline_span_ms,
                "estimated_verification_sum_ms": effective_evaluation.verification_sum_ms,
                "estimated_max_draft_verify_ms": effective_evaluation.max_draft_verify_ms,
                "dip_sd_model_config": asdict(model_config),
                "latency_calibration_profile": context.method_config.get("dip_sd_calibration_profile"),
                "latency_calibration_enabled": context.method_config.get("dip_sd_calibration_enabled"),
                "latency_calibration_applied": context.method_config.get("dip_sd_calibration_applied"),
                "latency_calibration_overrides": dict(
                    context.method_config.get("dip_sd_calibration_overrides") or {}
                ),
                "dip_sd_solution": solution.to_trace_dict(),
                "preferred_batch_metadata": preferred_batch_metadata,
                **adaptive_length_metadata,
                **ready_rebatch_metadata,
                "worker_preferences": dict(worker_preferences),
                "speed_aware_worker_assignment": speed_aware_assignment,
                "worker_assignment_order": list(worker_assignment_order),
                "acceptance_feedback_enabled": acceptance_feedback_enabled,
                "acceptance_feedback_by_request": acceptance_feedback,
                "acceptance_feedback_applied_count": len(acceptance_feedback),
                "acceptance_cache_bucket": _acceptance_cache_bucket(context),
                "solver_cache_shape_level": False,
                "shape_cache_key": _shape_cache_key_label(shape_cache_key),
                "shape_cache_request_order": [str(session.request_id) for session in active_sessions],
                "solver_preferred_batches": [list(batch) for batch in solution.batches],
                "solver_cache_hit": False,
                "solver_cache_key": _cache_key_label(cache_key),
                "assignment_objective": "maximize_expected_accepted_tokens_per_pipeline_span",
            },
        )
        if cache_enabled:
            self._plan_cache[cache_key] = _clone_hints(hints)
            if shape_cache_enabled:
                self._shape_plan_cache[shape_cache_key] = _clone_hints(hints)
        return hints

    def _cache_key(
        self,
        active_sessions: list[Any],
        worker_ids: list[str],
        worker_preferences: dict[str, str],
        users: list[DiPSDUserParams],
        model_config: DiPSDModelConfig,
        context: RuntimeContext,
    ) -> tuple[Any, ...]:
        request_part = tuple(
            (
                str(session.request_id),
                len(getattr(session, "prompt_ids", []) or []),
                int(getattr(session, "max_new_tokens", 0) or 0),
                str(worker_preferences.get(str(session.request_id), "")),
            )
            for session in active_sessions
        )
        user_part = tuple(
            (
                user.request_id,
                user.worker_id,
                round(float(user.acceptance), 8),
                round(float(user.comm_latency_ms), 8),
                round(float(user.draft_c), 18),
                round(float(user.draft_beta), 8),
            )
            for user in users
        )
        config_part = (
            int(context.method_config.get("dip_sd_max_draft_length", self.max_draft_length)),
            int(context.method_config.get("dip_sd_initial_draft_length", self.initial_draft_length)),
            int(context.method_config.get("dip_sd_min_batch_count", self.min_batch_count)),
            _optional_int(context.method_config.get("dip_sd_max_batch_count"), self.max_batch_count),
            str(context.method_config.get("dip_sd_solver", self.solver_mode)),
            _config_bool(context.method_config.get("dip_sd_no_online_solver", False)),
            _config_bool(context.method_config.get("dip_sd_speed_aware_worker_assignment", True)),
            str(context.method_config.get("dip_sd_worker_assignment_strategy", "latency_first")),
            _config_bool(context.method_config.get("dip_sd_prefetch_sticky_worker_enabled", True)),
            _config_bool(context.method_config.get("dip_sd_use_measured_worker_latency", True)),
            _config_bool(context.method_config.get("dip_sd_adaptive_draft_length_enabled", True)),
            int(context.method_config.get("dip_sd_min_draft_length", 1) or 1),
            round(float(context.method_config.get("dip_sd_adaptive_length_target_acceptance", 0.78) or 0.78), 8),
            round(float(context.method_config.get("dip_sd_adaptive_length_min_factor", 0.35) or 0.35), 8),
            _config_bool(context.method_config.get("dip_sd_ready_aware_rebatch_enabled", True)),
            round(float(context.method_config.get("dip_sd_ready_aware_rebatch_min_spread_ms", 5.0) or 0.0), 8),
            _acceptance_cache_bucket(context),
            int(context.method_config.get("dip_sd_single_batch_small_request_threshold", self.single_batch_small_request_threshold) or 0),
            round(float(context.method_config.get("dip_sd_slow_worker_length_threshold", 1.75) or 1.75), 8),
            round(float(context.method_config.get("dip_sd_slow_worker_length_multiplier", 1.25) or 1.25), 8),
            round(float(model_config.verify_c), 18),
            round(float(model_config.verify_beta), 8),
            round(float(model_config.memory_cap_bytes), 1),
        )
        return (
            request_part,
            tuple(str(worker_id) for worker_id in worker_ids),
            user_part,
            config_part,
        )

    def _shape_cache_key(
        self,
        active_sessions: list[Any],
        worker_ids: list[str],
        users: list[DiPSDUserParams],
        model_config: DiPSDModelConfig,
        context: RuntimeContext,
    ) -> tuple[Any, ...]:
        prefix_bucket = int(context.method_config.get("dip_sd_offline_plan_prefix_bucket", 32) or 0)
        max_prefix_len = max((user.prefix_len for user in users), default=0)
        prefix_part = _ceil_bucket(max_prefix_len, prefix_bucket) if prefix_bucket > 0 else max_prefix_len
        max_new = max((int(getattr(session, "max_new_tokens", 0) or 0) for session in active_sessions), default=0)
        max_remaining = max((int(getattr(session, "remaining_tokens", 0) or 0) for session in active_sessions), default=0)
        acceptance_part = tuple(
            sorted(_acceptance_cache_value(float(user.acceptance), context) for user in users)
        )
        worker_latency_part = tuple(
            sorted(round(float(user.draft_beta), 8) for user in users)
        )
        config_part = (
            int(context.method_config.get("dip_sd_max_draft_length", self.max_draft_length)),
            int(context.method_config.get("dip_sd_initial_draft_length", self.initial_draft_length)),
            int(context.method_config.get("dip_sd_min_batch_count", self.min_batch_count)),
            _optional_int(context.method_config.get("dip_sd_max_batch_count"), self.max_batch_count),
            str(context.method_config.get("dip_sd_solver", self.solver_mode)),
            _config_bool(context.method_config.get("dip_sd_no_online_solver", False)),
            _config_bool(context.method_config.get("dip_sd_speed_aware_worker_assignment", True)),
            str(context.method_config.get("dip_sd_worker_assignment_strategy", "latency_first")),
            _config_bool(context.method_config.get("dip_sd_prefetch_sticky_worker_enabled", True)),
            _config_bool(context.method_config.get("dip_sd_use_measured_worker_latency", True)),
            _config_bool(context.method_config.get("dip_sd_adaptive_draft_length_enabled", True)),
            int(context.method_config.get("dip_sd_min_draft_length", 1) or 1),
            round(float(context.method_config.get("dip_sd_adaptive_length_target_acceptance", 0.78) or 0.78), 8),
            round(float(context.method_config.get("dip_sd_adaptive_length_min_factor", 0.35) or 0.35), 8),
            _config_bool(context.method_config.get("dip_sd_ready_aware_rebatch_enabled", True)),
            round(float(context.method_config.get("dip_sd_ready_aware_rebatch_min_spread_ms", 5.0) or 0.0), 8),
            _acceptance_cache_bucket(context),
            int(context.method_config.get("dip_sd_single_batch_small_request_threshold", self.single_batch_small_request_threshold) or 0),
            round(float(context.method_config.get("dip_sd_slow_worker_length_threshold", 1.75) or 1.75), 8),
            round(float(context.method_config.get("dip_sd_slow_worker_length_multiplier", 1.25) or 1.25), 8),
            round(float(model_config.verify_c), 18),
            round(float(model_config.verify_beta), 8),
            round(float(model_config.memory_cap_bytes), 1),
        )
        return (
            len(active_sessions),
            len(worker_ids),
            max_new,
            max_remaining,
            prefix_part,
            acceptance_part,
            worker_latency_part,
            config_part,
        )

    def _user_params(
        self,
        session: Any,
        worker_id: str | None,
        worker_metadata: dict[str, dict[str, Any]],
        history: Any,
        context: RuntimeContext,
    ) -> DiPSDUserParams:
        request_id = str(session.request_id)
        metadata = worker_metadata.get(str(worker_id), {}) if worker_id is not None else {}
        profile = speed_profile(metadata)
        prefix_len = len(getattr(session, "prefix_ids", []) or getattr(session, "prompt_ids", []) or [])
        default_acceptance = (
            profile.get("quality")
            if profile.get("quality") is not None
            else context.method_config.get("dip_sd_alpha", self.default_acceptance)
        )
        acceptance = _per_request_float(
            context.method_config.get("dip_sd_acceptance"),
            request_id,
            default=default_acceptance,
        )
        acceptance_feedback = _acceptance_feedback_for_request(
            history,
            request_id,
            prior_acceptance=float(acceptance),
            enabled=_config_bool(context.method_config.get("dip_sd_acceptance_feedback_enabled", True)),
            min_draft_tokens=int(context.method_config.get("dip_sd_acceptance_feedback_min_draft_tokens", 1) or 1),
            prior_weight=float(context.method_config.get("dip_sd_acceptance_feedback_prior_weight", 1.0) or 0.0),
        )
        if acceptance_feedback:
            acceptance = float(acceptance_feedback["effective_acceptance"])
        comm_latency_ms = _per_request_float(
            context.method_config.get("dip_sd_comm_latency_ms"),
            request_id,
            default=self.default_comm_latency_ms,
        )
        draft_beta = _per_request_float(
            context.method_config.get("dip_sd_draft_beta"),
            request_id,
            default=profile.get("draft_beta", self.default_draft_beta),
        )
        draft_c = _per_request_float(
            context.method_config.get("dip_sd_draft_c"),
            request_id,
            default=profile.get("draft_c", self.default_draft_c),
        )
        if profile.get("tokens_per_second") or profile.get("relative_speed"):
            ms_per_token = worker_ms_per_token(metadata, default_ms=max(1e-6, self.default_draft_beta))
            if _config_bool(context.method_config.get("dip_sd_use_measured_worker_latency", True)):
                draft_beta = ms_per_token
            else:
                draft_beta = min(float(draft_beta), ms_per_token)
        return DiPSDUserParams(
            request_id=request_id,
            prefix_len=int(prefix_len),
            acceptance=float(acceptance),
            comm_latency_ms=float(comm_latency_ms),
            draft_c=float(draft_c),
            draft_beta=float(draft_beta),
            remaining_tokens=getattr(session, "remaining_tokens", None),
            worker_id=worker_id,
            metadata={
                "worker_id": worker_id,
                "worker_profile": profile,
                "acceptance_feedback": acceptance_feedback,
            },
        )

    def _model_config(self, context: RuntimeContext) -> DiPSDModelConfig:
        base = self.model_config or DiPSDModelConfig()
        return DiPSDModelConfig(
            draft_blocks=int(context.method_config.get("dip_sd_draft_blocks", base.draft_blocks)),
            draft_hidden=int(context.method_config.get("dip_sd_draft_hidden", base.draft_hidden)),
            draft_ffn_hidden=int(context.method_config.get("dip_sd_draft_ffn_hidden", base.draft_ffn_hidden)),
            verify_blocks=int(context.method_config.get("dip_sd_verify_blocks", base.verify_blocks)),
            verify_hidden=int(context.method_config.get("dip_sd_verify_hidden", base.verify_hidden)),
            verify_ffn_hidden=int(context.method_config.get("dip_sd_verify_ffn_hidden", base.verify_ffn_hidden)),
            verify_c=float(context.method_config.get("dip_sd_verify_c", base.verify_c)),
            verify_beta=float(context.method_config.get("dip_sd_verify_beta", base.verify_beta)),
            memory_cap_bytes=float(context.method_config.get("dip_sd_memory_cap_bytes", base.memory_cap_bytes)),
        )


def paper_default_users(count: int | None = None) -> list[DiPSDUserParams]:
    defaults = DiPSDPaperDefaults()
    total = int(count or defaults.user_count)
    return [
        DiPSDUserParams(
            request_id=f"u{index + 1}",
            prefix_len=defaults.prefix_len,
            acceptance=defaults.acceptance,
            comm_latency_ms=defaults.comm_latency_ms,
            draft_c=defaults.draft_c,
            draft_beta=defaults.draft_beta,
        )
        for index in range(total)
    ]


def _rebind_shape_cached_hints(
    cached: PlanHints,
    *,
    active_sessions: list[Any],
    users: list[DiPSDUserParams],
    worker_preferences: dict[str, str],
    worker_assignment_order: list[str],
    model_config: DiPSDModelConfig,
    context: RuntimeContext,
    cache_key: tuple[Any, ...],
    offline_shape_key: str,
    requested_solver_mode: str,
    no_online_solver: bool,
    acceptance_feedback_enabled: bool,
    acceptance_feedback: dict[str, dict[str, Any]],
    speed_aware_assignment: bool,
) -> PlanHints | None:
    request_ids = [str(session.request_id) for session in active_sessions]
    source_order = _shape_cache_request_order(cached)
    draft_lengths = _rebind_draft_lengths_by_position(
        cached,
        source_order=source_order,
        active_sessions=active_sessions,
        context=context,
    )
    draft_lengths, adaptive_length_metadata = _maybe_adapt_draft_lengths(users, draft_lengths, context)
    preferred_batches = _rebind_batches_by_position(
        cached.preferred_batches,
        source_order=source_order,
        request_ids=request_ids,
    )
    preferred_batches, ready_rebatch_metadata = _maybe_ready_aware_rebatch(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
        context=context,
    )
    if not preferred_batches:
        return None
    evaluation = evaluate_schedule(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
    )
    if not evaluation.feasible:
        return None
    metadata = deepcopy(cached.metadata)
    preferred_batch_metadata = _preferred_batch_metadata_from_evaluation(evaluation, len(preferred_batches))
    trace = {
        "requested_solver_mode": requested_solver_mode,
        "solver_mode": "shape_cache",
        "backend_name": "shape_plan_cache",
        "backend_info": {
            "source": "shape_plan_cache",
            "shape_key": _shape_cache_key_label(cache_key),
            "offline_shape_key": offline_shape_key,
        },
        "paper_solver_complete": bool(metadata.get("paper_solver_complete", False)),
        "backend_fallback_used": False,
        "backend_fallback_reason": None,
        "batch_count": int(metadata.get("solver_planned_batch_count") or len(preferred_batches)),
        "batches": [list(batch) for batch in preferred_batches],
        "draft_lengths": dict(draft_lengths),
        "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
        "throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
        "expected_tokens": evaluation.expected_tokens,
        "pipeline_span_ms": evaluation.pipeline_span_ms,
        "verification_sum_ms": evaluation.verification_sum_ms,
        "max_draft_verify_ms": evaluation.max_draft_verify_ms,
        "shape_plan_cache_hit": True,
    }
    metadata.update(
        {
            "method_family": "dip_sd",
            "solver_active": False,
            "online_solver_enabled": not no_online_solver,
            "offline_plan_table_hit": False,
            "offline_plan_shape_key": offline_shape_key,
            "solver_mode": "shape_cache",
            "requested_solver_mode": requested_solver_mode,
            "solver_backend_name": "shape_plan_cache",
            "solver_backend_fallback_used": False,
            "solver_backend_fallback_reason": None,
            "solver_backend_info": trace["backend_info"],
            "planned_batch_count": len(preferred_batches),
            "estimated_throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
            "estimated_throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
            "estimated_expected_tokens": evaluation.expected_tokens,
            "estimated_pipeline_span_ms": evaluation.pipeline_span_ms,
            "estimated_verification_sum_ms": evaluation.verification_sum_ms,
            "estimated_max_draft_verify_ms": evaluation.max_draft_verify_ms,
            "dip_sd_model_config": asdict(model_config),
            "dip_sd_solution": trace,
            "preferred_batch_metadata": preferred_batch_metadata,
            **adaptive_length_metadata,
            **ready_rebatch_metadata,
            "worker_preferences": dict(worker_preferences),
            "speed_aware_worker_assignment": speed_aware_assignment,
            "worker_assignment_order": list(worker_assignment_order),
            "acceptance_feedback_enabled": acceptance_feedback_enabled,
            "acceptance_feedback_by_request": acceptance_feedback,
            "acceptance_feedback_applied_count": len(acceptance_feedback),
            "acceptance_cache_bucket": _acceptance_cache_bucket(context),
            "solver_preferred_batches": [list(batch) for batch in preferred_batches],
            "solver_cache_hit": True,
            "solver_cache_key": _shape_cache_key_label(cache_key),
            "solver_cache_shape_level": True,
            "shape_cache_key": _shape_cache_key_label(cache_key),
            "shape_cache_source_key": str(metadata.get("shape_cache_key") or metadata.get("solver_cache_key") or ""),
            "shape_cache_request_order": list(request_ids),
            "assignment_objective": "shape_plan_cache_replay",
        }
    )
    return PlanHints(
        draft_lengths=draft_lengths,
        worker_preferences=dict(worker_preferences),
        preferred_batches=[list(batch) for batch in preferred_batches],
        metadata=metadata,
    )


def _heuristic_no_online_plan_hints(
    *,
    active_sessions: list[Any],
    users: list[DiPSDUserParams],
    worker_preferences: dict[str, str],
    worker_assignment_order: list[str],
    model_config: DiPSDModelConfig,
    context: RuntimeContext,
    shape_key: str,
    requested_solver_mode: str,
    acceptance_feedback_enabled: bool,
    acceptance_feedback: dict[str, dict[str, Any]],
    speed_aware_assignment: bool,
    source: str,
) -> PlanHints:
    request_ids = [str(session.request_id) for session in active_sessions]
    draft_lengths = _heuristic_draft_lengths(active_sessions, context)
    draft_lengths, adaptive_length_metadata = _maybe_adapt_draft_lengths(users, draft_lengths, context)
    preferred_batches = _heuristic_preferred_batches(request_ids, context)
    preferred_batches, ready_rebatch_metadata = _maybe_ready_aware_rebatch(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
        context=context,
    )
    evaluation = evaluate_schedule(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
    )
    if not evaluation.feasible:
        raise ValueError(f"DiP-SD no-online heuristic plan is infeasible: {evaluation.reason}")
    preferred_batch_metadata = _preferred_batch_metadata_from_evaluation(evaluation, len(preferred_batches))
    solver_planned_batch_count = len(preferred_batches)
    trace = {
        "requested_solver_mode": requested_solver_mode,
        "solver_mode": "offline_heuristic",
        "backend_name": "no_online_heuristic",
        "backend_info": {"source": source, "shape_key": shape_key},
        "paper_solver_complete": False,
        "backend_fallback_used": False,
        "backend_fallback_reason": None,
        "batch_count": solver_planned_batch_count,
        "batches": [list(batch) for batch in preferred_batches],
        "draft_lengths": dict(draft_lengths),
        "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
        "throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
        "expected_tokens": evaluation.expected_tokens,
        "pipeline_span_ms": evaluation.pipeline_span_ms,
        "verification_sum_ms": evaluation.verification_sum_ms,
        "max_draft_verify_ms": evaluation.max_draft_verify_ms,
        "offline_plan_table_hit": False,
    }
    return PlanHints(
        draft_lengths=dict(draft_lengths),
        worker_preferences=dict(worker_preferences),
        preferred_batches=[list(batch) for batch in preferred_batches],
        metadata={
            "method_family": "dip_sd",
            "solver_active": False,
            "online_solver_enabled": False,
            "offline_plan_table_hit": False,
            "offline_plan_key": None,
            "offline_plan_shape_key": shape_key,
            "offline_plan_source": source,
            "solver_mode": "offline_heuristic",
            "requested_solver_mode": requested_solver_mode,
            "solver_backend_name": "no_online_heuristic",
            "paper_solver_complete": False,
            "solver_backend_fallback_used": False,
            "solver_backend_fallback_reason": None,
            "solver_backend_info": {"source": source, "shape_key": shape_key},
            "joint_batch_assignment": True,
            "joint_draft_length": True,
            "phase_level_pipeline_required": True,
            "distributed_local_drafting": True,
            "central_batch_verification": True,
            "planned_batch_count": len(preferred_batches),
            "solver_planned_batch_count": solver_planned_batch_count,
            "hybrid_single_batch_threshold": context.method_config.get("dip_sd_single_batch_small_request_threshold"),
            "hybrid_single_batch_applied": len(preferred_batches) == 1 and len(request_ids) > 1,
            "hybrid_single_batch_reason": "no_online_heuristic_single_batch" if len(preferred_batches) == 1 and len(request_ids) > 1 else None,
            "estimated_throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
            "estimated_throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
            "estimated_expected_tokens": evaluation.expected_tokens,
            "estimated_pipeline_span_ms": evaluation.pipeline_span_ms,
            "estimated_verification_sum_ms": evaluation.verification_sum_ms,
            "estimated_max_draft_verify_ms": evaluation.max_draft_verify_ms,
            "dip_sd_model_config": asdict(model_config),
            "latency_calibration_profile": context.method_config.get("dip_sd_calibration_profile"),
            "latency_calibration_enabled": context.method_config.get("dip_sd_calibration_enabled"),
            "latency_calibration_applied": context.method_config.get("dip_sd_calibration_applied"),
            "latency_calibration_overrides": dict(context.method_config.get("dip_sd_calibration_overrides") or {}),
            "dip_sd_solution": trace,
            "preferred_batch_metadata": preferred_batch_metadata,
            **adaptive_length_metadata,
            **ready_rebatch_metadata,
            "worker_preferences": dict(worker_preferences),
            "speed_aware_worker_assignment": speed_aware_assignment,
            "worker_assignment_order": list(worker_assignment_order),
            "acceptance_feedback_enabled": acceptance_feedback_enabled,
            "acceptance_feedback_by_request": acceptance_feedback,
            "acceptance_feedback_applied_count": len(acceptance_feedback),
            "acceptance_cache_bucket": _acceptance_cache_bucket(context),
            "solver_preferred_batches": [list(batch) for batch in preferred_batches],
            "solver_cache_hit": False,
            "solver_cache_key": shape_key,
            "solver_cache_shape_level": False,
            "assignment_objective": "no_online_solver_heuristic",
        },
    )


def _offline_plan_hints(
    *,
    active_sessions: list[Any],
    users: list[DiPSDUserParams],
    worker_preferences: dict[str, str],
    worker_assignment_order: list[str],
    model_config: DiPSDModelConfig,
    context: RuntimeContext,
    entry: dict[str, Any],
    shape_key: str,
    acceptance_feedback_enabled: bool,
    acceptance_feedback: dict[str, dict[str, Any]],
    speed_aware_assignment: bool,
) -> PlanHints:
    request_ids = [str(session.request_id) for session in active_sessions]
    draft_lengths = _offline_draft_lengths(entry, active_sessions, context)
    draft_lengths, adaptive_length_metadata = _maybe_adapt_draft_lengths(users, draft_lengths, context)
    preferred_batches = _offline_preferred_batches(entry, request_ids)
    preferred_batches, ready_rebatch_metadata = _maybe_ready_aware_rebatch(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
        context=context,
    )
    evaluation = evaluate_schedule(
        users=users,
        batches=preferred_batches,
        draft_lengths=draft_lengths,
        config=model_config,
    )
    if not evaluation.feasible:
        raise ValueError(f"DiP-SD offline plan table entry is infeasible: {evaluation.reason}")
    preferred_batch_metadata = [
        {
            "stage_index": metric.stage_index,
            "planned_batch_count": len(preferred_batches),
            "max_draft_len": metric.max_draft_len,
            "max_prefix_len": metric.max_prefix_len,
            "estimated_verify_ms": metric.verify_ms,
            "estimated_memory_bytes": metric.memory_bytes,
            "estimated_draft_complete_ms": metric.draft_complete_ms,
            "estimated_stage_duration_ms": metric.stage_duration_ms,
            "request_ids": list(metric.request_ids),
        }
        for metric in evaluation.batch_metrics
    ]
    solver_planned_batch_count = int(entry.get("solver_planned_batch_count") or len(preferred_batches))
    requested_solver_mode = str(context.method_config.get("dip_sd_solver", "offline_table"))
    source = str(entry.get("source") or "offline_plan_table")
    trace = {
        "requested_solver_mode": requested_solver_mode,
        "solver_mode": "offline_table",
        "backend_name": "offline_plan_table",
        "backend_info": {"source": source, "shape_key": shape_key},
        "paper_solver_complete": bool(entry.get("paper_solver_complete", False)),
        "backend_fallback_used": False,
        "backend_fallback_reason": None,
        "batch_count": solver_planned_batch_count,
        "batches": [list(batch) for batch in preferred_batches],
        "draft_lengths": dict(draft_lengths),
        "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
        "throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
        "expected_tokens": evaluation.expected_tokens,
        "pipeline_span_ms": evaluation.pipeline_span_ms,
        "verification_sum_ms": evaluation.verification_sum_ms,
        "max_draft_verify_ms": evaluation.max_draft_verify_ms,
        "offline_plan_table_hit": True,
    }
    return PlanHints(
        draft_lengths=dict(draft_lengths),
        worker_preferences=dict(worker_preferences),
        preferred_batches=[list(batch) for batch in preferred_batches],
        metadata={
            "method_family": "dip_sd",
            "solver_active": False,
            "online_solver_enabled": False,
            "offline_plan_table_hit": True,
            "offline_plan_key": str(entry.get("key") or shape_key),
            "offline_plan_shape_key": shape_key,
            "offline_plan_source": source,
            "solver_mode": "offline_table",
            "requested_solver_mode": requested_solver_mode,
            "solver_backend_name": "offline_plan_table",
            "paper_solver_complete": bool(entry.get("paper_solver_complete", False)),
            "solver_backend_fallback_used": False,
            "solver_backend_fallback_reason": None,
            "solver_backend_info": {"source": source, "shape_key": shape_key},
            "joint_batch_assignment": True,
            "joint_draft_length": True,
            "phase_level_pipeline_required": True,
            "distributed_local_drafting": True,
            "central_batch_verification": True,
            "planned_batch_count": len(preferred_batches),
            "solver_planned_batch_count": solver_planned_batch_count,
            "hybrid_single_batch_threshold": context.method_config.get("dip_sd_single_batch_small_request_threshold"),
            "hybrid_single_batch_applied": bool(entry.get("hybrid_single_batch_applied", False)),
            "hybrid_single_batch_reason": entry.get("hybrid_single_batch_reason"),
            "estimated_throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
            "estimated_throughput_tokens_per_s": evaluation.throughput_tokens_per_ms * 1000.0,
            "estimated_expected_tokens": evaluation.expected_tokens,
            "estimated_pipeline_span_ms": evaluation.pipeline_span_ms,
            "estimated_verification_sum_ms": evaluation.verification_sum_ms,
            "estimated_max_draft_verify_ms": evaluation.max_draft_verify_ms,
            "dip_sd_model_config": asdict(model_config),
            "latency_calibration_profile": context.method_config.get("dip_sd_calibration_profile"),
            "latency_calibration_enabled": context.method_config.get("dip_sd_calibration_enabled"),
            "latency_calibration_applied": context.method_config.get("dip_sd_calibration_applied"),
            "latency_calibration_overrides": dict(context.method_config.get("dip_sd_calibration_overrides") or {}),
            "dip_sd_solution": trace,
            "preferred_batch_metadata": preferred_batch_metadata,
            **adaptive_length_metadata,
            **ready_rebatch_metadata,
            "worker_preferences": dict(worker_preferences),
            "speed_aware_worker_assignment": speed_aware_assignment,
            "worker_assignment_order": list(worker_assignment_order),
            "acceptance_feedback_enabled": acceptance_feedback_enabled,
            "acceptance_feedback_by_request": acceptance_feedback,
            "acceptance_feedback_applied_count": len(acceptance_feedback),
            "acceptance_cache_bucket": _acceptance_cache_bucket(context),
            "solver_preferred_batches": [list(batch) for batch in preferred_batches],
            "solver_cache_hit": True,
            "solver_cache_key": shape_key,
            "solver_cache_shape_level": True,
            "assignment_objective": "offline_plan_table_no_online_solver",
        },
    )


def _offline_plan_table_entry(
    active_sessions: list[Any],
    worker_ids: list[str],
    context: RuntimeContext,
) -> tuple[str, dict[str, Any] | None]:
    shape_key = _offline_plan_shape_key(active_sessions, worker_ids, context)
    raw = context.method_config.get("dip_sd_offline_plan_table")
    if not isinstance(raw, dict) or not raw:
        return shape_key, None
    table = raw.get("entries") if isinstance(raw.get("entries"), dict) else raw
    entry = None
    if isinstance(table, dict):
        entry = table.get(shape_key) or table.get("default")
        if entry is None and _looks_like_offline_entry(table):
            entry = table
    if entry is None:
        return shape_key, None
    if not isinstance(entry, dict):
        raise ValueError("DiP-SD offline plan table entry must be a mapping.")
    return shape_key, dict(entry)


def _offline_plan_shape_key(active_sessions: list[Any], worker_ids: list[str], context: RuntimeContext) -> str:
    max_new = max((int(getattr(session, "max_new_tokens", 0) or 0) for session in active_sessions), default=0)
    max_remaining = max((int(getattr(session, "remaining_tokens", 0) or 0) for session in active_sessions), default=0)
    max_prefix = max(
        (
            len(getattr(session, "prefix_ids", []) or getattr(session, "prompt_ids", []) or [])
            for session in active_sessions
        ),
        default=0,
    )
    bucket = int(context.method_config.get("dip_sd_offline_plan_prefix_bucket", 32) or 0)
    prefix_part = _ceil_bucket(max_prefix, bucket) if bucket > 0 else max_prefix
    return (
        f"requests={len(active_sessions)}|workers={len(worker_ids)}|"
        f"max_new={max_new}|remaining={max_remaining}|prefix<={prefix_part}"
    )


def _offline_draft_lengths(
    entry: dict[str, Any],
    active_sessions: list[Any],
    context: RuntimeContext,
) -> dict[str, int]:
    raw = entry.get("draft_lengths", entry.get("draft_length"))
    default_length = int(context.method_config.get("dip_sd_initial_draft_length", 1) or 1)
    max_length = int(context.method_config.get("dip_sd_max_draft_length", default_length) or default_length)
    lengths: dict[str, int] = {}
    for index, session in enumerate(active_sessions):
        request_id = str(session.request_id)
        value = _offline_value_for_request(raw, request_id, index, default_length)
        remaining = int(getattr(session, "remaining_tokens", value) or value)
        lengths[request_id] = max(1, min(int(value), max_length, max(1, remaining)))
    return lengths


def _offline_value_for_request(raw: Any, request_id: str, index: int, default: int) -> int:
    if isinstance(raw, dict):
        if request_id in raw:
            return int(raw[request_id])
        if str(index) in raw:
            return int(raw[str(index)])
        if "default" in raw:
            return int(raw["default"])
    if isinstance(raw, list) and raw:
        return int(raw[min(index, len(raw) - 1)])
    if raw is not None and not isinstance(raw, (dict, list)):
        return int(raw)
    return int(default)


def _offline_preferred_batches(entry: dict[str, Any], request_ids: list[str]) -> list[list[str]]:
    raw = entry.get("preferred_batches")
    if (isinstance(raw, str) and raw in {"single", "single_batch", "all"}) or entry.get("single_batch"):
        return [list(request_ids)]
    if raw is None:
        batch_size = int(entry.get("batch_size") or len(request_ids) or 1)
        return [
            request_ids[index : index + batch_size]
            for index in range(0, len(request_ids), max(1, batch_size))
        ]
    if not isinstance(raw, list):
        raise ValueError("DiP-SD offline preferred_batches must be a list or 'single_batch'.")
    batches: list[list[str]] = []
    seen: set[str] = set()
    for batch in raw:
        if not isinstance(batch, list):
            raise ValueError("DiP-SD offline preferred_batches entries must be lists.")
        converted: list[str] = []
        for item in batch:
            request_id = _offline_request_id(item, request_ids)
            if request_id in seen:
                raise ValueError(f"DiP-SD offline plan duplicates request_id={request_id!r}.")
            seen.add(request_id)
            converted.append(request_id)
        if converted:
            batches.append(converted)
    missing = [request_id for request_id in request_ids if request_id not in seen]
    if missing:
        raise ValueError(f"DiP-SD offline plan missing request ids: {missing}")
    return batches


def _offline_request_id(value: Any, request_ids: list[str]) -> str:
    if isinstance(value, int):
        if value < 0 or value >= len(request_ids):
            raise ValueError(f"DiP-SD offline request index out of range: {value}")
        return request_ids[value]
    text = str(value)
    if text in request_ids:
        return text
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(request_ids):
            return request_ids[index]
    raise ValueError(f"DiP-SD offline request id is unknown: {value!r}")


def _looks_like_offline_entry(value: dict[str, Any]) -> bool:
    return any(key in value for key in {"draft_lengths", "draft_length", "preferred_batches", "single_batch", "batch_size"})


def _ceil_bucket(value: int, bucket: int) -> int:
    if bucket <= 0:
        return int(value)
    return ((int(value) + bucket - 1) // bucket) * bucket


def _shape_cache_request_order(hints: PlanHints) -> list[str]:
    raw = dict(getattr(hints, "metadata", {}) or {}).get("shape_cache_request_order")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return [str(request_id) for request_id in hints.draft_lengths]


def _rebind_draft_lengths_by_position(
    hints: PlanHints,
    *,
    source_order: list[str],
    active_sessions: list[Any],
    context: RuntimeContext,
) -> dict[str, int]:
    default_length = int(context.method_config.get("dip_sd_initial_draft_length", 1) or 1)
    max_length = int(context.method_config.get("dip_sd_max_draft_length", default_length) or default_length)
    source_lengths = [
        int(hints.draft_lengths.get(request_id, default_length))
        for request_id in source_order
    ]
    lengths: dict[str, int] = {}
    for index, session in enumerate(active_sessions):
        request_id = str(session.request_id)
        source_length = source_lengths[min(index, len(source_lengths) - 1)] if source_lengths else default_length
        remaining = int(getattr(session, "remaining_tokens", source_length) or source_length)
        lengths[request_id] = max(1, min(int(source_length), max_length, max(1, remaining)))
    return lengths


def _rebind_batches_by_position(
    raw_batches: list[list[str]],
    *,
    source_order: list[str],
    request_ids: list[str],
) -> list[list[str]]:
    source_index = {request_id: index for index, request_id in enumerate(source_order)}
    seen: set[str] = set()
    batches: list[list[str]] = []
    for raw_batch in raw_batches:
        batch: list[str] = []
        for item in raw_batch:
            index = None
            if isinstance(item, int):
                index = item
            else:
                text = str(item)
                if text in source_index:
                    index = source_index[text]
                elif text.isdigit():
                    index = int(text)
            if index is None or index < 0 or index >= len(request_ids):
                continue
            request_id = request_ids[index]
            if request_id not in seen:
                seen.add(request_id)
                batch.append(request_id)
        if batch:
            batches.append(batch)
    missing = [request_id for request_id in request_ids if request_id not in seen]
    if missing:
        if batches:
            batches[-1].extend(missing)
        else:
            batches.append(missing)
    return batches


def _preferred_batch_metadata_from_evaluation(
    evaluation: DiPSDScheduleEvaluation,
    planned_batch_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "stage_index": metric.stage_index,
            "planned_batch_count": int(planned_batch_count),
            "max_draft_len": metric.max_draft_len,
            "max_prefix_len": metric.max_prefix_len,
            "estimated_verify_ms": metric.verify_ms,
            "estimated_memory_bytes": metric.memory_bytes,
            "estimated_draft_complete_ms": metric.draft_complete_ms,
            "estimated_stage_duration_ms": metric.stage_duration_ms,
            "request_ids": list(metric.request_ids),
        }
        for metric in evaluation.batch_metrics
    ]


def _maybe_adapt_draft_lengths(
    users: list[DiPSDUserParams],
    draft_lengths: dict[str, int],
    context: RuntimeContext,
) -> tuple[dict[str, int], dict[str, Any]]:
    enabled = _config_bool(context.method_config.get("dip_sd_adaptive_draft_length_enabled", True))
    metadata: dict[str, Any] = {
        "adaptive_draft_length_enabled": enabled,
        "adaptive_draft_length_applied": False,
        "adaptive_draft_length_changes": [],
    }
    lengths = {str(request_id): max(1, int(length)) for request_id, length in draft_lengths.items()}
    if not enabled or not users:
        return lengths, metadata

    min_length = max(1, int(context.method_config.get("dip_sd_min_draft_length", 1) or 1))
    max_length = max(min_length, int(context.method_config.get("dip_sd_max_draft_length", 20) or 20))
    target_acceptance = max(
        1e-6,
        float(context.method_config.get("dip_sd_adaptive_length_target_acceptance", 0.78) or 0.78),
    )
    min_acceptance_factor = max(
        0.05,
        min(float(context.method_config.get("dip_sd_adaptive_length_min_factor", 0.35) or 0.35), 1.0),
    )
    slow_threshold = max(
        1.0,
        float(context.method_config.get("dip_sd_slow_worker_length_threshold", 1.75) or 1.75),
    )
    slow_multiplier = max(
        0.25,
        float(context.method_config.get("dip_sd_slow_worker_length_multiplier", 1.25) or 1.25),
    )
    fastest_beta = min((max(1e-6, float(user.draft_beta)) for user in users), default=1.0)

    changes: list[dict[str, Any]] = []
    for user in users:
        request_id = str(user.request_id)
        base = max(min_length, min(int(lengths.get(request_id, min_length)), max_length))
        remaining = int(user.remaining_tokens) if user.remaining_tokens is not None else base
        target = max(min_length, min(base, max(1, remaining)))
        reasons: list[str] = []

        feedback = dict(user.metadata.get("acceptance_feedback") or {})
        if feedback:
            effective = max(0.0, min(float(feedback.get("effective_acceptance", user.acceptance)), 1.0))
            factor = max(min_acceptance_factor, min(effective / target_acceptance, 1.0))
            acceptance_cap = max(min_length, int(ceil(base * factor)))
            if acceptance_cap < target:
                target = acceptance_cap
                reasons.append("acceptance_feedback")

        worker_beta = max(1e-6, float(user.draft_beta))
        if worker_beta > fastest_beta * slow_threshold:
            latency_cap = max(min_length, int(ceil(base * fastest_beta / worker_beta * slow_multiplier)))
            if latency_cap < target:
                target = latency_cap
                reasons.append("slow_worker_latency")

        target = max(min_length, min(target, max_length, max(1, remaining)))
        lengths[request_id] = target
        if target != base:
            changes.append(
                {
                    "request_id": request_id,
                    "from": int(base),
                    "to": int(target),
                    "worker_id": user.worker_id,
                    "draft_beta": float(user.draft_beta),
                    "acceptance": float(user.acceptance),
                    "reasons": reasons,
                }
            )

    metadata["adaptive_draft_length_applied"] = bool(changes)
    metadata["adaptive_draft_length_changes"] = changes
    metadata["adaptive_draft_length_min_tokens"] = int(min_length)
    metadata["adaptive_draft_length_target_acceptance"] = float(target_acceptance)
    metadata["adaptive_draft_length_min_factor"] = float(min_acceptance_factor)
    metadata["adaptive_draft_length_fastest_beta"] = float(fastest_beta)
    return lengths, metadata


def _maybe_ready_aware_rebatch(
    *,
    users: list[DiPSDUserParams],
    batches: list[list[str]],
    draft_lengths: dict[str, int],
    config: DiPSDModelConfig,
    context: RuntimeContext,
) -> tuple[list[list[str]], dict[str, Any]]:
    enabled = _config_bool(context.method_config.get("dip_sd_ready_aware_rebatch_enabled", True))
    min_spread_ms = float(context.method_config.get("dip_sd_ready_aware_rebatch_min_spread_ms", 5.0) or 0.0)
    metadata: dict[str, Any] = {
        "ready_aware_rebatch_enabled": enabled,
        "ready_aware_rebatch_applied": False,
        "ready_aware_rebatch_min_spread_ms": min_spread_ms,
    }
    if not enabled or len(batches) <= 1:
        return [list(batch) for batch in batches], metadata

    request_ids = [str(request_id) for batch in batches for request_id in batch]
    if len(request_ids) != len(set(request_ids)):
        return [list(batch) for batch in batches], metadata
    users_by_id = {str(user.request_id): user for user in users}
    if any(request_id not in users_by_id for request_id in request_ids):
        return [list(batch) for batch in batches], metadata

    ready_ms = {
        request_id: draft_latency_ms(
            users_by_id[request_id],
            int(draft_lengths.get(request_id, 1)),
            config,
        )
        + max(0.0, float(users_by_id[request_id].comm_latency_ms))
        for request_id in request_ids
    }
    ready_values = [float(value) for value in ready_ms.values()]
    ready_spread_ms = max(ready_values) - min(ready_values) if ready_values else 0.0
    metadata.update(
        {
            "ready_aware_ready_time_ms": {request_id: round(float(value), 6) for request_id, value in ready_ms.items()},
            "ready_aware_rebatch_spread_ms": round(float(ready_spread_ms), 6),
        }
    )
    if ready_spread_ms < min_spread_ms:
        metadata["ready_aware_rebatch_reason"] = "ready_spread_below_threshold"
        return [list(batch) for batch in batches], metadata

    ordered = sorted(request_ids, key=lambda request_id: (ready_ms.get(request_id, 0.0), request_id))
    batch_sizes = [len(batch) for batch in batches if batch]
    rebatches: list[list[str]] = []
    cursor = 0
    for size in batch_sizes:
        rebatches.append(ordered[cursor : cursor + size])
        cursor += size
    if cursor < len(ordered):
        if rebatches:
            rebatches[-1].extend(ordered[cursor:])
        else:
            rebatches.append(ordered[cursor:])

    original = [list(batch) for batch in batches]
    applied = rebatches != original
    metadata.update(
        {
            "ready_aware_rebatch_applied": applied,
        }
    )
    if applied:
        metadata["ready_aware_original_batches"] = original
        metadata["ready_aware_rebatched_batches"] = [list(batch) for batch in rebatches]
    return rebatches, metadata


def _heuristic_draft_lengths(active_sessions: list[Any], context: RuntimeContext) -> dict[str, int]:
    default_length = int(context.method_config.get("dip_sd_initial_draft_length", 1) or 1)
    max_length = int(context.method_config.get("dip_sd_max_draft_length", default_length) or default_length)
    lengths: dict[str, int] = {}
    for session in active_sessions:
        request_id = str(session.request_id)
        remaining = int(getattr(session, "remaining_tokens", default_length) or default_length)
        lengths[request_id] = max(1, min(default_length, max_length, max(1, remaining)))
    return lengths


def _heuristic_preferred_batches(request_ids: list[str], context: RuntimeContext) -> list[list[str]]:
    if not request_ids:
        return []
    threshold = int(context.method_config.get("dip_sd_single_batch_small_request_threshold", 0) or 0)
    if 1 < len(request_ids) <= threshold:
        return [list(request_ids)]
    min_batch_count = int(context.method_config.get("dip_sd_min_batch_count", 2) or 2)
    max_batch_count = _optional_int(context.method_config.get("dip_sd_max_batch_count"), None)
    batch_count = 1 if len(request_ids) == 1 else max(1, min(len(request_ids), min_batch_count))
    if max_batch_count is not None:
        batch_count = min(batch_count, max(1, max_batch_count))
    return _balanced_batches(request_ids, batch_count)


def _balanced_batches(request_ids: list[str], batch_count: int) -> list[list[str]]:
    count = max(1, min(len(request_ids), int(batch_count)))
    base = len(request_ids) // count
    extra = len(request_ids) % count
    batches: list[list[str]] = []
    cursor = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        batch = request_ids[cursor : cursor + size]
        cursor += size
        if batch:
            batches.append(batch)
    return batches


def _assign_user_devices(
    active_sessions: list[Any],
    worker_ids: list[str],
    *,
    history: Any,
    context: RuntimeContext,
) -> dict[str, str]:
    if not worker_ids:
        return {}
    strategy = str(context.method_config.get("dip_sd_worker_assignment_strategy", "latency_first") or "latency_first")
    if strategy in {"round_robin", "stable", "request_order"}:
        ordered_sessions = list(active_sessions)
    else:
        ordered_sessions = sorted(
            active_sessions,
            key=lambda session: (
                -int(getattr(session, "remaining_tokens", 0) or 0),
                _observed_acceptance_sort_key(history, str(session.request_id)),
                str(session.request_id),
            ),
        )
    sticky_enabled = _config_bool(context.method_config.get("dip_sd_prefetch_sticky_worker_enabled", True))
    prefetch_by_request = _history_mapping(history, "dip_sd_prefetch_by_request")
    assignments: dict[str, str] = {}
    sticky_workers: set[str] = set()
    if sticky_enabled and prefetch_by_request:
        worker_set = {str(worker_id) for worker_id in worker_ids}
        for session in ordered_sessions:
            request_id = str(session.request_id)
            worker_id = _prefetched_worker_id(prefetch_by_request.get(request_id))
            if worker_id in worker_set:
                assignments[request_id] = worker_id
                sticky_workers.add(worker_id)
    fallback_workers = [worker_id for worker_id in worker_ids if worker_id not in sticky_workers] or list(worker_ids)
    fallback_index = 0
    for session in ordered_sessions:
        request_id = str(session.request_id)
        if request_id in assignments:
            continue
        assignments[request_id] = fallback_workers[fallback_index % len(fallback_workers)]
        fallback_index += 1
    return assignments


def _prefetched_worker_id(raw: Any) -> str | None:
    if isinstance(raw, dict):
        worker_id = raw.get("worker_id")
    else:
        worker_id = raw
    if worker_id is None:
        return None
    return str(worker_id)


def _observed_acceptance_sort_key(history: Any, request_id: str) -> float:
    raw = _history_mapping(history, "dip_sd_acceptance_stats").get(str(request_id), {})
    if not isinstance(raw, dict):
        return 1.0
    if raw.get("observed_acceptance") is not None:
        return max(0.0, min(float(raw["observed_acceptance"]), 1.0))
    draft_token_count = int(raw.get("draft_token_count") or 0)
    accepted_draft_count = int(raw.get("accepted_draft_count") or 0)
    if draft_token_count <= 0:
        return 1.0
    return max(0.0, min(accepted_draft_count / max(1, draft_token_count), 1.0))


def _worker_assignment_order(
    worker_ids: list[str],
    worker_metadata: dict[str, dict[str, Any]],
    *,
    speed_aware: bool,
    default_ms_per_token: float,
) -> list[str]:
    ordered_workers = [str(worker_id) for worker_id in worker_ids]
    if not speed_aware:
        return ordered_workers
    original_index = {worker_id: index for index, worker_id in enumerate(ordered_workers)}
    return sorted(
        ordered_workers,
        key=lambda worker_id: (
            worker_ms_per_token(
                worker_metadata.get(worker_id, {}),
                default_ms=max(1e-6, float(default_ms_per_token)),
            ),
            worker_latency_ms(worker_metadata.get(worker_id, {})),
            original_index[worker_id],
        ),
    )


def _clone_hints(hints: PlanHints) -> PlanHints:
    return PlanHints(
        draft_lengths=dict(hints.draft_lengths),
        candidate_draft_lengths=deepcopy(hints.candidate_draft_lengths),
        worker_preferences=dict(hints.worker_preferences),
        candidate_worker_preferences=deepcopy(hints.candidate_worker_preferences),
        preferred_batches=[list(batch) for batch in hints.preferred_batches],
        metadata=deepcopy(hints.metadata),
    )


def _cache_key_label(cache_key: tuple[Any, ...]) -> str:
    request_part = cache_key[0] if cache_key else ()
    worker_part = cache_key[1] if len(cache_key) > 1 else ()
    requests = ",".join(str(item[0]) for item in request_part)
    workers = ",".join(str(worker_id) for worker_id in worker_part)
    return f"requests=[{requests}]|workers=[{workers}]"


def _shape_cache_key_label(cache_key: tuple[Any, ...]) -> str:
    if len(cache_key) < 5:
        return "shape=unknown"
    return (
        f"requests={cache_key[0]}|workers={cache_key[1]}|"
        f"max_new={cache_key[2]}|remaining={cache_key[3]}|prefix<={cache_key[4]}"
    )


def _no_online_solver_fallback_mode(context: RuntimeContext, requested_solver_mode: str) -> str:
    if requested_solver_mode in {"offline_heuristic", "no_online_heuristic"}:
        return "offline_heuristic"
    raw = context.method_config.get("dip_sd_no_online_solver_fallback", "error")
    mode = str(raw or "error").strip().lower()
    aliases = {
        "strict": "error",
        "fail": "error",
        "raise": "error",
        "none": "error",
        "heuristic": "offline_heuristic",
        "offline_heuristic": "offline_heuristic",
        "no_online_heuristic": "offline_heuristic",
    }
    return aliases.get(mode, mode)


def _per_request_float(raw: Any, request_id: str, *, default: Any) -> float:
    if isinstance(raw, dict):
        return float(raw.get(request_id, default))
    if isinstance(raw, list):
        try:
            index = int(str(request_id).rsplit("-", 1)[-1])
        except ValueError:
            index = 0
        if raw:
            return float(raw[index % len(raw)])
    if raw is not None and not isinstance(raw, (dict, list)):
        return float(raw)
    return float(default)


def _acceptance_feedback_for_request(
    history: Any,
    request_id: str,
    *,
    prior_acceptance: float,
    enabled: bool,
    min_draft_tokens: int,
    prior_weight: float,
) -> dict[str, Any]:
    if not enabled:
        return {}
    all_stats = _history_mapping(history, "dip_sd_acceptance_stats")
    raw = all_stats.get(str(request_id))
    if not isinstance(raw, dict):
        return {}
    draft_token_count = int(raw.get("draft_token_count") or 0)
    accepted_draft_count = int(raw.get("accepted_draft_count") or 0)
    if draft_token_count < max(1, int(min_draft_tokens)):
        return {}
    observed = accepted_draft_count / max(1, draft_token_count)
    prior = max(0.0, min(float(prior_acceptance), 1.0))
    weight = max(0.0, float(prior_weight))
    effective = (accepted_draft_count + weight * prior) / max(1e-12, draft_token_count + weight)
    effective = max(0.0, min(float(effective), 1.0))
    return {
        "effective_acceptance": effective,
        "observed_acceptance": max(0.0, min(float(observed), 1.0)),
        "prior_acceptance": prior,
        "prior_weight": weight,
        "draft_token_count": draft_token_count,
        "accepted_draft_count": accepted_draft_count,
        "proposal_count": int(raw.get("proposal_count") or 0),
        "bonus_count": int(raw.get("bonus_count") or 0),
        "output_token_count": int(raw.get("output_token_count") or 0),
        "last_observed_acceptance": raw.get("last_observed_acceptance"),
    }


def _history_mapping(history: Any, key: str) -> dict[str, Any]:
    if isinstance(history, dict):
        raw = history.get(key, {})
    else:
        raw = getattr(history, key, {})
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _acceptance_cache_bucket(context: RuntimeContext) -> float:
    bucket = float(context.method_config.get("dip_sd_acceptance_cache_bucket", 0.25) or 0.0)
    return max(0.0, min(bucket, 1.0))


def _acceptance_cache_value(value: float, context: RuntimeContext) -> float:
    bounded = max(0.0, min(float(value), 1.0))
    bucket = _acceptance_cache_bucket(context)
    if bucket <= 0.0:
        return round(bounded, 8)
    return round(max(0.0, min(round(bounded / bucket) * bucket, 1.0)), 8)


def _config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _optional_int(raw: Any, default: int | None) -> int | None:
    if raw is None:
        return default
    value = int(raw)
    return value if value > 0 else None
