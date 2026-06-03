from __future__ import annotations

"""DiP-SD batch-count, association and draft-length solver."""

import importlib
import importlib.util
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from specplatform.methods.dip_sd.model import (
    DiPSDModelConfig,
    DiPSDScheduleEvaluation,
    DiPSDUserParams,
    draft_latency_ms,
    evaluate_schedule,
    expected_accepted_tokens,
)


@dataclass(frozen=True)
class DiPSDSolution:
    batch_count: int
    batches: tuple[tuple[str, ...], ...]
    draft_lengths: dict[str, int]
    evaluation: DiPSDScheduleEvaluation
    trace: list[dict[str, Any]] = field(default_factory=list)
    solver_mode: str = "enumerate"
    requested_solver_mode: str = "enumerate"
    backend_name: str = "enumerate"
    paper_solver_complete: bool = False
    backend_fallback_used: bool = False
    backend_fallback_reason: str | None = None
    backend_info: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_tokens_per_ms(self) -> float:
        return self.evaluation.throughput_tokens_per_ms

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "solver_mode": self.solver_mode,
            "requested_solver_mode": self.requested_solver_mode,
            "backend_name": self.backend_name,
            "paper_solver_complete": self.paper_solver_complete,
            "backend_fallback_used": self.backend_fallback_used,
            "backend_fallback_reason": self.backend_fallback_reason,
            "backend_info": dict(self.backend_info),
            "batch_count": self.batch_count,
            "batches": [list(batch) for batch in self.batches],
            "draft_lengths": dict(self.draft_lengths),
            "throughput_tokens_per_ms": self.evaluation.throughput_tokens_per_ms,
            "throughput_tokens_per_s": self.evaluation.throughput_tokens_per_ms * 1000.0,
            "expected_tokens": self.evaluation.expected_tokens,
            "pipeline_span_ms": self.evaluation.pipeline_span_ms,
            "verification_sum_ms": self.evaluation.verification_sum_ms,
            "max_draft_verify_ms": self.evaluation.max_draft_verify_ms,
            "batches_metrics": [
                {
                    "stage_index": metric.stage_index,
                    "request_ids": list(metric.request_ids),
                    "batch_size": metric.batch_size,
                    "max_draft_len": metric.max_draft_len,
                    "max_prefix_len": metric.max_prefix_len,
                    "draft_complete_ms": metric.draft_complete_ms,
                    "verify_ms": metric.verify_ms,
                    "stage_duration_ms": metric.stage_duration_ms,
                    "memory_bytes": metric.memory_bytes,
                    "memory_feasible": metric.memory_feasible,
                }
                for metric in self.evaluation.batch_metrics
            ],
            "scan_trace": list(self.trace),
        }


@dataclass
class DiPSDSolver:
    """Small-scale solver following the paper's N-scan and alternating shape.

    The first implementation uses exact enumeration when the subproblem is
    small enough and deterministic coordinate search otherwise.  This keeps the
    platform dependency-free while preserving the paper's optimization
    boundary; a MILP backend can replace the subproblem methods later.
    """

    max_draft_length: int = 20
    initial_draft_length: int = 7
    min_batch_count: int = 2
    max_batch_count: int | None = None
    max_outer_iterations: int = 8
    tolerance: float = 1e-6
    max_assignment_enumerations: int = 200_000
    max_length_enumerations: int = 200_000
    dinkelbach_max_iterations: int = 16
    dinkelbach_tolerance: float = 1e-6
    solver_mode: str = "enumerate"

    def solve(self, users: list[DiPSDUserParams], config: DiPSDModelConfig) -> DiPSDSolution:
        mode = _normalize_solver_mode(self.solver_mode)
        if mode == "paper_milp":
            return self._solve_with_paper_milp(users, config)
        if mode == "paper_milp_or_enumerate":
            try:
                return self._solve_with_paper_milp(users, config)
            except DiPSDSolverBackendUnavailable as exc:
                solution = self._solve_with_enumeration(users, config)
                fallback_trace = {
                    "solver_backend_event": "paper_milp_unavailable",
                    "requested_solver_mode": self.solver_mode,
                    "fallback_solver_mode": "enumerate",
                    "reason": str(exc),
                }
                return _solution_with_backend(
                    solution,
                    requested_solver_mode=self.solver_mode,
                    solver_mode="enumerate",
                    backend_name="enumerate",
                    paper_solver_complete=False,
                    backend_fallback_used=True,
                    backend_fallback_reason=str(exc),
                    backend_info={
                        "paper_backend": "pyscipopt",
                        "paper_backend_available": False,
                        "fallback_policy": "paper_milp_or_enumerate",
                    },
                    prepend_trace=[fallback_trace],
                )
        if mode == "paper_milp_or_dinkelbach":
            try:
                return self._solve_with_paper_milp(users, config)
            except DiPSDSolverBackendUnavailable as exc:
                solution = self._solve_with_enumeration(users, config, length_solver_mode="dinkelbach")
                fallback_trace = {
                    "solver_backend_event": "paper_milp_unavailable",
                    "requested_solver_mode": self.solver_mode,
                    "fallback_solver_mode": "dinkelbach",
                    "reason": str(exc),
                }
                return _solution_with_backend(
                    solution,
                    requested_solver_mode=self.solver_mode,
                    solver_mode="dinkelbach",
                    backend_name="dinkelbach_coordinate",
                    paper_solver_complete=False,
                    backend_fallback_used=True,
                    backend_fallback_reason=str(exc),
                    backend_info={
                        "paper_backend": "pyscipopt",
                        "paper_backend_available": False,
                        "fallback_policy": "paper_milp_or_dinkelbach",
                        "length_subproblem": "dinkelbach_coordinate",
                    },
                    prepend_trace=[fallback_trace],
                )
        if mode == "dinkelbach":
            return self._solve_with_enumeration(users, config, length_solver_mode="dinkelbach")
        if mode in {"enumerate", "heuristic"}:
            return self._solve_with_enumeration(users, config, force_heuristic=(mode == "heuristic"))
        raise ValueError(
            "Unsupported DiP-SD solver mode "
            f"{self.solver_mode!r}. Expected enumerate, heuristic, paper_milp, "
            "dinkelbach, paper_milp_or_dinkelbach, or paper_milp_or_enumerate."
        )

    def _solve_with_paper_milp(self, users: list[DiPSDUserParams], config: DiPSDModelConfig) -> DiPSDSolution:
        if not _pyscipopt_available():
            raise DiPSDSolverBackendUnavailable(
                "DiP-SD paper_milp solver requires PySCIPOpt/SCIP, but pyscipopt is not installed."
            )
        if not users:
            raise ValueError("DiPSDSolver requires at least one user.")
        request_ids = [user.request_id for user in users]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("DiPSDSolver requires unique request ids.")

        all_trace: list[dict[str, Any]] = []
        best_solution: DiPSDSolution | None = None
        for batch_count in self._batch_count_scan(len(users)):
            solution = self._solve_fixed_batch_count(
                users,
                config,
                batch_count,
                assignment_solver_mode="paper_milp",
                length_solver_mode="paper_milp",
            )
            all_trace.extend(solution.trace)
            if not solution.evaluation.feasible:
                continue
            if best_solution is None or _better(solution.evaluation, best_solution.evaluation):
                best_solution = solution
        if best_solution is None:
            raise ValueError("No feasible DiP-SD schedule found.")
        return DiPSDSolution(
            batch_count=best_solution.batch_count,
            batches=best_solution.batches,
            draft_lengths=best_solution.draft_lengths,
            evaluation=best_solution.evaluation,
            trace=all_trace,
            solver_mode="paper_milp",
            requested_solver_mode=self.solver_mode,
            backend_name="pyscipopt_scip",
            paper_solver_complete=True,
            backend_info={
                "backend_family": "paper_milp",
                "paper_backend": "pyscipopt",
                "paper_backend_available": True,
                "x_subproblem": "set_partitioning_milp",
                "length_subproblem": "binary_selector_dinkelbach_milp",
                "pyscipopt_version": _pyscipopt_version(),
            },
        )

    def _solve_with_enumeration(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        *,
        force_heuristic: bool = False,
        length_solver_mode: str = "auto",
    ) -> DiPSDSolution:
        if not users:
            raise ValueError("DiPSDSolver requires at least one user.")
        request_ids = [user.request_id for user in users]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("DiPSDSolver requires unique request ids.")

        original_assignment_limit = self.max_assignment_enumerations
        original_length_limit = self.max_length_enumerations
        if force_heuristic:
            self.max_assignment_enumerations = -1
            self.max_length_enumerations = -1
        batch_counts = self._batch_count_scan(len(users))
        all_trace: list[dict[str, Any]] = []
        best_solution: DiPSDSolution | None = None
        try:
            for batch_count in batch_counts:
                solution = self._solve_fixed_batch_count(
                    users,
                    config,
                    batch_count,
                    length_solver_mode=length_solver_mode,
                )
                all_trace.extend(solution.trace)
                if not solution.evaluation.feasible:
                    continue
                if best_solution is None or _better(solution.evaluation, best_solution.evaluation):
                    best_solution = solution
            if best_solution is None:
                raise ValueError("No feasible DiP-SD schedule found.")
            actual_mode = "dinkelbach" if length_solver_mode == "dinkelbach" else ("heuristic" if force_heuristic else "enumerate")
            backend_name = "dinkelbach_coordinate" if length_solver_mode == "dinkelbach" else actual_mode
            return DiPSDSolution(
                batch_count=best_solution.batch_count,
                batches=best_solution.batches,
                draft_lengths=best_solution.draft_lengths,
                evaluation=best_solution.evaluation,
                trace=all_trace,
                solver_mode=actual_mode,
                requested_solver_mode=self.solver_mode,
                backend_name=backend_name,
                paper_solver_complete=False,
                backend_info={
                    "backend_family": "dependency_free",
                    "assignment_limit": original_assignment_limit,
                    "length_limit": original_length_limit,
                    "force_heuristic": bool(force_heuristic),
                    "length_subproblem": length_solver_mode,
                },
            )
        finally:
            self.max_assignment_enumerations = original_assignment_limit
            self.max_length_enumerations = original_length_limit

    def _batch_count_scan(self, user_count: int) -> list[int]:
        if user_count == 1:
            return [1]
        lo = max(2, int(self.min_batch_count))
        hi = int(self.max_batch_count or user_count)
        hi = max(lo, min(user_count, hi))
        return list(range(lo, hi + 1))

    def _solve_fixed_batch_count(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batch_count: int,
        *,
        assignment_solver_mode: str = "auto",
        length_solver_mode: str = "auto",
    ) -> DiPSDSolution:
        lengths = {
            user.request_id: self._clamp_length(self.initial_draft_length, user)
            for user in users
        }
        previous_throughput = -1.0
        trace: list[dict[str, Any]] = []
        best_batches: list[list[str]] | None = None
        best_eval: DiPSDScheduleEvaluation | None = None
        best_lengths: dict[str, int] | None = None
        for iteration in range(max(1, int(self.max_outer_iterations))):
            if assignment_solver_mode == "paper_milp":
                batches, x_eval, x_mode = self._solve_assignment_paper_milp(users, config, batch_count, lengths)
            else:
                batches, x_eval, x_mode = self._solve_assignment(users, config, batch_count, lengths)
            lengths, l_eval, l_mode, l_trace = self._solve_lengths(
                users,
                config,
                batches,
                lengths,
                length_solver_mode=length_solver_mode,
            )
            evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
            trace.append(
                {
                    "batch_count": batch_count,
                    "iteration": iteration,
                    "assignment_mode": x_mode,
                    "length_mode": l_mode,
                    "batches": [list(batch) for batch in batches],
                    "draft_lengths": dict(lengths),
                    "assignment_span_ms": x_eval.pipeline_span_ms,
                    "length_throughput_tokens_per_ms": l_eval.throughput_tokens_per_ms,
                    "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
                    "pipeline_span_ms": evaluation.pipeline_span_ms,
                    "feasible": evaluation.feasible,
                    "reason": evaluation.reason,
                    "length_trace": list(l_trace),
                }
            )
            if best_eval is None or _better(evaluation, best_eval):
                best_batches = batches
                best_eval = evaluation
                best_lengths = dict(lengths)
            if abs(evaluation.throughput_tokens_per_ms - previous_throughput) <= self.tolerance:
                break
            previous_throughput = evaluation.throughput_tokens_per_ms
        assert best_batches is not None and best_eval is not None and best_lengths is not None
        return DiPSDSolution(
            batch_count=batch_count,
            batches=tuple(tuple(batch) for batch in best_batches),
            draft_lengths=dict(best_lengths),
            evaluation=best_eval,
            trace=trace,
            solver_mode=self.solver_mode,
        )

    def _solve_assignment(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batch_count: int,
        draft_lengths: dict[str, int],
    ) -> tuple[list[list[str]], DiPSDScheduleEvaluation, str]:
        request_ids = [user.request_id for user in users]
        assignment_count = batch_count ** len(request_ids)
        if assignment_count <= self.max_assignment_enumerations:
            candidates = _nonempty_ordered_assignments(request_ids, batch_count)
            mode = "exact_assignment_enumeration"
        else:
            candidates = self._heuristic_assignments(users, config, batch_count, draft_lengths)
            mode = "heuristic_assignment_search"

        best_batches: list[list[str]] | None = None
        best_eval: DiPSDScheduleEvaluation | None = None
        for batches in candidates:
            evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=draft_lengths, config=config)
            if not evaluation.feasible:
                continue
            if best_eval is None or (
                evaluation.pipeline_span_ms,
                -evaluation.throughput_tokens_per_ms,
            ) < (
                best_eval.pipeline_span_ms,
                -best_eval.throughput_tokens_per_ms,
            ):
                best_batches = [list(batch) for batch in batches]
                best_eval = evaluation
        if best_batches is None or best_eval is None:
            fallback = [request_ids[index::batch_count] for index in range(batch_count)]
            fallback = [batch for batch in fallback if batch]
            return fallback, evaluate_schedule(users=users, batches=fallback, draft_lengths=draft_lengths, config=config), mode
        return best_batches, best_eval, mode

    def _solve_assignment_paper_milp(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batch_count: int,
        draft_lengths: dict[str, int],
    ) -> tuple[list[list[str]], DiPSDScheduleEvaluation, str]:
        scip = _import_pyscipopt()
        request_ids = [user.request_id for user in users]
        subset_metrics: dict[tuple[str, ...], tuple[float, float]] = {}
        for subset in _nonempty_subsets(request_ids):
            evaluation = evaluate_schedule(
                users=users,
                batches=[list(subset)],
                draft_lengths=draft_lengths,
                config=config,
            )
            if not evaluation.feasible or not evaluation.batch_metrics:
                continue
            metric = evaluation.batch_metrics[0]
            subset_metrics[subset] = (float(metric.draft_complete_ms), float(metric.verify_ms))
        if not subset_metrics:
            fallback = [request_ids[index::batch_count] for index in range(batch_count)]
            fallback = [batch for batch in fallback if batch]
            return (
                fallback,
                evaluate_schedule(users=users, batches=fallback, draft_lengths=draft_lengths, config=config),
                "paper_milp_x_subproblem_infeasible",
            )

        model = scip.Model("dip_sd_x_subproblem")
        _configure_scip_model(model)
        select: dict[tuple[int, tuple[str, ...]], Any] = {}
        for stage_index in range(batch_count):
            for subset in subset_metrics:
                select[(stage_index, subset)] = model.addVar(
                    vtype="B",
                    name=f"x_stage{stage_index}_{'_'.join(subset)}",
                )
        span = model.addVar(vtype="C", lb=0.0, name="S")
        for stage_index in range(batch_count):
            model.addCons(
                scip.quicksum(select[(stage_index, subset)] for subset in subset_metrics) == 1.0,
                name=f"one_subset_stage{stage_index}",
            )
        for request_id in request_ids:
            model.addCons(
                scip.quicksum(
                    select[(stage_index, subset)]
                    for stage_index in range(batch_count)
                    for subset in subset_metrics
                    if request_id in subset
                )
                == 1.0,
                name=f"cover_{request_id}",
            )
        verify_sum = scip.quicksum(
            select[(stage_index, subset)] * subset_metrics[subset][1]
            for stage_index in range(batch_count)
            for subset in subset_metrics
        )
        model.addCons(span >= verify_sum, name="span_covers_verify_sum")
        for stage_index in range(batch_count):
            ready_plus_verify = scip.quicksum(
                select[(stage_index, subset)] * (subset_metrics[subset][0] + subset_metrics[subset][1])
                for subset in subset_metrics
            )
            model.addCons(span >= ready_plus_verify, name=f"span_covers_stage{stage_index}")
        model.setObjective(span, "minimize")
        model.optimize()
        if not _scip_has_solution(model):
            fallback = [request_ids[index::batch_count] for index in range(batch_count)]
            fallback = [batch for batch in fallback if batch]
            return (
                fallback,
                evaluate_schedule(users=users, batches=fallback, draft_lengths=draft_lengths, config=config),
                f"paper_milp_x_subproblem_{model.getStatus()}",
            )

        batches: list[list[str]] = []
        for stage_index in range(batch_count):
            selected_subset = max(
                subset_metrics,
                key=lambda subset: float(model.getVal(select[(stage_index, subset)])),
            )
            batches.append(list(selected_subset))
        evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=draft_lengths, config=config)
        return batches, evaluation, "paper_milp_x_subproblem"

    def _solve_lengths(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batches: list[list[str]],
        current_lengths: dict[str, int],
        *,
        length_solver_mode: str = "auto",
    ) -> tuple[dict[str, int], DiPSDScheduleEvaluation, str, list[dict[str, Any]]]:
        if length_solver_mode == "paper_milp":
            return self._solve_lengths_paper_milp_dinkelbach(users, config, batches, current_lengths)
        if length_solver_mode == "dinkelbach":
            return self._solve_lengths_dinkelbach(users, config, batches, current_lengths)
        length_ranges = [
            range(1, self._max_length_for_user(user) + 1)
            for user in users
        ]
        enumeration_count = 1
        for values in length_ranges:
            enumeration_count *= len(values)
        if enumeration_count <= self.max_length_enumerations:
            best_lengths: dict[str, int] | None = None
            best_eval: DiPSDScheduleEvaluation | None = None
            for values in product(*length_ranges):
                lengths = {
                    user.request_id: int(value)
                    for user, value in zip(users, values)
                }
                evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
                if not evaluation.feasible:
                    continue
                if best_eval is None or _better(evaluation, best_eval):
                    best_lengths = lengths
                    best_eval = evaluation
            if best_lengths is not None and best_eval is not None:
                return best_lengths, best_eval, "exact_length_enumeration", []

        lengths = {user.request_id: self._clamp_length(current_lengths.get(user.request_id, self.initial_draft_length), user) for user in users}
        best_eval = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
        improved = True
        while improved:
            improved = False
            for user in users:
                local_best_length = lengths[user.request_id]
                local_best_eval = best_eval
                for length in range(1, self._max_length_for_user(user) + 1):
                    candidate_lengths = {**lengths, user.request_id: length}
                    evaluation = evaluate_schedule(
                        users=users,
                        batches=batches,
                        draft_lengths=candidate_lengths,
                        config=config,
                    )
                    if evaluation.feasible and _better(evaluation, local_best_eval):
                        local_best_length = length
                        local_best_eval = evaluation
                if local_best_length != lengths[user.request_id]:
                    lengths[user.request_id] = local_best_length
                    best_eval = local_best_eval
                    improved = True
        return lengths, best_eval, "coordinate_length_search", []

    def _solve_lengths_dinkelbach(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batches: list[list[str]],
        current_lengths: dict[str, int],
    ) -> tuple[dict[str, int], DiPSDScheduleEvaluation, str, list[dict[str, Any]]]:
        lengths = {
            user.request_id: self._clamp_length(current_lengths.get(user.request_id, self.initial_draft_length), user)
            for user in users
        }
        evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
        best_lengths = dict(lengths)
        best_eval = evaluation
        q = evaluation.throughput_tokens_per_ms if evaluation.feasible else 0.0
        trace: list[dict[str, Any]] = []
        for iteration in range(max(1, int(self.dinkelbach_max_iterations))):
            lengths, surrogate, evaluation = self._maximize_dinkelbach_surrogate(
                users,
                config,
                batches,
                lengths,
                q,
            )
            residual = (
                evaluation.expected_tokens - q * evaluation.pipeline_span_ms
                if evaluation.feasible
                else float("-inf")
            )
            trace.append(
                {
                    "length_solver": "dinkelbach_coordinate",
                    "iteration": iteration,
                    "q_tokens_per_ms": q,
                    "surrogate_objective": surrogate,
                    "residual": residual,
                    "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
                    "pipeline_span_ms": evaluation.pipeline_span_ms,
                    "draft_lengths": dict(lengths),
                    "feasible": evaluation.feasible,
                    "reason": evaluation.reason,
                }
            )
            if evaluation.feasible and (not best_eval.feasible or _better(evaluation, best_eval)):
                best_lengths = dict(lengths)
                best_eval = evaluation
            next_q = evaluation.throughput_tokens_per_ms if evaluation.feasible else q
            if abs(residual) <= self.dinkelbach_tolerance or abs(next_q - q) <= self.dinkelbach_tolerance:
                break
            q = next_q
        return best_lengths, best_eval, "dinkelbach_coordinate", trace

    def _solve_lengths_paper_milp_dinkelbach(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batches: list[list[str]],
        current_lengths: dict[str, int],
    ) -> tuple[dict[str, int], DiPSDScheduleEvaluation, str, list[dict[str, Any]]]:
        lengths = {
            user.request_id: self._clamp_length(current_lengths.get(user.request_id, self.initial_draft_length), user)
            for user in users
        }
        evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
        best_lengths = dict(lengths)
        best_eval = evaluation
        q = evaluation.throughput_tokens_per_ms if evaluation.feasible else 0.0
        trace: list[dict[str, Any]] = []
        for iteration in range(max(1, int(self.dinkelbach_max_iterations))):
            lengths, surrogate, evaluation, status = self._maximize_paper_milp_dinkelbach_surrogate(
                users,
                config,
                batches,
                q,
            )
            residual = (
                evaluation.expected_tokens - q * evaluation.pipeline_span_ms
                if evaluation.feasible
                else float("-inf")
            )
            trace.append(
                {
                    "length_solver": "paper_milp_dinkelbach",
                    "iteration": iteration,
                    "q_tokens_per_ms": q,
                    "surrogate_objective": surrogate,
                    "residual": residual,
                    "throughput_tokens_per_ms": evaluation.throughput_tokens_per_ms,
                    "pipeline_span_ms": evaluation.pipeline_span_ms,
                    "draft_lengths": dict(lengths),
                    "feasible": evaluation.feasible,
                    "reason": evaluation.reason,
                    "milp_status": status,
                }
            )
            if evaluation.feasible and (not best_eval.feasible or _better(evaluation, best_eval)):
                best_lengths = dict(lengths)
                best_eval = evaluation
            next_q = evaluation.throughput_tokens_per_ms if evaluation.feasible else q
            if abs(residual) <= self.dinkelbach_tolerance or abs(next_q - q) <= self.dinkelbach_tolerance:
                break
            q = next_q
        return best_lengths, best_eval, "paper_milp_dinkelbach", trace

    def _maximize_paper_milp_dinkelbach_surrogate(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batches: list[list[str]],
        q: float,
    ) -> tuple[dict[str, int], float, DiPSDScheduleEvaluation, str]:
        scip = _import_pyscipopt()
        users_by_id = {user.request_id: user for user in users}
        for batch in batches:
            if not batch:
                return {}, float("-inf"), _infeasible_length_result("empty_batch"), "empty_batch"
            memory_eval = evaluate_schedule(
                users=users,
                batches=[list(batch)],
                draft_lengths={user.request_id: 1 for user in users},
                config=config,
            )
            if not memory_eval.batch_metrics or memory_eval.batch_metrics[0].memory_bytes > float(config.memory_cap_bytes):
                return {}, float("-inf"), _infeasible_length_result("memory_infeasible"), "memory_infeasible"

        model = scip.Model("dip_sd_l_subproblem")
        _configure_scip_model(model)
        choose_len: dict[tuple[str, int], Any] = {}
        utility_terms: list[Any] = []
        length_exprs: dict[str, Any] = {}
        for user in users:
            max_len = self._max_length_for_user(user)
            for length in range(1, max_len + 1):
                choose_len[(user.request_id, length)] = model.addVar(
                    vtype="B",
                    name=f"y_{user.request_id}_{length}",
                )
                utility_terms.append(
                    choose_len[(user.request_id, length)]
                    * expected_accepted_tokens(length, user.acceptance)
                )
            model.addCons(
                scip.quicksum(choose_len[(user.request_id, length)] for length in range(1, max_len + 1))
                == 1.0,
                name=f"choose_len_{user.request_id}",
            )
            length_exprs[user.request_id] = scip.quicksum(
                length * choose_len[(user.request_id, length)]
                for length in range(1, max_len + 1)
            )

        span = model.addVar(vtype="C", lb=0.0, name="S")
        verify_terms: list[Any] = []
        for stage_index, batch in enumerate(batches):
            batch_users = [users_by_id[request_id] for request_id in batch]
            max_stage_len = max(self._max_length_for_user(user) for user in batch_users)
            max_prefix_len = max(int(user.prefix_len) for user in batch_users)
            choose_max: dict[int, Any] = {}
            for length in range(1, max_stage_len + 1):
                choose_max[length] = model.addVar(vtype="B", name=f"z_{stage_index}_{length}")
            model.addCons(
                scip.quicksum(choose_max[length] for length in range(1, max_stage_len + 1)) == 1.0,
                name=f"choose_stage_max_{stage_index}",
            )
            max_len_expr = scip.quicksum(
                length * choose_max[length]
                for length in range(1, max_stage_len + 1)
            )
            for user in batch_users:
                model.addCons(
                    length_exprs[user.request_id] <= max_len_expr,
                    name=f"max_len_{stage_index}_{user.request_id}",
                )
            verify_expr = scip.quicksum(
                choose_max[length]
                * evaluate_schedule(
                    users=users,
                    batches=[list(batch)],
                    draft_lengths={request_id: length for request_id in batch},
                    config=config,
                ).batch_metrics[0].verify_ms
                for length in range(1, max_stage_len + 1)
            )
            verify_terms.append(verify_expr)
            draft_ready = model.addVar(vtype="C", lb=0.0, name=f"draft_ready_{stage_index}")
            for user in batch_users:
                user_draft_expr = scip.quicksum(
                    choose_len[(user.request_id, length)]
                    * (draft_latency_ms(user, length, config) + max(0.0, float(user.comm_latency_ms)))
                    for length in range(1, self._max_length_for_user(user) + 1)
                )
                model.addCons(
                    draft_ready >= user_draft_expr,
                    name=f"draft_ready_{stage_index}_{user.request_id}",
                )
            model.addCons(span >= draft_ready + verify_expr, name=f"span_covers_stage{stage_index}")
        model.addCons(span >= scip.quicksum(verify_terms), name="span_covers_verify_sum")
        objective = scip.quicksum(utility_terms) - float(q) * span
        model.setObjective(objective, "maximize")
        model.optimize()
        status = str(model.getStatus())
        if not _scip_has_solution(model):
            return {}, float("-inf"), _infeasible_length_result(status), status
        lengths: dict[str, int] = {}
        for user in users:
            max_len = self._max_length_for_user(user)
            lengths[user.request_id] = max(
                range(1, max_len + 1),
                key=lambda length: float(model.getVal(choose_len[(user.request_id, length)])),
            )
        evaluation = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
        return lengths, float(model.getObjVal()), evaluation, status

    def _maximize_dinkelbach_surrogate(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batches: list[list[str]],
        start_lengths: dict[str, int],
        q: float,
    ) -> tuple[dict[str, int], float, DiPSDScheduleEvaluation]:
        lengths = dict(start_lengths)
        best_eval = evaluate_schedule(users=users, batches=batches, draft_lengths=lengths, config=config)
        best_score = _dinkelbach_score(best_eval, q)
        improved = True
        while improved:
            improved = False
            for user in users:
                local_best_length = lengths[user.request_id]
                local_best_eval = best_eval
                local_best_score = best_score
                for length in range(1, self._max_length_for_user(user) + 1):
                    candidate_lengths = {**lengths, user.request_id: length}
                    evaluation = evaluate_schedule(
                        users=users,
                        batches=batches,
                        draft_lengths=candidate_lengths,
                        config=config,
                    )
                    score = _dinkelbach_score(evaluation, q)
                    if (
                        score,
                        evaluation.throughput_tokens_per_ms,
                        -evaluation.pipeline_span_ms,
                    ) > (
                        local_best_score,
                        local_best_eval.throughput_tokens_per_ms,
                        -local_best_eval.pipeline_span_ms,
                    ):
                        local_best_length = length
                        local_best_eval = evaluation
                        local_best_score = score
                if local_best_length != lengths[user.request_id]:
                    lengths[user.request_id] = local_best_length
                    best_eval = local_best_eval
                    best_score = local_best_score
                    improved = True
        return lengths, best_score, best_eval

    def _heuristic_assignments(
        self,
        users: list[DiPSDUserParams],
        config: DiPSDModelConfig,
        batch_count: int,
        draft_lengths: dict[str, int],
    ) -> list[list[list[str]]]:
        ordered = sorted(
            users,
            key=lambda user: (
                draft_lengths.get(user.request_id, self.initial_draft_length),
                user.prefix_len,
            ),
            reverse=True,
        )
        batches = [[] for _ in range(batch_count)]
        for user in ordered:
            best_index = 0
            best_eval: DiPSDScheduleEvaluation | None = None
            for index in range(batch_count):
                candidate = [list(batch) for batch in batches]
                candidate[index].append(user.request_id)
                if any(not batch for batch in candidate[: index]):
                    continue
                evaluation = evaluate_schedule(
                    users=users,
                    batches=[batch for batch in candidate if batch],
                    draft_lengths=draft_lengths,
                    config=config,
                )
                if evaluation.feasible and (best_eval is None or evaluation.pipeline_span_ms < best_eval.pipeline_span_ms):
                    best_index = index
                    best_eval = evaluation
            batches[best_index].append(user.request_id)
        nonempty = [batch for batch in batches if batch]
        return [nonempty]

    def _max_length_for_user(self, user: DiPSDUserParams) -> int:
        limit = max(1, int(self.max_draft_length))
        if user.remaining_tokens is not None:
            limit = min(limit, max(1, int(user.remaining_tokens)))
        return limit

    def _clamp_length(self, value: int, user: DiPSDUserParams) -> int:
        return max(1, min(int(value), self._max_length_for_user(user)))


def _nonempty_ordered_assignments(request_ids: list[str], batch_count: int) -> list[list[list[str]]]:
    assignments: list[list[list[str]]] = []
    for slots in product(range(batch_count), repeat=len(request_ids)):
        if len(set(slots)) != batch_count:
            continue
        batches = [[] for _ in range(batch_count)]
        for request_id, slot in zip(request_ids, slots):
            batches[int(slot)].append(request_id)
        assignments.append(batches)
    return assignments


def _nonempty_subsets(request_ids: list[str]) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for mask in range(1, 1 << len(request_ids)):
        subsets.append(
            tuple(request_id for index, request_id in enumerate(request_ids) if mask & (1 << index))
        )
    return subsets


def _better(candidate: DiPSDScheduleEvaluation, incumbent: DiPSDScheduleEvaluation) -> bool:
    return (
        candidate.feasible,
        candidate.throughput_tokens_per_ms,
        -candidate.pipeline_span_ms,
    ) > (
        incumbent.feasible,
        incumbent.throughput_tokens_per_ms,
        -incumbent.pipeline_span_ms,
    )


def _dinkelbach_score(evaluation: DiPSDScheduleEvaluation, q: float) -> float:
    if not evaluation.feasible:
        return float("-inf")
    return float(evaluation.expected_tokens) - float(q) * float(evaluation.pipeline_span_ms)


def _infeasible_length_result(reason: str) -> DiPSDScheduleEvaluation:
    return DiPSDScheduleEvaluation(
        feasible=False,
        throughput_tokens_per_ms=0.0,
        expected_tokens=0.0,
        pipeline_span_ms=0.0,
        verification_sum_ms=0.0,
        max_draft_verify_ms=0.0,
        batch_metrics=(),
        reason=reason,
    )


class DiPSDSolverBackendUnavailable(RuntimeError):
    """Raised when a requested DiP-SD solver backend cannot run on this host."""


def _solution_with_backend(
    solution: DiPSDSolution,
    *,
    requested_solver_mode: str,
    solver_mode: str,
    backend_name: str,
    paper_solver_complete: bool,
    backend_fallback_used: bool,
    backend_fallback_reason: str | None,
    backend_info: dict[str, Any],
    prepend_trace: list[dict[str, Any]] | None = None,
) -> DiPSDSolution:
    return DiPSDSolution(
        batch_count=solution.batch_count,
        batches=solution.batches,
        draft_lengths=solution.draft_lengths,
        evaluation=solution.evaluation,
        trace=[*(prepend_trace or []), *solution.trace],
        solver_mode=solver_mode,
        requested_solver_mode=requested_solver_mode,
        backend_name=backend_name,
        paper_solver_complete=paper_solver_complete,
        backend_fallback_used=backend_fallback_used,
        backend_fallback_reason=backend_fallback_reason,
        backend_info=dict(backend_info),
    )


def _normalize_solver_mode(value: str) -> str:
    mode = str(value or "enumerate").strip().lower().replace("-", "_")
    aliases = {
        "milp": "paper_milp",
        "scip": "paper_milp",
        "pyscipopt": "paper_milp",
        "paper": "paper_milp",
        "auto": "paper_milp_or_enumerate",
        "paper_milp_or_fallback": "paper_milp_or_enumerate",
        "paper_milp_fallback": "paper_milp_or_enumerate",
        "paper_milp_or_dinkelbach_fallback": "paper_milp_or_dinkelbach",
        "paper_dinkelbach": "dinkelbach",
        "dinkelbach_coordinate": "dinkelbach",
        "fallback": "paper_milp_or_enumerate",
        "enumeration": "enumerate",
        "exact": "enumerate",
        "fast": "heuristic",
    }
    return aliases.get(mode, mode)


def _pyscipopt_available() -> bool:
    return importlib.util.find_spec("pyscipopt") is not None


def _import_pyscipopt() -> Any:
    try:
        return importlib.import_module("pyscipopt")
    except ImportError as exc:
        raise DiPSDSolverBackendUnavailable(
            "DiP-SD paper_milp solver requires PySCIPOpt/SCIP, but pyscipopt is not installed."
        ) from exc


def _pyscipopt_version() -> str | None:
    if not _pyscipopt_available():
        return None
    module = _import_pyscipopt()
    return None if getattr(module, "__version__", None) is None else str(module.__version__)


def _configure_scip_model(model: Any) -> None:
    try:
        model.hideOutput()
    except AttributeError:
        pass


def _scip_has_solution(model: Any) -> bool:
    try:
        if int(model.getNSols()) > 0:
            return True
    except (AttributeError, TypeError, ValueError):
        pass
    return str(model.getStatus()).lower() in {"optimal", "bestsollimit"}
