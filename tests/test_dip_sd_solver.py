"""DiP-SD paper cost model and solver tests."""

import importlib.util
import unittest
from unittest.mock import patch

from specplatform.core import RuntimeContext
from specplatform.methods.dip_sd import DiPSDPlanningPolicy
from specplatform.methods.dip_sd.model import (
    DiPSDModelConfig,
    DiPSDUserParams,
    evaluate_schedule,
    expected_accepted_tokens,
)
from specplatform.methods.dip_sd.solver import DiPSDSolver, DiPSDSolverBackendUnavailable


class SessionStub:
    def __init__(self, request_id: str, remaining_tokens: int, prefix_len: int = 5) -> None:
        self.request_id = request_id
        self.remaining_tokens = remaining_tokens
        self.prompt_ids = list(range(prefix_len))
        self.generated_ids = []

    @property
    def prefix_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.generated_ids]


class DiPSDSolverTest(unittest.TestCase):
    def test_expected_accepted_tokens_includes_bonus_token(self) -> None:
        self.assertAlmostEqual(expected_accepted_tokens(0, 0.78), 1.0)
        self.assertAlmostEqual(expected_accepted_tokens(2, 0.5), 1.75)
        self.assertAlmostEqual(expected_accepted_tokens(3, 1.0), 4.0)

    def test_evaluate_schedule_reports_pipeline_span_and_memory(self) -> None:
        users = [
            _user("r1", prefix_len=8),
            _user("r2", prefix_len=16),
        ]
        config = DiPSDModelConfig(memory_cap_bytes=8.0e10)

        evaluation = evaluate_schedule(
            users=users,
            batches=[["r1", "r2"]],
            draft_lengths={"r1": 3, "r2": 5},
            config=config,
        )

        self.assertTrue(evaluation.feasible)
        self.assertGreater(evaluation.expected_tokens, 2.0)
        self.assertGreater(evaluation.pipeline_span_ms, 0.0)
        self.assertEqual(evaluation.batch_metrics[0].max_draft_len, 5)
        self.assertEqual(evaluation.batch_metrics[0].max_prefix_len, 16)

    def test_evaluate_schedule_rejects_memory_infeasible_batch(self) -> None:
        users = [_user("r1", prefix_len=512), _user("r2", prefix_len=512)]
        config = DiPSDModelConfig(memory_cap_bytes=1.0)

        evaluation = evaluate_schedule(
            users=users,
            batches=[["r1", "r2"]],
            draft_lengths={"r1": 2, "r2": 2},
            config=config,
        )

        self.assertFalse(evaluation.feasible)
        self.assertEqual(evaluation.reason, "memory_infeasible")

    def test_solver_scans_batch_count_and_returns_positive_lengths(self) -> None:
        users = [_user(f"r{index}", prefix_len=8 + index) for index in range(1, 4)]
        solver = DiPSDSolver(
            max_draft_length=5,
            initial_draft_length=2,
            max_batch_count=3,
            max_length_enumerations=10_000,
        )

        solution = solver.solve(users, DiPSDModelConfig())

        self.assertIn(solution.batch_count, {2, 3})
        self.assertEqual(set().union(*[set(batch) for batch in solution.batches]), {"r1", "r2", "r3"})
        self.assertTrue(all(1 <= length <= 5 for length in solution.draft_lengths.values()))
        self.assertTrue(solution.evaluation.feasible)
        self.assertGreater(solution.throughput_tokens_per_ms, 0.0)
        self.assertGreater(len(solution.trace), 0)

    def test_paper_milp_backend_is_explicitly_unavailable_without_backend(self) -> None:
        solver = DiPSDSolver(max_draft_length=3, initial_draft_length=2, solver_mode="paper_milp")

        with patch("specplatform.methods.dip_sd.solver._pyscipopt_available", return_value=False):
            with self.assertRaises(DiPSDSolverBackendUnavailable):
                solver.solve([_user("r1"), _user("r2")], DiPSDModelConfig())

    @unittest.skipIf(importlib.util.find_spec("pyscipopt") is None, "PySCIPOpt is not installed")
    def test_paper_milp_backend_solves_with_pyscipopt(self) -> None:
        solver = DiPSDSolver(
            max_draft_length=4,
            initial_draft_length=2,
            max_batch_count=2,
            max_outer_iterations=3,
            dinkelbach_max_iterations=8,
            solver_mode="paper_milp",
        )

        solution = solver.solve([_user("r1"), _user("r2"), _user("r3")], DiPSDModelConfig())

        self.assertEqual(solution.solver_mode, "paper_milp")
        self.assertEqual(solution.backend_name, "pyscipopt_scip")
        self.assertTrue(solution.paper_solver_complete)
        self.assertTrue(solution.evaluation.feasible)
        self.assertTrue(all(1 <= length <= 4 for length in solution.draft_lengths.values()))
        self.assertTrue(any(item.get("assignment_mode") == "paper_milp_x_subproblem" for item in solution.trace))
        self.assertTrue(any(item.get("length_mode") == "paper_milp_dinkelbach" for item in solution.trace))
        self.assertTrue(
            any(
                step.get("length_solver") == "paper_milp_dinkelbach"
                for item in solution.trace
                for step in item.get("length_trace", [])
            )
        )

        exact = DiPSDSolver(
            max_draft_length=4,
            initial_draft_length=2,
            max_batch_count=2,
            max_outer_iterations=3,
            max_length_enumerations=10_000,
            solver_mode="enumerate",
        ).solve([_user("r1"), _user("r2"), _user("r3")], DiPSDModelConfig())
        self.assertGreaterEqual(
            solution.evaluation.throughput_tokens_per_ms,
            exact.evaluation.throughput_tokens_per_ms * 0.95,
        )

    def test_paper_milp_or_enumerate_records_backend_fallback(self) -> None:
        solver = DiPSDSolver(
            max_draft_length=3,
            initial_draft_length=2,
            max_batch_count=2,
            solver_mode="paper_milp_or_enumerate",
        )

        with patch("specplatform.methods.dip_sd.solver._pyscipopt_available", return_value=False):
            solution = solver.solve([_user("r1"), _user("r2")], DiPSDModelConfig())

        self.assertEqual(solution.requested_solver_mode, "paper_milp_or_enumerate")
        self.assertEqual(solution.solver_mode, "enumerate")
        self.assertEqual(solution.backend_name, "enumerate")
        self.assertFalse(solution.paper_solver_complete)
        self.assertTrue(solution.backend_fallback_used)
        self.assertIn("PySCIPOpt", solution.backend_fallback_reason or "")
        self.assertEqual(solution.trace[0]["solver_backend_event"], "paper_milp_unavailable")

    def test_paper_milp_or_dinkelbach_records_backend_fallback(self) -> None:
        solver = DiPSDSolver(
            max_draft_length=3,
            initial_draft_length=2,
            max_batch_count=2,
            solver_mode="paper_milp_or_dinkelbach",
        )

        with patch("specplatform.methods.dip_sd.solver._pyscipopt_available", return_value=False):
            solution = solver.solve([_user("r1"), _user("r2")], DiPSDModelConfig())

        self.assertEqual(solution.requested_solver_mode, "paper_milp_or_dinkelbach")
        self.assertEqual(solution.solver_mode, "dinkelbach")
        self.assertEqual(solution.backend_name, "dinkelbach_coordinate")
        self.assertFalse(solution.paper_solver_complete)
        self.assertTrue(solution.backend_fallback_used)
        self.assertIn("PySCIPOpt", solution.backend_fallback_reason or "")
        self.assertEqual(solution.trace[0]["fallback_solver_mode"], "dinkelbach")

    def test_dinkelbach_mode_records_length_subproblem_trace(self) -> None:
        solver = DiPSDSolver(
            max_draft_length=5,
            initial_draft_length=2,
            max_batch_count=2,
            solver_mode="dinkelbach",
        )

        solution = solver.solve([_user("r1"), _user("r2"), _user("r3")], DiPSDModelConfig())

        self.assertEqual(solution.solver_mode, "dinkelbach")
        self.assertEqual(solution.backend_name, "dinkelbach_coordinate")
        self.assertTrue(solution.evaluation.feasible)
        self.assertFalse(solution.paper_solver_complete)
        self.assertTrue(
            any(item.get("length_mode") == "dinkelbach_coordinate" for item in solution.trace)
        )
        self.assertTrue(
            any(
                step.get("length_solver") == "dinkelbach_coordinate"
                for item in solution.trace
                for step in item.get("length_trace", [])
            )
        )

    def test_policy_maps_solution_to_plan_hints(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6), SessionStub("r3", 6)]
        resources = {
            "draft_worker_ids": ["w0", "w1", "w2"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
                "w1": {"speed_profile": {"relative_speed": 2.0, "quality": 0.9}},
                "w2": {"speed_profile": {"relative_speed": 1.5, "quality": 0.85}},
            },
        }

        hints = DiPSDPlanningPolicy(
            max_draft_length=4,
            initial_draft_length=2,
            max_batch_count=2,
        ).plan(sessions, resources=resources, history={}, context=RuntimeContext())

        self.assertEqual(hints.metadata["method_family"], "dip_sd")
        self.assertTrue(hints.metadata["solver_active"])
        self.assertTrue(hints.metadata["distributed_local_drafting"])
        self.assertTrue(hints.metadata["central_batch_verification"])
        self.assertEqual(len(hints.preferred_batches), hints.metadata["planned_batch_count"])
        self.assertEqual(len(hints.metadata["preferred_batch_metadata"]), len(hints.preferred_batches))
        self.assertTrue(all(length > 0 for length in hints.draft_lengths.values()))

    def test_policy_can_hybrid_single_batch_small_request_sets(self) -> None:
        """实验平台可把小并发 DiP-SD 计划落成真实 batched verification。"""
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}

        hints = DiPSDPlanningPolicy(
            max_draft_length=4,
            initial_draft_length=2,
            max_batch_count=2,
        ).plan(
            sessions,
            resources=resources,
            history={},
            context=RuntimeContext(method_config={"dip_sd_single_batch_small_request_threshold": 2}),
        )

        self.assertEqual(hints.preferred_batches, [["r1", "r2"]])
        self.assertEqual(hints.metadata["planned_batch_count"], 1)
        self.assertEqual(hints.metadata["solver_planned_batch_count"], 2)
        self.assertTrue(hints.metadata["hybrid_single_batch_applied"])
        self.assertEqual(hints.metadata["hybrid_single_batch_reason"], "small_request_batching")
        self.assertEqual(hints.metadata["preferred_batch_metadata"][0]["request_ids"], ["r1", "r2"])

    def test_policy_exposes_solver_backend_audit_metadata(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}

        with patch("specplatform.methods.dip_sd.solver._pyscipopt_available", return_value=False):
            hints = DiPSDPlanningPolicy(max_draft_length=3, initial_draft_length=2, max_batch_count=2).plan(
                sessions,
                resources=resources,
                history={},
                context=RuntimeContext(
                    method_config={
                        "dip_sd_solver": "paper_milp_or_enumerate",
                        "dip_sd_calibration_profile": "/tmp/profile.json",
                        "dip_sd_calibration_applied": True,
                        "dip_sd_calibration_overrides": {"dip_sd_draft_beta": 12.5},
                    }
                ),
            )

        self.assertEqual(hints.metadata["requested_solver_mode"], "paper_milp_or_enumerate")
        self.assertEqual(hints.metadata["solver_backend_name"], "enumerate")
        self.assertFalse(hints.metadata["paper_solver_complete"])
        self.assertTrue(hints.metadata["solver_backend_fallback_used"])
        self.assertIn("PySCIPOpt", hints.metadata["solver_backend_fallback_reason"])
        self.assertEqual(hints.metadata["latency_calibration_profile"], "/tmp/profile.json")
        self.assertTrue(hints.metadata["latency_calibration_applied"])
        self.assertEqual(hints.metadata["latency_calibration_overrides"], {"dip_sd_draft_beta": 12.5})

    def test_policy_reuses_cached_plan_for_same_active_set(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {
            "draft_worker_ids": ["w0", "w1"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
                "w1": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
            },
        }
        policy = DiPSDPlanningPolicy(max_draft_length=3, initial_draft_length=2, max_batch_count=2)

        first = policy.plan(sessions, resources=resources, history={}, context=RuntimeContext())
        second = policy.plan(sessions, resources=resources, history={}, context=RuntimeContext())

        self.assertFalse(first.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["solver_cache_hit"])
        self.assertEqual(first.draft_lengths, second.draft_lengths)
        self.assertEqual(first.preferred_batches, second.preferred_batches)

    def test_policy_reuses_shape_cache_for_same_shape_different_requests(self) -> None:
        first_sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        second_sessions = [SessionStub("x1", 6), SessionStub("x2", 6)]
        resources = {
            "draft_worker_ids": ["w0", "w1"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
                "w1": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
            },
        }
        policy = DiPSDPlanningPolicy(max_draft_length=3, initial_draft_length=2, max_batch_count=2)

        first = policy.plan(first_sessions, resources=resources, history={}, context=RuntimeContext())
        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            second = policy.plan(second_sessions, resources=resources, history={}, context=RuntimeContext())

        self.assertFalse(first.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["solver_cache_shape_level"])
        self.assertFalse(second.metadata["solver_active"])
        self.assertEqual(second.metadata["solver_backend_name"], "shape_plan_cache")
        self.assertEqual(set(second.draft_lengths), {"x1", "x2"})
        self.assertEqual(set().union(*[set(batch) for batch in second.preferred_batches]), {"x1", "x2"})

    def test_policy_falls_back_to_alpha_when_worker_quality_is_missing(self) -> None:
        sessions = [SessionStub("r1", 4), SessionStub("r2", 4)]
        resources = {
            "draft_worker_ids": ["w0"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0, "quality": None}},
            },
        }

        hints = DiPSDPlanningPolicy(max_draft_length=3, initial_draft_length=2).plan(
            sessions,
            resources=resources,
            history={},
            context=RuntimeContext(method_config={"dip_sd_alpha": 0.78}),
        )

        self.assertTrue(hints.metadata["solver_active"])
        self.assertTrue(all(length > 0 for length in hints.draft_lengths.values()))

    def test_policy_uses_acceptance_history_for_length_model(self) -> None:
        sessions = [SessionStub("r1", 8)]
        resources = {"draft_worker_ids": ["w0"], "draft_worker_metadata": {}}
        history = {
            "dip_sd_acceptance_stats": {
                "r1": {
                    "proposal_count": 2,
                    "draft_token_count": 4,
                    "accepted_draft_count": 1,
                    "bonus_count": 2,
                    "output_token_count": 3,
                    "last_observed_acceptance": 0.0,
                }
            }
        }

        hints = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2).plan(
            sessions,
            resources=resources,
            history=history,
            context=RuntimeContext(
                method_config={
                    "dip_sd_alpha": 0.78,
                    "dip_sd_acceptance_feedback_prior_weight": 0.0,
                }
            ),
        )

        feedback = hints.metadata["acceptance_feedback_by_request"]["r1"]
        self.assertTrue(hints.metadata["acceptance_feedback_enabled"])
        self.assertEqual(hints.metadata["acceptance_feedback_applied_count"], 1)
        self.assertAlmostEqual(feedback["observed_acceptance"], 0.25)
        self.assertAlmostEqual(feedback["effective_acceptance"], 0.25)
        self.assertAlmostEqual(feedback["prior_acceptance"], 0.78)

    def test_policy_uses_shape_level_cache_for_binned_acceptance_feedback(self) -> None:
        sessions = [SessionStub("r1", 8)]
        resources = {"draft_worker_ids": ["w0"], "draft_worker_metadata": {}}
        policy = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2)
        context = RuntimeContext(
            method_config={
                "dip_sd_acceptance_feedback_prior_weight": 0.0,
                "dip_sd_acceptance_cache_bucket": 0.25,
            }
        )

        first = policy.plan(
            sessions,
            resources=resources,
            history={
                "dip_sd_acceptance_stats": {
                    "r1": {"draft_token_count": 4, "accepted_draft_count": 2, "proposal_count": 1}
                }
            },
            context=context,
        )
        second = policy.plan(
            sessions,
            resources=resources,
            history={
                "dip_sd_acceptance_stats": {
                    "r1": {"draft_token_count": 5, "accepted_draft_count": 3, "proposal_count": 2}
                }
            },
            context=context,
        )

        self.assertFalse(first.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["solver_cache_shape_level"])
        self.assertAlmostEqual(
            second.metadata["acceptance_feedback_by_request"]["r1"]["effective_acceptance"],
            0.6,
        )

    def test_policy_online_solver_metadata_exposes_offline_shape_key(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}
        policy = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2)
        context = RuntimeContext(method_config={"dip_sd_offline_plan_prefix_bucket": 8})

        first = policy.plan(sessions, resources=resources, history={}, context=context)
        second = policy.plan(sessions, resources=resources, history={}, context=context)

        self.assertTrue(first.metadata["online_solver_enabled"])
        self.assertFalse(first.metadata["offline_plan_table_hit"])
        self.assertIn("requests=2|workers=2", first.metadata["offline_plan_shape_key"])
        self.assertTrue(second.metadata["solver_cache_hit"])
        self.assertTrue(second.metadata["online_solver_enabled"])
        self.assertFalse(second.metadata["offline_plan_table_hit"])
        self.assertEqual(
            second.metadata["offline_plan_shape_key"],
            first.metadata["offline_plan_shape_key"],
        )

    def test_policy_uses_offline_plan_table_without_online_solver(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}
        context = RuntimeContext(
            method_config={
                "dip_sd_solver": "offline_table",
                "dip_sd_no_online_solver": True,
                "dip_sd_max_draft_length": 4,
                "dip_sd_offline_plan_table": {
                    "default": {
                        "draft_lengths": 2,
                        "preferred_batches": "single_batch",
                        "solver_planned_batch_count": 2,
                        "hybrid_single_batch_applied": True,
                        "hybrid_single_batch_reason": "offline_small_request_batching",
                        "source": "unit-test-table",
                    }
                },
            }
        )

        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            hints = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2).plan(
                sessions,
                resources=resources,
                history={},
                context=context,
            )

        self.assertEqual(hints.draft_lengths, {"r1": 2, "r2": 2})
        self.assertEqual(hints.preferred_batches, [["r1", "r2"]])
        self.assertFalse(hints.metadata["solver_active"])
        self.assertFalse(hints.metadata["online_solver_enabled"])
        self.assertTrue(hints.metadata["offline_plan_table_hit"])
        self.assertTrue(hints.metadata["solver_cache_hit"])
        self.assertEqual(hints.metadata["solver_backend_name"], "offline_plan_table")
        self.assertEqual(hints.metadata["solver_planned_batch_count"], 2)

    def test_policy_adapts_slow_worker_lengths_and_rebatches_by_ready_time(self) -> None:
        sessions = [SessionStub(f"r{index}", 6) for index in range(4)]
        resources = {
            "draft_worker_ids": ["slow", "fast"],
            "draft_worker_metadata": {
                "slow": {"speed_profile": {"relative_speed": 1.0, "latency_ms": 4.0}},
                "fast": {"speed_profile": {"relative_speed": 4.0, "latency_ms": 1.0}},
            },
        }
        context = RuntimeContext(
            method_config={
                "dip_sd_solver": "offline_table",
                "dip_sd_no_online_solver": True,
                "dip_sd_initial_draft_length": 4,
                "dip_sd_max_draft_length": 4,
                "dip_sd_ready_aware_rebatch_min_spread_ms": 0.0,
                "dip_sd_offline_plan_table": {
                    "default": {
                        "draft_lengths": 4,
                        "preferred_batches": [[0, 1], [2, 3]],
                    }
                },
            }
        )

        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            hints = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=4).plan(
                sessions,
                resources=resources,
                history={},
                context=context,
            )

        self.assertEqual(hints.worker_preferences, {"r0": "fast", "r1": "slow", "r2": "fast", "r3": "slow"})
        self.assertEqual(hints.draft_lengths["r0"], 4)
        self.assertEqual(hints.draft_lengths["r2"], 4)
        self.assertLess(hints.draft_lengths["r1"], 4)
        self.assertLess(hints.draft_lengths["r3"], 4)
        self.assertTrue(hints.metadata["adaptive_draft_length_applied"])
        self.assertTrue(hints.metadata["ready_aware_rebatch_applied"])
        self.assertEqual(hints.preferred_batches[0], ["r0", "r2"])

    def test_policy_skips_ready_aware_rebatch_when_ready_spread_is_tiny(self) -> None:
        sessions = [SessionStub(f"r{index}", 6) for index in range(4)]
        resources = {
            "draft_worker_ids": ["w0", "w1"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
                "w1": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
            },
        }
        context = RuntimeContext(
            method_config={
                "dip_sd_solver": "offline_table",
                "dip_sd_no_online_solver": True,
                "dip_sd_initial_draft_length": 2,
                "dip_sd_max_draft_length": 2,
                "dip_sd_adaptive_draft_length_enabled": False,
                "dip_sd_ready_aware_rebatch_min_spread_ms": 5.0,
                "dip_sd_offline_plan_table": {
                    "default": {
                        "draft_lengths": 2,
                        "preferred_batches": [[1, 0], [3, 2]],
                    }
                },
            }
        )

        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            hints = DiPSDPlanningPolicy(max_draft_length=2, initial_draft_length=2).plan(
                sessions,
                resources=resources,
                history={},
                context=context,
            )

        self.assertEqual(hints.preferred_batches, [["r1", "r0"], ["r3", "r2"]])
        self.assertFalse(hints.metadata["ready_aware_rebatch_applied"])
        self.assertEqual(hints.metadata["ready_aware_rebatch_reason"], "ready_spread_below_threshold")
        self.assertLess(hints.metadata["ready_aware_rebatch_spread_ms"], 5.0)

    def test_policy_prefers_prefetched_worker_for_next_round(self) -> None:
        sessions = [SessionStub("r0", 6), SessionStub("r1", 6)]
        resources = {
            "draft_worker_ids": ["fast", "slow"],
            "draft_worker_metadata": {
                "fast": {"speed_profile": {"relative_speed": 4.0, "quality": 0.8}},
                "slow": {"speed_profile": {"relative_speed": 1.0, "quality": 0.8}},
            },
        }
        context = RuntimeContext(
            method_config={
                "dip_sd_solver": "offline_heuristic",
                "dip_sd_no_online_solver": True,
                "dip_sd_prefetch_sticky_worker_enabled": True,
                "dip_sd_ready_aware_rebatch_enabled": False,
            }
        )

        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            hints = DiPSDPlanningPolicy(max_draft_length=2, initial_draft_length=2).plan(
                sessions,
                resources=resources,
                history={"dip_sd_prefetch_by_request": {"r0": {"worker_id": "slow", "budget_tokens": 2}}},
                context=context,
            )

        self.assertEqual(hints.worker_preferences["r0"], "slow")
        self.assertEqual(hints.worker_preferences["r1"], "fast")

    def test_policy_no_online_solver_can_use_heuristic_fallback(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6), SessionStub("r3", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}
        context = RuntimeContext(
            method_config={
                "dip_sd_solver": "offline_heuristic",
                "dip_sd_no_online_solver": True,
                "dip_sd_initial_draft_length": 2,
                "dip_sd_max_draft_length": 4,
                "dip_sd_min_batch_count": 2,
            }
        )

        with patch.object(DiPSDSolver, "solve", side_effect=AssertionError("online solver called")):
            hints = DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2).plan(
                sessions,
                resources=resources,
                history={},
                context=context,
            )

        self.assertFalse(hints.metadata["solver_active"])
        self.assertFalse(hints.metadata["online_solver_enabled"])
        self.assertFalse(hints.metadata["offline_plan_table_hit"])
        self.assertEqual(hints.metadata["solver_backend_name"], "no_online_heuristic")
        self.assertEqual(hints.metadata["solver_mode"], "offline_heuristic")
        self.assertEqual(len(hints.preferred_batches), 2)
        self.assertEqual(hints.draft_lengths, {"r1": 2, "r2": 2, "r3": 2})

    def test_policy_no_online_solver_requires_offline_plan_entry(self) -> None:
        sessions = [SessionStub("r1", 6), SessionStub("r2", 6)]
        resources = {"draft_worker_ids": ["w0", "w1"], "draft_worker_metadata": {}}

        with self.assertRaisesRegex(ValueError, "offline plan table entry"):
            DiPSDPlanningPolicy(max_draft_length=4, initial_draft_length=2).plan(
                sessions,
                resources=resources,
                history={},
                context=RuntimeContext(
                    method_config={
                        "dip_sd_solver": "offline_table",
                        "dip_sd_no_online_solver": True,
                    }
                ),
            )


def _user(request_id: str, *, prefix_len: int = 8, acceptance: float = 0.78) -> DiPSDUserParams:
    return DiPSDUserParams(
        request_id=request_id,
        prefix_len=prefix_len,
        acceptance=acceptance,
        comm_latency_ms=3.0,
        draft_c=4.0305e-11,
        draft_beta=33.8151,
        remaining_tokens=8,
    )


if __name__ == "__main__":
    unittest.main()
