"""Draft worker registry and reusable scheduling policy tests."""

from pathlib import Path
import unittest
from typing import Any

from specplatform.core import DraftBudget, PlanHints, RuntimeContext
from specplatform.draft import DraftWorkerConfig, DraftWorkerRegistry
from specplatform.methods import DiPSDPlanningPolicy, SLEDPlanningPolicy
from specplatform.methods.planning_math import _assignment_score_and_tail
from specplatform.model import CausalLMRunner, ModelForwardInput, ModelForwardOutput
from specplatform.schedulers import PreferredBatchAssignmentPolicy, RoundRobinRequestScheduler, SchedulerResources


class TinyRunner(CausalLMRunner):
    max_len = 32

    def __init__(self, runner_id: str) -> None:
        self.runner_id = runner_id

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        del prefix_ids
        logits = [-1.0] * 8
        logits[1] = 1.0
        return logits


class SessionStub:
    def __init__(self, request_id: str, remaining_tokens: int, step_idx: int = 0) -> None:
        self.request_id = request_id
        self.remaining_tokens = remaining_tokens
        self.step_idx = step_idx
        self.prompt_ids = [1]
        self.generated_ids: list[int] = []


class WorkerRegistrySchedulerTest(unittest.TestCase):
    def test_registry_loads_independent_worker_configs_and_filters_runner_type(self) -> None:
        calls: list[dict[str, Any]] = []

        def loader(model_path: str, **kwargs: Any) -> TinyRunner:
            calls.append({"model_path": model_path, **kwargs})
            return TinyRunner(runner_id=str(kwargs["runner_id"]))

        configs = [
            DraftWorkerConfig.from_config(
                {
                    "worker_id": "edge-fast",
                    "model_path": "/models/draft-a",
                    "device": "cuda:0",
                    "backend": "hf_cached",
                    "draft_type": "tree",
                    "speed_profile": {"name": "fast", "relative_speed": 2.0, "quality": 0.9},
                },
                index=0,
                defaults={},
            ),
            DraftWorkerConfig.from_config(
                {
                    "worker_id": "edge-small",
                    "model_path": "/models/draft-b",
                    "device": "cuda:1",
                    "backend": "hf_eager",
                    "draft_type": "greedy",
                },
                index=1,
                defaults={},
            ),
        ]

        registry = DraftWorkerRegistry.from_configs(configs, loader=loader)

        self.assertEqual([call["model_path"] for call in calls], ["/models/draft-a", "/models/draft-b"])
        self.assertEqual(calls[0]["device"], "cuda:0")
        self.assertEqual(calls[1]["backend"], "hf_eager")
        self.assertEqual(list(registry.runners_for("tree")), ["edge-fast"])
        self.assertEqual(list(registry.runners_for("greedy")), ["edge-small"])
        metadata = registry.to_metadata()
        self.assertEqual(metadata["draft_worker_count"], 2)
        self.assertEqual(metadata["draft_workers"][0]["speed_profile"]["relative_speed"], 2.0)
        self.assertEqual(metadata["draft_workers"][0]["speed_profile"]["quality"], 0.9)

    def test_round_robin_scheduler_uses_request_pool_length_and_batch_policies(self) -> None:
        scheduler = RoundRobinRequestScheduler(
            default_budget=DraftBudget(max_tokens=4, max_branches=2),
            batch_assignment_policy=PreferredBatchAssignmentPolicy(batch_size=2),
        )
        sessions = [
            SessionStub("r1", remaining_tokens=3),
            SessionStub("r2", remaining_tokens=1),
            SessionStub("r3", remaining_tokens=5),
        ]
        hints = PlanHints(
            draft_lengths={"r1": 2, "r2": 8},
            candidate_worker_preferences={"r1": ["w1", "w0"]},
            preferred_batches=[["r3", "r1"], ["stale"]],
        )

        plan = scheduler.plan(
            sessions,
            resources=SchedulerResources(draft_worker_ids=["w0", "w1"]),
            hints=hints,
            context=RuntimeContext(),
        )

        self.assertEqual([job.worker_id for job in plan.draft_jobs], ["w1", "w0", "w0", "w1"])
        self.assertEqual([job.budget.max_tokens for job in plan.draft_jobs], [2, 2, 1, 4])
        self.assertEqual([job.metadata["candidate_count"] for job in plan.draft_jobs[:2]], [2, 2])
        self.assertEqual([batch.request_ids for batch in plan.verify_batches], [["r3", "r1"]])
        self.assertEqual(plan.metadata["request_pool_size"], 3)
        self.assertEqual(plan.metadata["batch_assignment_policy"], "PreferredBatchAssignmentPolicy")

    def test_scheduler_honors_candidate_specific_draft_lengths(self) -> None:
        scheduler = RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=4, max_branches=2))
        sessions = [SessionStub("r1", remaining_tokens=8)]
        hints = PlanHints(
            draft_lengths={"r1": 4},
            candidate_worker_preferences={"r1": ["w0", "w1"]},
            candidate_draft_lengths={"r1": {"w0": 2, "w1": 6}},
        )

        plan = scheduler.plan(
            sessions,
            resources=SchedulerResources(draft_worker_ids=["w0", "w1"]),
            hints=hints,
            context=RuntimeContext(),
        )

        self.assertEqual([job.worker_id for job in plan.draft_jobs], ["w0", "w1"])
        self.assertEqual([job.budget.max_tokens for job in plan.draft_jobs], [2, 6])

    def test_scheduler_attaches_generic_preferred_batch_metadata(self) -> None:
        scheduler = RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=4, max_branches=1))
        sessions = [SessionStub("r1", remaining_tokens=8), SessionStub("r2", remaining_tokens=8)]
        hints = PlanHints(
            preferred_batches=[["r1", "r2"]],
            metadata={
                "preferred_batch_metadata": [
                    {
                        "stage_index": 0,
                        "planned_batch_count": 1,
                        "max_draft_len": 4,
                        "max_prefix_len": 16,
                        "estimated_verify_ms": 12.5,
                        "estimated_memory_bytes": 1024.0,
                    }
                ]
            },
        )

        plan = scheduler.plan(
            sessions,
            resources=SchedulerResources(draft_worker_ids=["w0"]),
            hints=hints,
            context=RuntimeContext(),
        )

        self.assertEqual(plan.verify_batches[0].metadata["stage_index"], 0)
        self.assertEqual(plan.verify_batches[0].metadata["planned_batch_count"], 1)
        self.assertEqual(plan.verify_batches[0].metadata["max_draft_len"], 4)

    def test_method_planning_policies_emit_shared_scheduler_hints(self) -> None:
        sessions = [
            SessionStub("r1", remaining_tokens=8),
            SessionStub("r2", remaining_tokens=8),
            SessionStub("r3", remaining_tokens=8),
        ]
        resources = {
            "draft_worker_ids": ["slow", "fast"],
            "draft_worker_metadata": {
                "slow": {"speed_profile": {"relative_speed": 1.0}},
                "fast": {"speed_profile": {"relative_speed": 2.0}},
            },
        }

        dip_hints = DiPSDPlanningPolicy(
            initial_draft_length=3,
            max_draft_length=8,
            max_batch_count=2,
        ).plan(
            sessions,
            resources=resources,
            history={},
            context=RuntimeContext(),
        )
        self.assertEqual(len(dip_hints.preferred_batches), 2)
        self.assertEqual(set().union(*[set(batch) for batch in dip_hints.preferred_batches]), {"r1", "r2", "r3"})
        self.assertEqual(dip_hints.worker_preferences, {"r1": "fast", "r2": "slow", "r3": "fast"})
        self.assertTrue(dip_hints.metadata["speed_aware_worker_assignment"])
        self.assertEqual(dip_hints.metadata["worker_assignment_order"], ["fast", "slow"])
        self.assertTrue(all(1 <= depth <= 8 for depth in dip_hints.draft_lengths.values()))
        self.assertEqual(dip_hints.metadata["method_family"], "dip_sd")
        self.assertEqual(
            dip_hints.metadata["assignment_objective"],
            "maximize_expected_accepted_tokens_per_pipeline_span",
        )
        self.assertTrue(dip_hints.metadata["solver_active"])
        self.assertTrue(dip_hints.metadata["joint_batch_assignment"])
        self.assertTrue(dip_hints.metadata["joint_draft_length"])
        self.assertTrue(dip_hints.metadata["phase_level_pipeline_required"])
        self.assertIn("dip_sd_solution", dip_hints.metadata)
        self.assertEqual(len(dip_hints.metadata["preferred_batch_metadata"]), 2)

        round_robin_hints = DiPSDPlanningPolicy(
            initial_draft_length=3,
            max_draft_length=8,
            max_batch_count=2,
        ).plan(
            sessions,
            resources=resources,
            history={},
            context=RuntimeContext(method_config={"dip_sd_speed_aware_worker_assignment": False}),
        )
        self.assertEqual(round_robin_hints.worker_preferences, {"r1": "slow", "r2": "fast", "r3": "slow"})
        self.assertFalse(round_robin_hints.metadata["speed_aware_worker_assignment"])
        self.assertEqual(round_robin_hints.metadata["worker_assignment_order"], ["slow", "fast"])

        sled_policy = SLEDPlanningPolicy(
            max_speculation_tokens=2,
            max_depth=8,
            target_batch_size=3,
            confidence_threshold=0.6,
        )
        sled_hints = sled_policy.plan(
            sessions,
            resources=resources,
            history={},
            context=RuntimeContext(),
        )
        self.assertEqual(sled_hints.preferred_batches, [["r2", "r1", "r3"]])
        self.assertEqual(sled_hints.worker_preferences, {"r1": "slow", "r2": "fast", "r3": "fast"})
        self.assertEqual(sled_hints.candidate_worker_preferences, {})
        self.assertEqual(sled_hints.draft_lengths, {"r1": 2, "r2": 2, "r3": 2})
        self.assertTrue(sled_hints.metadata["single_edge_device_per_request"])
        self.assertTrue(sled_hints.metadata["dynamic_drafting"])
        self.assertEqual(sled_hints.metadata["confidence_threshold"], 0.6)
        self.assertEqual(sled_hints.metadata["method_family"], "sled")
        self.assertEqual(
            sled_hints.metadata["assignment_objective"],
            "stable_edge_device_assignment_with_confidence_triggered_verification",
        )

    def test_dip_sd_solver_batch_count_scan_emits_trace(self) -> None:
        sessions = [SessionStub(f"r{index}", remaining_tokens=8) for index in range(4)]
        resources = {
            "draft_worker_ids": ["w0", "w1"],
            "draft_worker_metadata": {
                "w0": {"speed_profile": {"relative_speed": 1.0}},
                "w1": {"speed_profile": {"relative_speed": 2.0}},
            },
        }

        hints = DiPSDPlanningPolicy(
            initial_draft_length=2,
            max_draft_length=6,
        ).plan(sessions, resources=resources, history={}, context=RuntimeContext())

        self.assertTrue(hints.metadata["solver_active"])
        self.assertGreater(len(hints.metadata["dip_sd_solution"]["scan_trace"]), 1)
        self.assertIn(hints.metadata["planned_batch_count"], {2, 3, 4})
        self.assertEqual(len(hints.preferred_batches), hints.metadata["planned_batch_count"])
        self.assertIn("throughput_tokens_per_ms", hints.metadata["dip_sd_solution"]["scan_trace"][0])

    def test_dip_sd_batch_objective_models_sequential_server_queue(self) -> None:
        ready_time_ms = {f"r{index}": 0.0 for index in range(4)}
        draft_lengths = {request_id: 2 for request_id in ready_time_ms}
        expected_accept_tokens = {request_id: 1.0 for request_id in ready_time_ms}
        timing = {
            "server_verify_ms": 100.0,
            "network_residual_ms": 0.0,
            "server_batch_per_request_ms": 0.0,
            "server_batch_per_token_ms": 0.0,
        }

        score_single, tail_single = _assignment_score_and_tail(
            ready_time_ms,
            draft_lengths,
            expected_accept_tokens,
            batch_size=1,
            timing=timing,
        )
        score_batch, tail_batch = _assignment_score_and_tail(
            ready_time_ms,
            draft_lengths,
            expected_accept_tokens,
            batch_size=4,
            timing=timing,
        )

        self.assertEqual(tail_single, 400.0)
        self.assertEqual(tail_batch, 100.0)
        self.assertGreater(score_single, score_batch)

    def test_sled_keeps_stable_single_edge_worker_per_request(self) -> None:
        sessions = [SessionStub("r1", remaining_tokens=8), SessionStub("r2", remaining_tokens=8)]
        resources = {
            "draft_worker_ids": ["fast", "accurate"],
            "draft_worker_metadata": {
                "fast": {
                    "speed_profile": {"relative_speed": 2.0},
                },
                "accurate": {
                    "speed_profile": {"relative_speed": 1.0},
                },
            },
        }
        policy = SLEDPlanningPolicy(max_speculation_tokens=3, target_batch_size=2)

        first = policy.plan(sessions, resources=resources, history={}, context=RuntimeContext())
        second = policy.plan(sessions, resources=resources, history={}, context=RuntimeContext())

        self.assertEqual(first.worker_preferences, second.worker_preferences)
        self.assertEqual(first.candidate_worker_preferences, {})
        self.assertEqual(second.candidate_worker_preferences, {})
        self.assertEqual(first.draft_lengths, {"r1": 3, "r2": 3})
        self.assertTrue(all(item["reused_assignment"] for item in second.metadata["assignment_trace"]))

    def test_sled_does_not_import_dip_sd_package(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src" / "specplatform" / "methods" / "sled.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("specplatform.methods.dip_sd", text)

    def test_dip_sd_uses_positive_paper_draft_lengths_without_zero_fallback(self) -> None:
        sessions = [SessionStub("r1", remaining_tokens=8)]
        resources = {
            "draft_worker_ids": ["slow"],
            "draft_worker_metadata": {
                "slow": {"speed_profile": {"relative_speed": 1.0, "quality": 0.9}},
            },
        }
        context = RuntimeContext(method_config={"dip_sd_max_draft_length": 4})

        hints = DiPSDPlanningPolicy(initial_draft_length=2).plan(
            sessions,
            resources=resources,
            history={},
            context=context,
        )
        self.assertGreaterEqual(hints.draft_lengths["r1"], 1)
        self.assertLessEqual(hints.draft_lengths["r1"], 4)
        self.assertTrue(hints.metadata["solver_active"])


if __name__ == "__main__":
    unittest.main()
