"""Distributed batch pipeline runtime tests."""

from concurrent.futures import Future
import unittest
from typing import Any

from specplatform.core import DraftBudget, DraftJob, PlanHints, RuntimeContext, VerificationResult
from specplatform.draft import GreedyDraftRunner
from specplatform.methods import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy, PlanningPolicy
from specplatform.model import CausalLMRunner, ModelForwardInput, ModelForwardOutput
from specplatform.runtime import DistributedBatchPipelineRuntimeEngine, GenerationSession
from specplatform.runtime.distributed_pipeline import _PrefetchedDraft, _prefetch_matches_job, _snapshot_session
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import VerifierBackend


class TinyRunner(CausalLMRunner):
    runner_id = "tiny"
    max_len = 32

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        del prefix_ids
        logits = [-10.0] * 4
        logits[1] = 10.0
        return logits


class AcceptAllVerifier(VerifierBackend):
    def verify_proposal(self, proposal: Any, context: RuntimeContext | None = None) -> VerificationResult:
        return self.verify_batch([proposal], context)[0]

    def verify_batch(self, proposals: list[Any], context: RuntimeContext | None = None) -> list[VerificationResult]:
        del context
        return [
            VerificationResult(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                shape=proposal.shape,
                accepted_prefix_len=len(proposal.tokens),
                verified_tokens=list(proposal.tokens),
                bonus_token=None,
                metadata={"backend_name": "accept_all"},
            )
            for proposal in proposals
        ]


class RejectWithBonusVerifier(VerifierBackend):
    def verify_proposal(self, proposal: Any, context: RuntimeContext | None = None) -> VerificationResult:
        return self.verify_batch([proposal], context)[0]

    def verify_batch(self, proposals: list[Any], context: RuntimeContext | None = None) -> list[VerificationResult]:
        del context
        return [
            VerificationResult(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                shape=proposal.shape,
                accepted_prefix_len=0,
                verified_tokens=[],
                bonus_token=1,
                metadata={"backend_name": "reject_with_bonus"},
            )
            for proposal in proposals
        ]


class FixedBatchPolicy(PlanningPolicy):
    def plan(self, active_sessions: list[Any], resources: Any, history: Any, context: RuntimeContext) -> PlanHints:
        del resources, history, context
        request_ids = [str(session.request_id) for session in active_sessions]
        return PlanHints(
            draft_lengths={request_id: 1 for request_id in request_ids},
            worker_preferences={request_id: "w0" for request_id in request_ids},
            preferred_batches=[request_ids[:2], request_ids[2:]],
            metadata={
                "preferred_batch_metadata": [
                    {"stage_index": 0, "planned_batch_count": 2, "max_draft_len": 1, "max_prefix_len": 1},
                    {"stage_index": 1, "planned_batch_count": 2, "max_draft_len": 1, "max_prefix_len": 1},
                ]
            },
        )


class ShrinkingDraftLengthPolicy(PlanningPolicy):
    def __init__(self) -> None:
        self.round_index = 0

    def plan(self, active_sessions: list[Any], resources: Any, history: Any, context: RuntimeContext) -> PlanHints:
        del resources, history, context
        request_ids = [str(session.request_id) for session in active_sessions]
        length = 3 if self.round_index == 0 else 1
        self.round_index += 1
        return PlanHints(
            draft_lengths={request_id: length for request_id in request_ids},
            worker_preferences={request_id: "w0" for request_id in request_ids},
            preferred_batches=[request_ids],
        )


class HistoryRecordingPolicy(PlanningPolicy):
    def __init__(self) -> None:
        self.histories: list[dict[str, Any]] = []

    def plan(self, active_sessions: list[Any], resources: Any, history: Any, context: RuntimeContext) -> PlanHints:
        del resources, context
        self.histories.append(
            {
                request_id: dict(stats)
                for request_id, stats in dict(history.get("dip_sd_acceptance_stats", {}) or {}).items()
            }
        )
        request_ids = [str(session.request_id) for session in active_sessions]
        return PlanHints(
            draft_lengths={request_id: 1 for request_id in request_ids},
            worker_preferences={request_id: "w0" for request_id in request_ids},
            preferred_batches=[request_ids],
        )


class PrefetchHistoryRecordingPolicy(PlanningPolicy):
    def __init__(self) -> None:
        self.prefetch_histories: list[dict[str, Any]] = []

    def plan(self, active_sessions: list[Any], resources: Any, history: Any, context: RuntimeContext) -> PlanHints:
        del resources, context
        self.prefetch_histories.append(
            {
                request_id: dict(prefetch)
                for request_id, prefetch in dict(history.get("dip_sd_prefetch_by_request", {}) or {}).items()
            }
        )
        request_ids = [str(session.request_id) for session in active_sessions]
        return PlanHints(
            draft_lengths={request_id: 1 for request_id in request_ids},
            worker_preferences={request_id: "w0" for request_id in request_ids},
            preferred_batches=[request_ids],
        )


class DistributedBatchPipelineRuntimeTest(unittest.TestCase):
    def test_runtime_verifies_planned_batches_in_pipeline_stages(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id=f"r{index}", prompt_ids=[0], max_new_tokens=1, max_len=16)
            for index in range(4)
        ]
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=AcceptAllVerifier(),
            planning_policy=FixedBatchPolicy(),
        )

        result = engine.run(
            run_id="pipe-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(run_config={"method": "dip_sd"}),
        )

        self.assertEqual(
            {item.request_id: item.output_token_ids for item in result.request_results},
            {"r0": [1], "r1": [1], "r2": [1], "r3": [1]},
        )
        phases = [event.phase for event in result.events.events]
        self.assertIn("planner.hints", phases)
        self.assertIn("pipeline.planner_wait", phases)
        self.assertIn("pipeline.stage", phases)
        self.assertEqual(phases.count("verify.batch_total"), 2)
        stage_events = [event for event in result.events.events if event.phase == "pipeline.stage"]
        self.assertEqual([event.metadata["stage_index"] for event in stage_events], [0, 1])

    def test_runtime_reuses_steady_state_prefetch_for_next_round(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id=f"r{index}", prompt_ids=[0], max_new_tokens=2, max_len=16)
            for index in range(3)
        ]
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=AcceptAllVerifier(),
            planning_policy=FixedBatchPolicy(),
        )

        result = engine.run(
            run_id="pipe-prefetch-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(
                run_config={"method": "dip_sd"},
                method_config={
                    "dip_sd_steady_state_enabled": True,
                    "dip_sd_prefetch_adaptive_length_enabled": False,
                },
            ),
        )

        phases = [event.phase for event in result.events.events]
        self.assertIn("pipeline.steady_state_prefetch_submit", phases)
        self.assertIn("pipeline.steady_state_prefetch_reuse", phases)
        prefetched_drafts = [
            event
            for event in result.events.events
            if event.phase == "draft.generate" and dict(event.metadata or {}).get("steady_state_prefetch")
        ]
        self.assertGreaterEqual(len(prefetched_drafts), 1)
        self.assertTrue(all(event.round == 1 for event in prefetched_drafts))
        self.assertEqual(
            {item.request_id: item.output_token_ids for item in result.request_results},
            {"r0": [1, 1], "r1": [1, 1], "r2": [1, 1]},
        )

    def test_runtime_passes_acceptance_history_to_next_planning_round(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=2, max_len=16)
        ]
        policy = HistoryRecordingPolicy()
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=AcceptAllVerifier(),
            planning_policy=policy,
        )

        engine.run(
            run_id="pipe-history-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(run_config={"method": "dip_sd"}),
        )

        self.assertEqual(policy.histories[0], {})
        second_round = policy.histories[1]["r0"]
        self.assertEqual(second_round["proposal_count"], 1)
        self.assertEqual(second_round["draft_token_count"], 1)
        self.assertEqual(second_round["accepted_draft_count"], 1)
        self.assertEqual(second_round["observed_acceptance"], 1.0)

    def test_runtime_passes_prefetch_history_to_next_planning_round(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=2, max_len=16)
        ]
        policy = PrefetchHistoryRecordingPolicy()
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=AcceptAllVerifier(),
            planning_policy=policy,
        )

        engine.run(
            run_id="pipe-prefetch-history-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(
                run_config={"method": "dip_sd"},
                method_config={
                    "dip_sd_steady_state_enabled": True,
                    "dip_sd_prefetch_adaptive_length_enabled": False,
                },
            ),
        )

        self.assertEqual(policy.prefetch_histories[0], {})
        second_round = policy.prefetch_histories[1]["r0"]
        self.assertEqual(second_round["worker_id"], "w0")
        self.assertEqual(second_round["budget_tokens"], 1)

    def test_steady_state_prefetch_requires_matching_session_prefix(self) -> None:
        session = GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=4, max_len=16)
        session.append_tokens([1])
        snapshot = _snapshot_session(session)
        job = DraftJob(request_id="r0", worker_id="w0", budget=DraftBudget(max_tokens=1))
        prefetch = _PrefetchedDraft(
            request_id="r0",
            worker_id="w0",
            budget_tokens=1,
            prefix_ids=tuple(snapshot.prefix_ids),
            step_idx=snapshot.step_idx,
            future=Future(),
            submitted_ns=0,
            source_round=0,
            source_stage=0,
            source_batch_id=None,
        )

        self.assertEqual(snapshot.prefix_ids, [0, 1])
        self.assertTrue(_prefetch_matches_job(prefetch, job, session))
        shorter_job = DraftJob(request_id="r0", worker_id="w0", budget=DraftBudget(max_tokens=1))
        longer_prefetch_job = _PrefetchedDraft(
            request_id="r0",
            worker_id="w0",
            budget_tokens=3,
            prefix_ids=tuple(snapshot.prefix_ids),
            step_idx=snapshot.step_idx,
            future=Future(),
            submitted_ns=0,
            source_round=0,
            source_stage=0,
            source_batch_id=None,
        )
        self.assertTrue(_prefetch_matches_job(longer_prefetch_job, shorter_job, session))

        session.append_tokens([2])

        self.assertEqual(snapshot.prefix_ids, [0, 1])
        self.assertFalse(_prefetch_matches_job(prefetch, job, session))

    def test_runtime_reuses_longer_prefetch_for_shorter_next_round_job(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=3, max_len=16)
        ]
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=RejectWithBonusVerifier(),
            planning_policy=ShrinkingDraftLengthPolicy(),
        )

        result = engine.run(
            run_id="pipe-prefetch-truncate-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(
                run_config={"method": "dip_sd"},
                method_config={"dip_sd_steady_state_enabled": True},
            ),
        )

        reused_drafts = [
            event
            for event in result.events.events
            if event.phase == "draft.generate"
            and event.round == 1
            and dict(event.metadata or {}).get("steady_state_prefetch")
        ]
        self.assertEqual({item.request_id: item.output_token_ids for item in result.request_results}, {"r0": [1, 1, 1]})
        self.assertTrue(reused_drafts)
        self.assertTrue(dict(reused_drafts[0].metadata or {}).get("prefetch_truncated"))
        self.assertEqual(dict(reused_drafts[0].metadata or {}).get("prefetch_original_draft_length"), 2)
        self.assertEqual(dict(reused_drafts[0].metadata or {}).get("prefetch_reused_budget_tokens"), 1)

    def test_runtime_adapts_prefetch_length_to_observed_output(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=3, max_len=16)
        ]
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=RejectWithBonusVerifier(),
            planning_policy=ShrinkingDraftLengthPolicy(),
        )

        result = engine.run(
            run_id="pipe-prefetch-adaptive-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(
                run_config={"method": "dip_sd"},
                method_config={
                    "dip_sd_steady_state_enabled": True,
                    "dip_sd_prefetch_adaptive_length_enabled": True,
                    "dip_sd_prefetch_acceptance_lookahead_tokens": 0,
                    "dip_sd_prefetch_use_source_budget_floor": False,
                },
            ),
        )

        submit_events = [
            event
            for event in result.events.events
            if event.phase == "pipeline.steady_state_prefetch_submit"
        ]
        self.assertTrue(submit_events)
        first_submit = dict(submit_events[0].metadata or {})
        self.assertEqual(first_submit["source_budget_max_tokens"], 3)
        self.assertEqual(first_submit["max_tokens"], 1)
        self.assertEqual(first_submit["prefetch_length_reason"], "acceptance_output_length")
        self.assertEqual(first_submit["prefetch_observed_output_tokens"], 1)
        self.assertEqual({item.request_id: item.output_token_ids for item in result.request_results}, {"r0": [1, 1, 1]})

    def test_runtime_prefetch_can_use_source_budget_floor_for_dip_sd(self) -> None:
        runner = GreedyDraftRunner(TinyRunner(), runner_id="w0")
        sessions = [
            GenerationSession(request_id="r0", prompt_ids=[0], max_new_tokens=3, max_len=16)
        ]
        engine = DistributedBatchPipelineRuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="pipe"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=RejectWithBonusVerifier(),
            planning_policy=ShrinkingDraftLengthPolicy(),
        )

        result = engine.run(
            run_id="pipe-prefetch-source-budget-floor-test",
            sessions=sessions,
            draft_runners={"w0": runner},
            context=RuntimeContext(
                run_config={"method": "dip_sd"},
                method_config={
                    "dip_sd_steady_state_enabled": True,
                    "dip_sd_prefetch_adaptive_length_enabled": True,
                    "dip_sd_prefetch_acceptance_lookahead_tokens": 0,
                    "dip_sd_prefetch_use_source_budget_floor": True,
                },
            ),
        )

        submit_events = [
            event
            for event in result.events.events
            if event.phase == "pipeline.steady_state_prefetch_submit"
        ]
        self.assertTrue(submit_events)
        first_submit = dict(submit_events[0].metadata or {})
        self.assertEqual(first_submit["source_budget_max_tokens"], 3)
        self.assertEqual(first_submit["max_tokens"], 2)
        self.assertTrue(first_submit["prefetch_use_source_budget_floor"])


if __name__ == "__main__":
    unittest.main()
