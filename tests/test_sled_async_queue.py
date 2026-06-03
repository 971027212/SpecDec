import threading
import time
import unittest
from typing import Any

from specplatform.core import CandidateProposal, DraftBudget, RuntimeContext, VerificationResult
from specplatform.draft import DraftGeneration
from specplatform.methods import GreedyPrefixAcceptancePolicy, SLEDAsyncDraftPolicy, SLEDAsyncReconcilePolicy
from specplatform.runtime import AsyncPipelineRuntimeEngine, GenerationSession
from specplatform.schedulers import (
    PoissonArrivalConfig,
    RoundRobinRequestScheduler,
    StaticQueueBatchPlanner,
    VerificationArrival,
    generate_poisson_arrivals,
    summarize_queue_batches,
)


class SLEDAsyncQueueTest(unittest.TestCase):
    def test_sled_async_policy_continues_same_edge_stream_after_parent_tokens(self) -> None:
        class Runner:
            runner_id = "edge-0"
            metadata = {"runner_id": "edge-0"}

            def generate_tokens_until_confidence_drop(self, **kwargs: Any) -> DraftGeneration:
                self.kwargs = dict(kwargs)
                return DraftGeneration(
                    tokens=[5, 6],
                    metadata={
                        "runner_id": "edge-0",
                        "dynamic_drafting": True,
                        "dynamic_stop_reason": "max_tokens",
                    },
                )

        session = GenerationSession(request_id="r1", prompt_ids=[1, 2], max_new_tokens=4, max_len=16)
        proposal = CandidateProposal(
            proposal_id="p1",
            request_id="r1",
            worker_id="edge-0",
            shape="linear",
            tokens=[3, 4],
            draft_length=2,
            metadata={"prefix_ids": [1, 2], "runner_id": "edge-0"},
        )
        runner = Runner()

        proactive = SLEDAsyncDraftPolicy(default_max_tokens=3).propose_proactive(
            session,
            proposal,
            runner,
            RuntimeContext(method_config={"sled_confidence_threshold": 0.7}),
        )

        self.assertIsNotNone(proactive)
        assert proactive is not None
        self.assertEqual(runner.kwargs["prefix_ids"], [1, 2, 3, 4])
        self.assertEqual(proactive.tokens, [5, 6])
        self.assertEqual(proactive.worker_id, "edge-0")
        self.assertEqual(proactive.metadata["prefix_ids"], [1, 2, 3, 4])
        self.assertFalse(proactive.metadata["allow_bonus"])

    def test_async_runtime_records_timeout_and_fallback_release(self) -> None:
        class ImmediateCandidateStrategy:
            def propose(
                self,
                session: GenerationSession,
                draft_runner: Any,
                budget: DraftBudget,
                context: RuntimeContext,
            ) -> CandidateProposal:
                del draft_runner, budget, context
                return CandidateProposal(
                    proposal_id=f"proposal:{session.request_id}",
                    request_id=session.request_id,
                    worker_id="edge-0",
                    shape="linear",
                    tokens=[2],
                    draft_length=1,
                    metadata={"prefix_ids": list(session.prefix_ids), "allow_bonus": False},
                )

        class SlowVerifier:
            def verify_batch(
                self,
                proposals: list[CandidateProposal],
                context: RuntimeContext | None = None,
            ) -> list[VerificationResult]:
                del context
                time.sleep(0.03)
                return [
                    VerificationResult(
                        request_id=proposal.request_id,
                        proposal_id=proposal.proposal_id,
                        shape=proposal.shape,
                        accepted_prefix_len=0,
                        verified_tokens=[9],
                        bonus_token=9,
                    )
                    for proposal in proposals
                ]

        class NoopProactive:
            def propose_proactive(
                self,
                session: GenerationSession,
                proposal: CandidateProposal,
                draft_runner: Any,
                context: RuntimeContext,
            ) -> None:
                del session, proposal, draft_runner, context
                return None

        class DummyRunner:
            metadata = {"runner_id": "edge-0"}

        session = GenerationSession(request_id="r1", prompt_ids=[1], max_new_tokens=1, max_len=8)
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=ImmediateCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=SlowVerifier(),
            proactive_policy=NoopProactive(),
            reconcile_policy=SLEDAsyncReconcilePolicy(),
        )

        result = engine.run(
            run_id="timeout-fallback",
            sessions=[session],
            draft_runners={"edge-0": DummyRunner()},
            context=RuntimeContext(
                run_config={"method": "sled_async"},
                method_config={
                    "sled_verify_timeout_ms": 1,
                    "sled_fallback_failure_threshold": 1,
                    "sled_enable_fallback_release": True,
                },
            ),
        )

        self.assertEqual(session.generated_ids, [2])
        phases = [event.phase for event in result.events.events]
        self.assertIn("verify.timeout", phases)
        self.assertIn("verify.fallback_release", phases)
        verify_batch = next(event for event in result.events.events if event.phase == "verify.batch_total")
        self.assertTrue(verify_batch.metadata["fallback_release"])
        self.assertEqual(verify_batch.metadata["fallback_released_token_count"], 1)

    def test_static_queue_batch_planner_batches_full_and_timeout_flushes(self) -> None:
        arrivals = [
            VerificationArrival("a0", "r0", "d0", arrival_ms=0.0, draft_length=2),
            VerificationArrival("a1", "r1", "d1", arrival_ms=1.0, draft_length=4),
            VerificationArrival("a2", "r2", "d2", arrival_ms=20.0, draft_length=3),
        ]

        batches = StaticQueueBatchPlanner(batch_size=2, max_wait_ms=5.0).plan(arrivals)

        self.assertEqual([batch.metadata["dispatch_reason"] for batch in batches], ["batch_full", "final_flush"])
        self.assertEqual([len(batch.arrivals) for batch in batches], [2, 1])
        self.assertEqual(batches[0].padded_draft_length, 4)
        self.assertEqual(batches[0].padding_token_count, 2)
        self.assertEqual(batches[0].queue_wait_ms_by_request, {"r0": 1.0, "r1": 0.0})
        self.assertEqual(batches[1].dispatch_ms, 25.0)

    def test_sled_async_runtime_uses_static_queue_to_replan_verify_batches(self) -> None:
        class ImmediateCandidateStrategy:
            def propose(
                self,
                session: GenerationSession,
                draft_runner: Any,
                budget: DraftBudget,
                context: RuntimeContext,
            ) -> CandidateProposal:
                del budget, context
                worker_id = str(getattr(draft_runner, "worker_id", "edge-0"))
                return CandidateProposal(
                    proposal_id=f"proposal:{session.request_id}",
                    request_id=session.request_id,
                    worker_id=worker_id,
                    shape="linear",
                    tokens=[2],
                    draft_length=1,
                    metadata={
                        "prefix_ids": list(session.prefix_ids),
                        "allow_bonus": False,
                        "runner_id": worker_id,
                    },
                )

        class RecordingVerifier:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def verify_batch(
                self,
                proposals: list[CandidateProposal],
                context: RuntimeContext | None = None,
            ) -> list[VerificationResult]:
                del context
                self.batch_sizes.append(len(proposals))
                return [
                    VerificationResult(
                        request_id=proposal.request_id,
                        proposal_id=proposal.proposal_id,
                        shape=proposal.shape,
                        accepted_prefix_len=0,
                        verified_tokens=[9],
                        bonus_token=9,
                        timing={"response_timing": {"batch_size": len(proposals)}},
                    )
                    for proposal in proposals
                ]

        class NoopProactive:
            def propose_proactive(
                self,
                session: GenerationSession,
                proposal: CandidateProposal,
                draft_runner: Any,
                context: RuntimeContext,
            ) -> None:
                del session, proposal, draft_runner, context
                return None

        class DummyRunner:
            def __init__(self, worker_id: str) -> None:
                self.worker_id = worker_id
                self.metadata = {"runner_id": worker_id}

        sessions = [
            GenerationSession(request_id=f"r{index}", prompt_ids=[1], max_new_tokens=1, max_len=8)
            for index in range(3)
        ]
        verifier = RecordingVerifier()
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=ImmediateCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=verifier,
            proactive_policy=NoopProactive(),
            reconcile_policy=SLEDAsyncReconcilePolicy(),
        )

        result = engine.run(
            run_id="static-queue-runtime",
            sessions=sessions,
            draft_runners={f"edge-{index}": DummyRunner(f"edge-{index}") for index in range(3)},
            context=RuntimeContext(
                run_config={"method": "sled_async"},
                method_config={
                    "sled_static_queue_enabled": True,
                    "sled_batch_size": 2,
                    "sled_queue_pad_to_max_length": True,
                },
            ),
        )

        self.assertEqual(verifier.batch_sizes, [2, 1])
        phases = [event.phase for event in result.events.events]
        self.assertIn("scheduler.sled_static_queue", phases)
        queue_event = next(event for event in result.events.events if event.phase == "scheduler.sled_static_queue")
        self.assertEqual(queue_event.metadata["batch_count"], 2)
        self.assertTrue(queue_event.metadata["sled_static_queue"])
        verify_batches = [event for event in result.events.events if event.phase == "verify.batch_total"]
        self.assertEqual([event.metadata["target_batch_size"] for event in verify_batches], [2, 2])
        self.assertTrue(all(event.metadata["sled_static_queue"] for event in verify_batches))

    def test_sled_async_runtime_submits_static_queue_batches_concurrently(self) -> None:
        class ImmediateCandidateStrategy:
            def propose(
                self,
                session: GenerationSession,
                draft_runner: Any,
                budget: DraftBudget,
                context: RuntimeContext,
            ) -> CandidateProposal:
                del budget, context
                worker_id = str(getattr(draft_runner, "worker_id", "edge-0"))
                return CandidateProposal(
                    proposal_id=f"proposal:{session.request_id}",
                    request_id=session.request_id,
                    worker_id=worker_id,
                    shape="linear",
                    tokens=[2],
                    draft_length=1,
                    metadata={"prefix_ids": list(session.prefix_ids), "allow_bonus": False},
                )

        class ConcurrentVerifier:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def verify_batch(
                self,
                proposals: list[CandidateProposal],
                context: RuntimeContext | None = None,
            ) -> list[VerificationResult]:
                del context
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.03)
                    return [
                        VerificationResult(
                            request_id=proposal.request_id,
                            proposal_id=proposal.proposal_id,
                            shape=proposal.shape,
                            accepted_prefix_len=0,
                            verified_tokens=[9],
                            bonus_token=9,
                        )
                        for proposal in proposals
                    ]
                finally:
                    with self.lock:
                        self.active -= 1

        class NoopProactive:
            def propose_proactive(
                self,
                session: GenerationSession,
                proposal: CandidateProposal,
                draft_runner: Any,
                context: RuntimeContext,
            ) -> None:
                del session, proposal, draft_runner, context
                return None

        class DummyRunner:
            def __init__(self, worker_id: str) -> None:
                self.worker_id = worker_id
                self.metadata = {"runner_id": worker_id}

        sessions = [
            GenerationSession(request_id=f"r{index}", prompt_ids=[1], max_new_tokens=1, max_len=8)
            for index in range(2)
        ]
        verifier = ConcurrentVerifier()
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=ImmediateCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=verifier,
            proactive_policy=NoopProactive(),
            reconcile_policy=SLEDAsyncReconcilePolicy(),
            max_verify_workers=2,
        )

        engine.run(
            run_id="static-queue-concurrent",
            sessions=sessions,
            draft_runners={"edge-0": DummyRunner("edge-0"), "edge-1": DummyRunner("edge-1")},
            context=RuntimeContext(
                run_config={"method": "sled_async"},
                method_config={
                    "sled_static_queue_enabled": True,
                    "sled_batch_size": 1,
                    "sled_queue_pad_to_max_length": True,
                },
            ),
        )

        self.assertEqual(verifier.max_active, 2)

    def test_poisson_arrivals_are_deterministic_and_summarized(self) -> None:
        config = PoissonArrivalConfig(
            device_count=3,
            arrival_rate_per_device_s=2.0,
            duration_s=1.0,
            seed=7,
            draft_length=2,
        )

        first = generate_poisson_arrivals(config)
        second = generate_poisson_arrivals(config)
        batches = StaticQueueBatchPlanner(batch_size=4, max_wait_ms=10.0).plan(first)
        summary = summarize_queue_batches(batches)

        self.assertEqual(first, second)
        self.assertEqual(summary["request_count"], len(first))
        self.assertGreaterEqual(summary["batch_count"], 0)
        self.assertIn("throughput_requests_per_s", summary)


if __name__ == "__main__":
    unittest.main()
