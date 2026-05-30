"""统一 runtime Phase 1 骨架测试。"""

import tempfile
import unittest
from pathlib import Path

from specplatform.core import DraftBudget, RuntimeContext
from specplatform.draft import FakeDraftRunner
from specplatform.methods.fake_linear import FakeLinearCandidateStrategy, LinearPrefixAcceptancePolicy
from specplatform.metrics import (
    write_phase_events_csv,
    write_phase_summary_csv,
    write_request_results_json,
)
from specplatform.runtime import GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import FakeProposalVerifier


class UnifiedRuntimePhase1Test(unittest.TestCase):
    """验证 fake method 能通过统一 runtime，并保持边界约束。"""

    def test_fake_linear_method_runs_through_unified_runtime(self) -> None:
        """fake linear method 应能走完 draft/verify/accept/append 流程。"""
        sessions = [
            GenerationSession(request_id="req0", prompt_ids=[1, 2], max_new_tokens=2, max_len=16),
            GenerationSession(request_id="req1", prompt_ids=[3, 4], max_new_tokens=2, max_len=16),
        ]
        engine = RuntimeEngine(
            candidate_strategy=FakeLinearCandidateStrategy(),
            acceptance_policy=LinearPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(
                default_budget=DraftBudget(max_tokens=1),
                batch_size=2,
            ),
            verifier=FakeProposalVerifier(vocab_size=16),
        )

        result = engine.run(
            run_id="unit",
            sessions=sessions,
            draft_runners={"draft0": FakeDraftRunner(runner_id="draft0", vocab_size=16)},
            context=RuntimeContext(run_config={"method": "fake_linear"}),
        )

        self.assertEqual(len(result.request_results), 2)
        self.assertTrue(all(item.output_token_ids for item in result.request_results))
        phases = {event.phase for event in result.events.events}
        self.assertIn("draft.generate", phases)
        self.assertIn("verify.batch_total", phases)
        self.assertIn("verify.request_attributed", phases)
        self.assertIn("request.generation_total", phases)

    def test_shared_batch_verify_is_recorded_once_and_attributed_to_requests(self) -> None:
        """共享 verify batch 只真实记录一次，再归因到各 request。"""
        sessions = [
            GenerationSession(request_id="req0", prompt_ids=[1, 2], max_new_tokens=1, max_len=16),
            GenerationSession(request_id="req1", prompt_ids=[3, 4], max_new_tokens=1, max_len=16),
        ]
        engine = RuntimeEngine(
            candidate_strategy=FakeLinearCandidateStrategy(),
            acceptance_policy=LinearPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(
                default_budget=DraftBudget(max_tokens=1),
                batch_size=2,
            ),
            verifier=FakeProposalVerifier(vocab_size=16),
        )

        result = engine.run(
            run_id="unit",
            sessions=sessions,
            draft_runners={"draft0": FakeDraftRunner(runner_id="draft0", vocab_size=16)},
            context=RuntimeContext(run_config={"method": "fake_linear"}),
        )

        shared = [
            event
            for event in result.events.events
            if event.phase == "verify.batch_total"
            and event.shared
            and event.attribution == "batch"
            and event.event_scope == "system"
            and event.span_kind == "leaf"
        ]
        attributed = [
            event
            for event in result.events.events
            if event.phase == "verify.request_attributed"
            and event.attribution == "request_average"
            and event.event_scope == "request"
            and event.span_kind == "attribution"
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(attributed), 2)
        self.assertEqual(
            sum(event.attributed_duration_ms for event in attributed),
            shared[0].measured_duration_ms,
        )
        for event in attributed:
            self.assertEqual(event.span_id, shared[0].span_id)
            self.assertEqual(event.parent_span_id, shared[0].span_id)

    def test_runtime_context_does_not_expose_execution_escape_hatches(self) -> None:
        """RuntimeContext 不应暴露 engine/verifier/recorder escape hatch。"""
        context = RuntimeContext()

        self.assertFalse(hasattr(context, "engine"))
        self.assertFalse(hasattr(context, "verifier"))
        self.assertFalse(hasattr(context, "metrics_recorder"))
        self.assertFalse(hasattr(context, "timing_recorder"))

    def test_runtime_engine_has_no_method_specific_branches(self) -> None:
        """RuntimeEngine 源码不应出现旧方法名分支。"""
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/specplatform/runtime/engine.py").read_text(encoding="utf-8")

        forbidden = [
            "".join(("spec", "edge")),
            "".join(("sl", "ed")),
            "_".join(("dip", "sd")),
        ]
        for marker in forbidden:
            self.assertNotIn(marker, source)

    def test_acceptance_policy_does_not_call_verifier(self) -> None:
        """AcceptancePolicy 不能直接调用 verifier。"""
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/specplatform/methods/fake_linear.py").read_text(encoding="utf-8")
        acceptance_source = source.split("class LinearPrefixAcceptancePolicy", maxsplit=1)[1]

        self.assertNotIn("verifier", acceptance_source)
        self.assertNotIn("verify_batch", acceptance_source)
        self.assertNotIn("verify_proposal", acceptance_source)

    def test_phase1_artifact_writers_create_required_outputs(self) -> None:
        """runtime 事件应能写出 Phase 1 所需 artifact。"""
        sessions = [
            GenerationSession(request_id="req0", prompt_ids=[1, 2], max_new_tokens=1, max_len=16),
        ]
        engine = RuntimeEngine(
            candidate_strategy=FakeLinearCandidateStrategy(),
            acceptance_policy=LinearPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=FakeProposalVerifier(vocab_size=16),
        )
        result = engine.run(
            run_id="unit",
            sessions=sessions,
            draft_runners={"draft0": FakeDraftRunner(runner_id="draft0", vocab_size=16)},
            context=RuntimeContext(run_config={"method": "fake_linear"}),
        )
        events = result.events.events
        self.assertTrue(all(event.method == "fake_linear" for event in events))
        self.assertTrue(all(event.plan_id for event in events))
        aggregate = [event for event in events if event.span_kind == "aggregate"]
        self.assertTrue(any(event.phase == "runtime.round_total" for event in aggregate))
        self.assertTrue(any(event.phase == "request.generation_total" for event in aggregate))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_phase_events_csv(result.events.events, root / "phase_events.csv")
            write_phase_summary_csv(result.events.events, root / "phase_summary.csv")
            write_request_results_json(result.request_results, root / "request_results.json")

            self.assertTrue((root / "phase_events.csv").exists())
            self.assertTrue((root / "phase_summary.csv").exists())
            self.assertTrue((root / "request_results.json").exists())


if __name__ == "__main__":
    unittest.main()
