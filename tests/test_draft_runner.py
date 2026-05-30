"""fake draft runner 的最小多 token 生成测试。"""

import unittest

from specplatform.core import DraftBudget
from specplatform.draft import FakeDraftRunner
from specplatform.methods.fake_linear import FakeLinearCandidateStrategy
from specplatform.runtime import GenerationSession


class FakeDraftRunnerTest(unittest.TestCase):
    """验证 Step 2 的 draft runner 多 token 生成能力。"""

    def test_generate_tokens_uses_prefix_and_max_tokens(self) -> None:
        """draft runner 应基于 prefix 连续生成多个候选 token。"""
        runner = FakeDraftRunner(runner_id="draft0", vocab_size=16)

        generation = runner.generate_tokens(
            prefix_ids=[1, 2],
            max_tokens=3,
            request_id="req0",
            metadata={"method": "fake_linear"},
        )

        self.assertEqual(generation.tokens, [4, 7, 11])
        self.assertEqual(generation.forward_timing_ms, [0.1, 0.1, 0.1])
        self.assertEqual(len(generation.forward_intervals_ns), 3)
        self.assertEqual(generation.metadata["prefix_ids"], [1, 2])
        self.assertEqual(generation.metadata["draft_runner_id"], "draft0")

    def test_candidate_strategy_wraps_draft_generation_as_candidate_proposal(self) -> None:
        """method strategy 只把 draft tokens 包装成 CandidateProposal。"""
        session = GenerationSession(
            request_id="req0",
            prompt_ids=[1, 2],
            max_new_tokens=3,
            max_len=16,
        )
        runner = FakeDraftRunner(runner_id="draft0", vocab_size=16)
        strategy = FakeLinearCandidateStrategy()

        proposal = strategy.propose(
            session,
            runner,
            DraftBudget(max_tokens=3),
            context=None,
        )

        self.assertEqual(proposal.tokens, [4, 7, 11])
        self.assertEqual(proposal.draft_length, 3)
        self.assertEqual(proposal.request_id, "req0")
        self.assertEqual(proposal.worker_id, "draft0")
        self.assertEqual(proposal.metadata["prefix_ids"], [1, 2])
        self.assertEqual(proposal.metadata["draft_budget_max_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
