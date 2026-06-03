"""最小真实 speculative decoding 闭环测试。"""

import inspect
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from specplatform.core import CandidateProposal, DraftBudget, PlanHints, RuntimeContext, VerificationResult
from specplatform.draft import GreedyDraftRunner
from specplatform.methods import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy
from specplatform.model import (
    CausalLMRunner,
    LinearForwardInput,
    LinearForwardOutput,
    ModelBackendCapabilities,
    ModelForwardInput,
    ModelForwardOutput,
)
from specplatform.runtime import GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import HttpLinearVerifierClient, HttpLinearVerifierPoolClient, LinearVerifier
from specplatform.verification.base import VerifierBackend
from specplatform.verification.schema import (
    BatchVerifyResponse,
    BatchVerifyResultItem,
    LinearVerifyRequest,
    LinearVerifyResponse,
)


class ScriptedCausalLMRunner(CausalLMRunner):
    """测试内部脚本化 causal LM。

    它只模拟真实 CausalLMRunner 接口：给定完整 prefix，返回预设的 greedy next token。
    生产代码不会依赖这个类。
    """

    runner_id = "scripted"
    max_len = 64

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        self.next_tokens_by_prefix = dict(next_tokens_by_prefix)
        self.seen_prefixes: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        """把空格分隔数字转成 token ids。"""
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 转回空格分隔文本。"""
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """保留 ModelRunner 抽象契约。"""
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """返回 one-hot 风格 logits，让 greedy_next_token 选中预设 token。"""
        self.seen_prefixes.append(list(prefix_ids))
        token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
        logits = [-10.0] * 16
        logits[token_id] = 10.0
        return logits


class RecordingLinearVerifier(LinearVerifier):
    """记录被验证 proposal 的 verifier，用来证明 runtime 真正调用了 verifier。"""

    def __init__(self, model: CausalLMRunner) -> None:
        super().__init__(model=model)
        self.verified_proposal_ids: list[str] = []

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        self.verified_proposal_ids.append(proposal.proposal_id)
        return super().verify_proposal(proposal, context)


class BatchedScriptedCausalLMRunner(ScriptedCausalLMRunner):
    """Scripted model that exposes a linear single-pass batch boundary."""

    backend_name = "scripted_batched"

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        super().__init__(next_tokens_by_prefix)
        self.batch_call_sizes: list[int] = []

    def backend_capabilities(self) -> ModelBackendCapabilities:
        return ModelBackendCapabilities(
            backend_name=self.backend_name,
            supports_batched_next_token=True,
            supports_linear_verify_batch=True,
        )

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        self.batch_call_sizes.append(len(requests))
        outputs: list[LinearForwardOutput] = []
        for request in requests:
            working_prefix = list(request.prefix_ids)
            target_tokens: list[int] = []
            for draft_token in request.draft_tokens:
                target_token = self.greedy_next_token(working_prefix)
                target_tokens.append(target_token)
                working_prefix.append(int(draft_token))
            bonus_token = self.greedy_next_token(working_prefix) if request.allow_bonus else None
            outputs.append(
                LinearForwardOutput(
                    draft_target_tokens=target_tokens,
                    bonus_token=bonus_token,
                    metadata={
                        "linear_forward_batch_kind": "linear_single_pass_batch",
                        "target_forward_call_count": 1,
                    },
                )
            )
        return outputs

    def next_token_logits_batch(self, prefix_ids_batch: list[list[int]]) -> list[list[float]]:
        self.batch_call_sizes.append(len(prefix_ids_batch))
        return [self.next_token_logits(prefix_ids) for prefix_ids in prefix_ids_batch]


class TreeAttentionBatchedScriptedCausalLMRunner(BatchedScriptedCausalLMRunner):
    """Scripted model exposing the qwen3 graph linear-tree attention kind."""

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        self.batch_call_sizes.append(len(requests))
        outputs = super().linear_verify_batch(requests)
        for output in outputs:
            output.metadata.update(
                {
                    "linear_forward_batch_kind": "linear_tree_attention_batch_qwen3_graph",
                    "linear_forward_kind": "linear_tree_attention_qwen3_graph",
                    "shared_forward_id": "tree-attn-shared",
                }
            )
        return outputs


class LengthStrictBatchedScriptedCausalLMRunner(BatchedScriptedCausalLMRunner):
    """Batched scripted model that rejects mixed prefix lengths."""

    def next_token_logits_batch(self, prefix_ids_batch: list[list[int]]) -> list[list[float]]:
        prefix_lens = {len(prefix_ids) for prefix_ids in prefix_ids_batch}
        if len(prefix_lens) != 1:
            raise AssertionError("mixed prefix lengths must be bucketed before batch forward")
        return super().next_token_logits_batch(prefix_ids_batch)


class DroppingVerifier(VerifierBackend):
    """故意漏掉 verifier result 的测试后端，用来验证 runtime 会 fail fast。"""

    backend_name = "dropping"

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        raise AssertionError("DroppingVerifier should be exercised through verify_batch.")

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        return []


class BonusVerifier(VerifierBackend):
    """Accepts every proposal, used by runtime scheduling tests."""

    backend_name = "bonus"

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=len(proposal.tokens),
            verified_tokens=list(proposal.tokens),
            timing={},
        )

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        return [self.verify_proposal(proposal, context) for proposal in proposals]


class SleepyCandidateStrategy:
    """Synthetic candidate strategy that exposes draft-job overlap."""

    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        del budget, context
        time.sleep(self.sleep_seconds)
        runner_id = str(getattr(draft_runner, "runner_id", "draft"))
        return CandidateProposal(
            proposal_id=f"sleepy:{session.request_id}:{runner_id}",
            request_id=session.request_id,
            worker_id=runner_id,
            shape="linear",
            tokens=[2],
            draft_length=1,
            metadata={"runner_id": runner_id},
        )


class DummyDraftRunner:
    def __init__(self, runner_id: str) -> None:
        self.runner_id = runner_id
        self.metadata = {"runner_id": runner_id}


class MultiCandidatePlanningPolicy:
    def plan(self, active_sessions: list[Any], resources: Any, history: Any, context: RuntimeContext) -> PlanHints:
        del resources, history, context
        return PlanHints(
            candidate_worker_preferences={
                session.request_id: ["draft-worker-bad", "draft-worker-good"]
                for session in active_sessions
            },
            metadata={"assignment_objective": "test_multi_candidate_objective"},
        )


class MinimalSpeculativeLoopTest(unittest.TestCase):
    """覆盖从 draft 到 append 的最小闭环。"""

    def test_linear_candidate_strategy_wraps_draft_generation(self) -> None:
        """method 只把 draft tokens 包装成 CandidateProposal，不做验证和写回。"""
        draft_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        draft_runner = GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=4,
            max_len=16,
        )
        strategy = LinearCandidateStrategy()

        proposal = strategy.propose(
            session,
            draft_runner,
            DraftBudget(max_tokens=2),
            RuntimeContext(),
        )

        self.assertEqual(proposal.shape, "linear")
        self.assertEqual(proposal.tokens, [2, 3])
        self.assertEqual(proposal.draft_length, 2)
        self.assertEqual(proposal.metadata["prefix_ids"], [1])
        self.assertTrue(proposal.metadata["allow_bonus"])
        self.assertEqual(session.generated_ids, [])

    def test_linear_candidate_strategy_disables_bonus_when_budget_fills_remaining_tokens(self) -> None:
        """draft 已经填满剩余空间时，proposal 应告诉 verifier 不要额外生成 bonus。"""
        draft_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        draft_runner = GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=2,
            max_len=8,
        )
        strategy = LinearCandidateStrategy()

        proposal = strategy.propose(
            session,
            draft_runner,
            DraftBudget(max_tokens=4),
            RuntimeContext(),
        )

        self.assertEqual(proposal.tokens, [2, 3])
        self.assertFalse(proposal.metadata["allow_bonus"])

    def test_linear_verifier_returns_match_prefix_and_bonus(self) -> None:
        """verifier 逐 token 比较 draft 和 target，只返回验证事实。"""
        target_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 4})
        verifier = LinearVerifier(model=target_model)
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2, 3],
            draft_length=2,
            metadata={"prefix_ids": [1]},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 1)
        self.assertEqual(result.verified_tokens, [2, 4])
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual([event["kind"] for event in result.timing["target_forward_events"]], ["linear_verify"])
        self.assertEqual(result.timing["target_forward_events"][0]["target_forward_call_count"], 2)
        self.assertEqual(proposal.tokens, [2, 3])

    def test_linear_verifier_skips_bonus_when_not_allowed(self) -> None:
        """allow_bonus=False 时，全匹配后 verifier 不应再多跑一次 target forward。"""
        target_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        verifier = LinearVerifier(model=target_model)
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2, 3],
            draft_length=2,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 2)
        self.assertEqual(result.verified_tokens, [2, 3])
        self.assertIsNone(result.bonus_token)
        self.assertEqual(len(result.timing["target_forward_events"]), 1)
        self.assertEqual(result.timing["target_forward_events"][0]["target_forward_call_count"], 2)
        self.assertEqual(target_model.seen_prefixes, [[1], [1, 2]])

    def test_linear_verifier_keeps_mismatch_token_when_bonus_not_allowed(self) -> None:
        """allow_bonus=False 只禁止全匹配后的额外 token，不禁止 mismatch 纠偏 token。"""
        target_model = ScriptedCausalLMRunner({(1,): 4})
        verifier = LinearVerifier(model=target_model)
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2, 3],
            draft_length=2,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 0)
        self.assertEqual(result.verified_tokens, [4])
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual([event["kind"] for event in result.timing["target_forward_events"]], ["linear_verify"])
        self.assertEqual(target_model.seen_prefixes, [[1]])

    def test_linear_verify_response_timing_round_trip_and_legacy_default(self) -> None:
        """response schema 应保留 timing，同时兼容旧响应没有 timing 的情况。"""
        response = LinearVerifyResponse(
            request_id="request-1",
            proposal_id="proposal-1",
            accepted_prefix_len=1,
            verified_tokens=[2],
            timing={"server_total_ms": 1.5, "target_forward_events": [{"kind": "draft"}]},
        )

        restored = LinearVerifyResponse.from_dict(response.to_dict())
        legacy = LinearVerifyResponse.from_dict(
            {
                "request_id": "request-1",
                "proposal_id": "proposal-1",
                "accepted_prefix_len": 1,
                "verified_tokens": [2],
                "bonus_token": None,
                "metadata": {},
            }
        )

        self.assertEqual(restored.timing["server_total_ms"], 1.5)
        self.assertEqual(restored.timing["target_forward_events"][0]["kind"], "draft")
        self.assertEqual(legacy.timing, {})

    def test_linear_verifier_batches_multi_request_single_pass(self) -> None:
        """DiP-SD/SLED 的 linear batch verify 应使用 single-pass batch forward。"""
        model = BatchedScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 4,
                (5,): 6,
                (5, 6): 7,
                (5, 6, 9): 8,
            }
        )
        verifier = LinearVerifier(model=model)

        responses = verifier.verify_requests_batch(
            [
                LinearVerifyRequest(
                    request_id="r1",
                    proposal_id="p1",
                    prefix_ids=[1],
                    draft_tokens=[2, 3],
                ),
                LinearVerifyRequest(
                    request_id="r2",
                    proposal_id="p2",
                    prefix_ids=[5],
                    draft_tokens=[6, 9],
                ),
            ],
            batch_id="batch-a",
        )

        self.assertEqual(model.batch_call_sizes, [2])
        self.assertEqual(responses[0].accepted_prefix_len, 2)
        self.assertEqual(responses[0].bonus_token, 4)
        self.assertEqual(responses[1].accepted_prefix_len, 1)
        self.assertEqual(responses[1].bonus_token, 7)
        self.assertEqual(
            responses[0].timing["linear_forward_batch_kinds"],
            ["linear_single_pass_batch"],
        )
        first_event = responses[0].timing["target_forward_events"][0]
        self.assertEqual(first_event["batch_size"], 2)
        self.assertEqual(first_event["draft_token_count"], 2)
        self.assertEqual(first_event["target_forward_call_count"], 1)
        self.assertEqual(first_event["linear_forward_batch_kind"], "linear_single_pass_batch")
        self.assertEqual(
            first_event["shared_batch_event_id"],
            responses[1].timing["target_forward_events"][0]["shared_batch_event_id"],
        )

    def test_linear_verifier_preserves_qwen3_graph_tree_attention_single_pass_kind(self) -> None:
        """qwen3 graph chain-tree linear verifier 应在 response 层保持 single-pass 语义。"""
        model = TreeAttentionBatchedScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 4,
                (5,): 6,
                (5, 6): 7,
                (5, 6, 7): 8,
            }
        )
        verifier = LinearVerifier(model=model)

        responses = verifier.verify_requests_batch(
            [
                LinearVerifyRequest(request_id="r1", proposal_id="p1", prefix_ids=[1], draft_tokens=[2, 3]),
                LinearVerifyRequest(request_id="r2", proposal_id="p2", prefix_ids=[5], draft_tokens=[6, 7]),
            ],
            batch_id="batch-tree-attn",
        )

        self.assertEqual(model.batch_call_sizes, [2, 2])
        self.assertEqual(responses[0].metadata["linear_forward_batch_kind"], "linear_tree_attention_batch_qwen3_graph")
        self.assertTrue(responses[0].metadata["single_pass_linear_verify"])
        first_event = responses[0].timing["target_forward_events"][0]
        second_event = responses[1].timing["target_forward_events"][0]
        self.assertEqual(first_event["linear_forward_batch_kind"], "linear_tree_attention_batch_qwen3_graph")
        self.assertEqual(first_event["shared_batch_event_id"], second_event["shared_batch_event_id"])
        self.assertEqual(first_event["target_forward_call_count"], 1)

    def test_linear_verifier_batches_variable_prefix_lengths(self) -> None:
        """single-pass batch forward 可以用 padding 处理变量长度 prefix。"""
        model = LengthStrictBatchedScriptedCausalLMRunner(
            {
                (1,): 2,
                (3, 4): 5,
                (6, 7): 8,
            }
        )
        verifier = LinearVerifier(model=model)

        responses = verifier.verify_requests_batch(
            [
                LinearVerifyRequest(
                    request_id="r1",
                    proposal_id="p1",
                    prefix_ids=[1],
                    draft_tokens=[2],
                    allow_bonus=False,
                ),
                LinearVerifyRequest(
                    request_id="r2",
                    proposal_id="p2",
                    prefix_ids=[3, 4],
                    draft_tokens=[5],
                    allow_bonus=False,
                ),
                LinearVerifyRequest(
                    request_id="r3",
                    proposal_id="p3",
                    prefix_ids=[6, 7],
                    draft_tokens=[8],
                    allow_bonus=False,
                ),
            ],
            batch_id="batch-varlen",
        )

        self.assertEqual(model.batch_call_sizes, [3])
        self.assertEqual([response.accepted_prefix_len for response in responses], [1, 1, 1])
        self.assertEqual(
            responses[0].timing["linear_forward_batch_kinds"],
            ["linear_single_pass_batch"],
        )
        self.assertEqual(responses[1].timing["linear_forward_batch_kinds"], ["linear_single_pass_batch"])

    def test_linear_verifier_attributes_shared_single_pass_batch(self) -> None:
        """同一 verify batch 的 single-pass target forward 应在 timing 中共享事件 id。"""
        model = LengthStrictBatchedScriptedCausalLMRunner(
            {
                (1, 2, 3): 4,
                (1, 2, 3, 9): 5,
                (1, 2, 3, 8): 6,
            }
        )
        verifier = LinearVerifier(model=model)

        responses = verifier.verify_requests_batch(
            [
                LinearVerifyRequest(
                    request_id="r1",
                    proposal_id="p1",
                    prefix_ids=[1, 2, 3],
                    draft_tokens=[9],
                    allow_bonus=False,
                ),
                LinearVerifyRequest(
                    request_id="r1",
                    proposal_id="p2",
                    prefix_ids=[1, 2, 3],
                    draft_tokens=[8],
                    allow_bonus=False,
                ),
            ],
            batch_id="batch-duplicates",
        )

        self.assertEqual(model.batch_call_sizes, [2])
        self.assertEqual([response.bonus_token for response in responses], [4, 4])
        event = responses[0].timing["target_forward_events"][0]
        other_event = responses[1].timing["target_forward_events"][0]
        self.assertEqual(event["linear_forward_batch_kind"], "linear_single_pass_batch")
        self.assertEqual(event["shared_batch_event_id"], other_event["shared_batch_event_id"])
        self.assertEqual(event["target_forward_call_count"], 1)
        self.assertEqual(
            responses[0].timing["target_forward_total_ms"],
            responses[1].timing["target_forward_total_ms"],
        )

    def test_greedy_prefix_acceptance_consumes_verification_result_only(self) -> None:
        """acceptance 根据 verifier result 切分 accepted/rejected/bonus。"""
        policy = GreedyPrefixAcceptancePolicy()
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2, 3, 5],
            draft_length=3,
        )
        verification = VerificationResult(
            request_id="request-1",
            proposal_id="proposal-1",
            shape="linear",
            accepted_prefix_len=2,
            verified_tokens=[2, 3, 4],
            bonus_token=4,
        )

        result = policy.accept(proposal, verification, RuntimeContext())

        self.assertEqual(result.accepted_tokens, [2, 3])
        self.assertEqual(result.rejected_tokens, [5])
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual(result.output_token_ids, [2, 3, 4])

    def test_generation_session_supports_multi_eos_and_length_limit(self) -> None:
        """session.append_tokens 支持多 EOS，并按 max_new_tokens 截断写回。"""
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=3,
            max_len=8,
            eos_token_ids=[8, 9],
        )

        emitted = session.append_tokens([2, 9, 3])

        self.assertEqual(emitted, [2, 9])
        self.assertEqual(session.generated_ids, [2, 9])
        self.assertTrue(session.is_finished)

    def test_runtime_runs_single_request_speculative_loop_to_eos(self) -> None:
        """runtime 串起 scheduler -> draft -> candidate -> verifier -> acceptance -> append。"""
        draft_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 4,
                (1, 2, 3): 6,
                (1, 2, 3, 6): 8,
            }
        )
        target_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 6,
                (1, 2, 3, 6): 9,
            }
        )
        verifier = RecordingLinearVerifier(model=target_model)
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=8,
            max_len=16,
            eos_token_ids=[9],
        )
        engine = RuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=2)),
            verifier=verifier,
        )

        result = engine.run(
            run_id="run-1",
            sessions=[session],
            draft_runners={"draft-worker-0": GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")},
            context=RuntimeContext(run_config={"eos_token_ids": [9], "method": "linear"}),
        )

        self.assertEqual(session.generated_ids, [2, 3, 6, 9])
        self.assertTrue(session.is_finished)
        self.assertGreaterEqual(len(verifier.verified_proposal_ids), 2)
        self.assertEqual(result.request_results[0].output_token_ids, [2, 3, 6, 9])
        self.assertEqual(result.request_results[0].stop_reason, "eos")
        draft_detail_events = [
            event
            for event in result.events.events
            if event.phase == "draft.token_forward" and event.span_kind == "detail"
        ]
        self.assertEqual(len(draft_detail_events), 4)

    def test_runtime_selects_best_multi_candidate_proposal_once(self) -> None:
        """同一 request 多个 draft 候选被验证后，只写回 acceptance 最好的一个。"""
        bad_draft = ScriptedCausalLMRunner({(1,): 5, (1, 5): 5})
        good_draft = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        target_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=2,
            max_len=8,
        )
        engine = RuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=2)),
            verifier=LinearVerifier(model=target_model),
            planning_policy=MultiCandidatePlanningPolicy(),
        )

        result = engine.run(
            run_id="run-multi-candidate",
            sessions=[session],
            draft_runners={
                "draft-worker-bad": GreedyDraftRunner(model=bad_draft, runner_id="draft-worker-bad"),
                "draft-worker-good": GreedyDraftRunner(model=good_draft, runner_id="draft-worker-good"),
            },
            context=RuntimeContext(run_config={"method": "linear"}),
        )

        self.assertEqual(session.generated_ids, [2, 3])
        self.assertEqual(len(result.request_results[0].proposals), 2)
        accept_events = [event for event in result.events.events if event.phase == "accept.apply"]
        self.assertEqual(sum(1 for event in accept_events if event.metadata.get("candidate_winner")), 1)
        self.assertTrue(
            next(event for event in accept_events if event.metadata.get("candidate_winner"))
            .proposal_id.endswith("draft-worker-good")
        )
        plan_event = next(event for event in result.events.events if event.phase == "scheduler.plan")
        self.assertEqual(
            plan_event.metadata["planning_hints"]["assignment_objective"],
            "test_multi_candidate_objective",
        )
        append_events = [event for event in result.events.events if event.phase == "session.append"]
        self.assertEqual(len(append_events), 1)

    def test_runtime_executes_independent_draft_workers_in_parallel(self) -> None:
        """Distinct draft runner objects should execute overlapping draft jobs."""
        sessions = [
            GenerationSession(request_id="r1", prompt_ids=[1], max_new_tokens=1, max_len=8),
            GenerationSession(request_id="r2", prompt_ids=[1], max_new_tokens=1, max_len=8),
        ]
        engine = RuntimeEngine(
            candidate_strategy=SleepyCandidateStrategy(sleep_seconds=0.08),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=BonusVerifier(),
        )

        result = engine.run(
            run_id="parallel-draft",
            sessions=sessions,
            draft_runners={
                "draft-worker-0": DummyDraftRunner("draft-worker-0"),
                "draft-worker-1": DummyDraftRunner("draft-worker-1"),
            },
            context=RuntimeContext(run_config={"method": "parallel-draft-test"}),
        )

        draft_events = [
            event
            for event in result.events.events
            if event.phase == "draft.generate" and event.span_kind == "leaf"
        ]
        self.assertEqual(len(draft_events), 2)
        union_ms = (max(event.end_ns for event in draft_events) - min(event.start_ns for event in draft_events)) / 1_000_000
        summed_ms = sum(event.measured_duration_ms for event in draft_events)
        self.assertLess(union_ms, summed_ms * 0.8)
        self.assertTrue(all(event.metadata["parallel_draft"] for event in draft_events))
        self.assertEqual([session.generated_ids for session in sessions], [[2], [2]])

    def test_runtime_rejects_missing_verification_result(self) -> None:
        """verifier 少返回结果时，runtime 应立即报错而不是让 request 反复进下一轮。"""
        draft_model = ScriptedCausalLMRunner({(1,): 2})
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=2,
            max_len=8,
        )
        engine = RuntimeEngine(
            candidate_strategy=LinearCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1)),
            verifier=DroppingVerifier(),
        )

        with self.assertRaisesRegex(ValueError, "missing"):
            engine.run(
                run_id="run-1",
                sessions=[session],
                draft_runners={"draft-worker-0": GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")},
            )

    def test_runtime_has_no_method_name_branch(self) -> None:
        """runtime 可以记录 method label，但不能按 method 名称写 if/elif 分支。"""
        source = inspect.getsource(RuntimeEngine.run)

        self.assertNotIn("if method", source)
        self.assertNotIn("elif method", source)
        self.assertNotIn("method ==", source)


class HttpLinearVerifierClientTest(unittest.TestCase):
    """验证 3090 HTTP client 和 /verify_linear JSON 契约一致。"""

    def test_http_client_posts_linear_verify_request(self) -> None:
        """client 应发送 prefix/draft/eos，并把响应还原成 VerificationResult。"""
        captured: dict[str, Any] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server 固定方法名
                content_length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["payload"] = json.loads(self.rfile.read(content_length).decode("utf-8"))
                body = json.dumps(
                    LinearVerifyResponse(
                        request_id="request-1",
                        proposal_id="proposal-1",
                        accepted_prefix_len=1,
                        verified_tokens=[2, 4],
                        bonus_token=4,
                        timing={"server_total_ms": 0.1, "target_forward_total_ms": 0.05},
                    ).to_dict()
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                """测试中关闭 http.server 默认日志。"""
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown_server, server, thread)

        client = HttpLinearVerifierClient(base_url=f"http://127.0.0.1:{server.server_port}")
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2, 3],
            draft_length=2,
            metadata={"prefix_ids": [1], "eos_token_ids": [9]},
        )

        result = client.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(captured["path"], "/verify_linear")
        self.assertEqual(captured["payload"]["prefix_ids"], [1])
        self.assertEqual(captured["payload"]["draft_tokens"], [2, 3])
        self.assertEqual(captured["payload"]["eos_token_ids"], [9])
        self.assertTrue(captured["payload"]["allow_bonus"])
        self.assertEqual(result.accepted_prefix_len, 1)
        self.assertEqual(result.verified_tokens, [2, 4])
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual(result.timing["response_timing"]["server_total_ms"], 0.1)
        self.assertIn("network_or_queue_residual_ms", result.timing)
        self.assertIn("verify.http_total", [event["phase"] for event in result.timing["client_events"]])

    def test_http_client_rejects_response_for_wrong_proposal(self) -> None:
        """HTTP response 没有 echo 当前 proposal_id 时，client 应拒绝继续 acceptance。"""

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server 固定方法名
                _ = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(
                    LinearVerifyResponse(
                        request_id="request-1",
                        proposal_id="other-proposal",
                        accepted_prefix_len=0,
                        verified_tokens=[],
                        bonus_token=None,
                    ).to_dict()
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                """测试中关闭 http.server 默认日志。"""
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown_server, server, thread)

        client = HttpLinearVerifierClient(base_url=f"http://127.0.0.1:{server.server_port}")
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-worker-0",
            shape="linear",
            tokens=[2],
            draft_length=1,
            metadata={"prefix_ids": [1]},
        )

        with self.assertRaisesRegex(ValueError, "proposal_id"):
            client.verify_proposal(proposal, RuntimeContext())

    def test_http_linear_pool_dispatches_whole_batches_round_robin(self) -> None:
        """Pool client should send each runtime verify batch to one target replica."""
        captured: list[tuple[str, str, int]] = []

        def make_handler(replica_id: str) -> type[BaseHTTPRequestHandler]:
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802 - http.server 固定方法名
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                    captured.append((replica_id, self.path, len(payload["items"])))
                    results = []
                    for item in payload["items"]:
                        request = LinearVerifyRequest.from_dict(dict(item["request"]))
                        results.append(
                            BatchVerifyResultItem(
                                kind="linear",
                                response=LinearVerifyResponse(
                                    request_id=request.request_id,
                                    proposal_id=request.proposal_id,
                                    accepted_prefix_len=0,
                                    verified_tokens=[9],
                                    bonus_token=9,
                                    timing={"server_batch_total_ms": 0.1, "batch_size": len(payload["items"])},
                                    metadata={"linear_forward_batch_kind": "linear_single_pass_qwen3_graph"},
                                ),
                            )
                        )
                    body = json.dumps(
                        BatchVerifyResponse(
                            batch_id=str(payload["batch_id"]),
                            results=results,
                            timing={"server_batch_total_ms": 0.1, "batch_size": len(payload["items"])},
                        ).to_dict()
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, format: str, *args: Any) -> None:
                    return None

            return Handler

        server_a = HTTPServer(("127.0.0.1", 0), make_handler("a100-0"))
        thread_a = threading.Thread(target=server_a.serve_forever, daemon=True)
        thread_a.start()
        self.addCleanup(_shutdown_server, server_a, thread_a)
        server_b = HTTPServer(("127.0.0.1", 0), make_handler("a100-1"))
        thread_b = threading.Thread(target=server_b.serve_forever, daemon=True)
        thread_b.start()
        self.addCleanup(_shutdown_server, server_b, thread_b)

        client = HttpLinearVerifierPoolClient(
            base_urls=[
                f"http://127.0.0.1:{server_a.server_port}",
                f"http://127.0.0.1:{server_b.server_port}",
            ]
        )
        proposals = [
            CandidateProposal(
                proposal_id=f"proposal-{index}",
                request_id=f"request-{index}",
                worker_id=f"edge-{index}",
                shape="linear",
                tokens=[2],
                draft_length=1,
                metadata={"prefix_ids": [1], "allow_bonus": True},
            )
            for index in range(4)
        ]

        first = client.verify_batch(proposals[:2], RuntimeContext())
        second = client.verify_batch(proposals[2:], RuntimeContext())

        self.assertEqual(captured, [("a100-0", "/verify_linear_batch", 2), ("a100-1", "/verify_linear_batch", 2)])
        self.assertEqual({result.metadata["target_pool_index"] for result in first}, {0})
        self.assertEqual({result.metadata["target_pool_index"] for result in second}, {1})
        self.assertEqual(first[0].metadata["backend_name"], "linear_http_pool")


def _shutdown_server(server: HTTPServer, thread: threading.Thread) -> None:
    """关闭测试 HTTP server，并等待线程退出，避免测试结束时打印线程噪声。"""
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
