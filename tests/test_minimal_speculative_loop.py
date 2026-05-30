"""最小真实 speculative decoding 闭环测试。"""

import inspect
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from specplatform.core import CandidateProposal, DraftBudget, RuntimeContext, VerificationResult
from specplatform.draft import GreedyDraftRunner
from specplatform.methods import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy
from specplatform.model import CausalLMRunner, ModelForwardInput, ModelForwardOutput
from specplatform.runtime import GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import HttpLinearVerifierClient, LinearVerifier
from specplatform.verification.schema import LinearVerifyResponse


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
        self.assertEqual(session.generated_ids, [])

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
        self.assertEqual(proposal.tokens, [2, 3])

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
                        accepted_prefix_len=1,
                        verified_tokens=[2, 4],
                        bonus_token=4,
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
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

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
        self.assertEqual(result.accepted_prefix_len, 1)
        self.assertEqual(result.verified_tokens, [2, 4])
        self.assertEqual(result.bonus_token, 4)


if __name__ == "__main__":
    unittest.main()
