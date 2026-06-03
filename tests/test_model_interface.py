"""Step 1：真实 CausalLMRunner 接口的最小测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specplatform.core import CandidateNode, CandidateTree
from specplatform.model import (
    CachedTransformersCausalLMRunner,
    CausalLMRunner,
    LinearForwardInput,
    ModelForwardInput,
    ModelForwardOutput,
    Qwen3GraphBackendUnavailable,
    Qwen3GraphCausalLMRunner,
    TorchKVCache,
    TreeForwardInput,
    TreeForwardNode,
    TransformersCausalLMRunner,
    load_causal_lm_runner,
    qwen3_graph_fallback_capabilities,
)
from specplatform.model.qwen3_graph import load_qwen3_graph_or_fallback


class ScriptedCausalLMRunner(CausalLMRunner):
    """测试内部使用的脚本化因果语言模型。

    它不是生产模型实现，只用于验证 CausalLMRunner 接口语义：
    给定 prefix，返回预先登记好的下一 token logits。
    """

    runner_id = "scripted"
    max_len = 16

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        self.next_tokens_by_prefix = dict(next_tokens_by_prefix)

    def encode(self, text: str) -> list[int]:
        """把空格分隔的数字文本编码成 token ids。"""
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 解码成空格分隔文本。"""
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """保留 ModelRunner 抽象契约；本测试直接覆盖 next_token_logits。"""
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """根据完整 prefix 返回一个 one-hot 风格的 logits 列表。"""
        token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
        logits = [-10.0] * 8
        logits[token_id] = 10.0
        return logits


class CausalLMRunnerInterfaceTest(unittest.TestCase):
    """验证真实模型接口的最小行为，不接真实权重。"""

    def test_encode_decode_round_trip_for_token_ids(self) -> None:
        """接口应能把文本和 token ids 互相转换。"""
        runner = ScriptedCausalLMRunner({})

        token_ids = runner.encode("1 2 3")

        self.assertEqual(token_ids, [1, 2, 3])
        self.assertEqual(runner.decode(token_ids), "1 2 3")

    def test_greedy_next_token_uses_next_token_logits(self) -> None:
        """greedy_next_token 应从 next_token_logits 里选择最大 logit。"""
        runner = ScriptedCausalLMRunner({(1, 2): 4})

        token_id = runner.greedy_next_token([1, 2])

        self.assertEqual(token_id, 4)

    def test_greedy_next_token_rejects_empty_prefix(self) -> None:
        """空 prefix 没有可验证的 causal context，应被拒绝。"""
        runner = ScriptedCausalLMRunner({})

        with self.assertRaises(ValueError):
            runner.greedy_next_token([])

    def test_next_token_topk_returns_deterministic_logprob_order(self) -> None:
        """默认 top-k 接口应可供 tree draft 使用，且同分按 token id 稳定。"""
        runner = ScriptedCausalLMRunner({(1,): 4})

        topk = runner.next_token_topk([1], 2)

        self.assertEqual([item.token_id for item in topk], [4, 0])
        self.assertEqual([item.rank for item in topk], [0, 1])
        self.assertGreater(topk[0].logprob, topk[1].logprob)

    def test_default_backend_capabilities_describe_eager_fallback_boundary(self) -> None:
        """普通 runner 默认是 eager/top-k 能力，不声称支持 tree attention/KV/CUDA graph。"""
        runner = ScriptedCausalLMRunner({(1,): 4})

        capabilities = runner.backend_capabilities().to_dict()

        self.assertEqual(capabilities["backend_name"], "eager")
        self.assertTrue(capabilities["supports_topk"])
        self.assertFalse(capabilities["supports_batched_next_token"])
        self.assertFalse(capabilities["supports_tree_attention"])
        self.assertFalse(capabilities["supports_kv_cache"])
        self.assertFalse(capabilities["supports_cuda_graph"])

    def test_loader_forwards_attention_backend_to_cached_runner(self) -> None:
        """target service 可通过 loader 启动 hf_cached single-pass verifier backend。"""
        sentinel = object()

        with patch.object(CachedTransformersCausalLMRunner, "from_pretrained", return_value=sentinel) as loader:
            runner = load_causal_lm_runner(
                "/models/qwen3",
                runner_id="target",
                backend="hf_cached",
                device="cuda",
                torch_dtype="fp16",
                attn_implementation="eager",
            )

        self.assertIs(runner, sentinel)
        loader.assert_called_once_with(
            "/models/qwen3",
            runner_id="target",
            device="cuda",
            torch_dtype="fp16",
            device_map=None,
            attn_implementation="eager",
            trust_remote_code=True,
        )

    def test_qwen3_graph_fallback_forwards_attention_backend_to_cached_runner(self) -> None:
        """qwen3_graph fallback 也应保留 attn_implementation，便于远端复现实验。"""
        fallback_runner = CachedTransformersCausalLMRunner(
            runner_id="target",
            tokenizer=object(),
            model=object(),
            max_len=16,
            device="cpu",
        )

        with (
            patch("specplatform.model.qwen3_graph._graph_unavailable_reason", return_value="not wired"),
            patch.object(CachedTransformersCausalLMRunner, "from_pretrained", return_value=fallback_runner) as loader,
        ):
            runner = load_qwen3_graph_or_fallback(
                "/models/qwen3",
                runner_id="target",
                device="cuda",
                torch_dtype="fp16",
                attn_implementation="eager",
            )

        self.assertEqual(runner.backend_name, "qwen3_graph_fallback_hf_cached")
        self.assertTrue(runner.backend_fallback)
        loader.assert_called_once_with(
            "/models/qwen3",
            runner_id="target",
            device="cuda",
            torch_dtype="fp16",
            device_map=None,
            attn_implementation="eager",
            trust_remote_code=True,
        )

    def test_torch_kv_cache_gather_reorders_and_clears_tail(self) -> None:
        """KV cache gather/reorder 应把 src 位置搬到 dest，并清空尾部。"""
        import torch

        cache = TorchKVCache(
            num_layers=1,
            batch_size=1,
            num_key_value_heads=1,
            max_len=4,
            head_dim=1,
            device="cpu",
            dtype=torch.float32,
        )
        cache.k_cache[0, 0, 0, :, 0] = torch.tensor([10.0, 11.0, 12.0, 13.0])
        cache.v_cache[0, 0, 0, :, 0] = torch.tensor([20.0, 21.0, 22.0, 23.0])

        cache.gather(0, [2, 0], [0, 1])

        self.assertEqual(cache.k_cache[0, 0, 0, :, 0].tolist(), [12.0, 10.0, 0.0, 0.0])
        self.assertEqual(cache.v_cache[0, 0, 0, :, 0].tolist(), [22.0, 20.0, 0.0, 0.0])

    def test_qwen3_graph_fallback_capabilities_are_explicit(self) -> None:
        """请求 graph 但 fallback 时，capabilities 必须诚实标记不能当 graph 性能。"""
        capabilities = qwen3_graph_fallback_capabilities("not wired").to_dict()

        self.assertEqual(capabilities["backend_name"], "qwen3_graph_fallback_hf_cached")
        self.assertTrue(capabilities["backend_fallback"])
        self.assertEqual(capabilities["fallback_reason"], "not wired")
        self.assertTrue(capabilities["supports_topk"])
        self.assertTrue(capabilities["supports_batched_next_token"])
        self.assertTrue(capabilities["supports_tree_attention"])
        self.assertTrue(capabilities["supports_kv_cache"])
        self.assertFalse(capabilities["supports_cuda_graph"])

    def test_qwen3_graph_without_fallback_rejects_non_qwen_path(self) -> None:
        """disable fallback 时，不支持的 graph backend 必须 fail fast。"""
        with self.assertRaises(Qwen3GraphBackendUnavailable):
            load_qwen3_graph_or_fallback(
                "/tmp/not-a-qwen-model",
                runner_id="draft",
                device="cpu",
                torch_dtype="fp32",
                allow_fallback=False,
            )

    def test_qwen3_graph_linear_verify_uses_tree_attention_batch_graph(self) -> None:
        """graph linear verifier 应把 linear draft 映射到 chain-tree batch graph。"""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls: list[dict[str, object]] = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                tokens = [int(value) for value in input_ids[0].tolist()]
                self.prefill_calls.append(
                    {
                        "input_ids": tuple(tokens),
                        "position_ids": tuple(int(value) for value in position_ids[0].tolist()),
                        "batch_idx": int(batch_idx),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                    }
                )
                logits = torch.full((1, len(tokens), 16), -10.0, dtype=torch.float32)
                logits[0, -1, (tokens[-1] + 1) % 16] = 10.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                self.forward_calls.append(
                    {
                        "input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids),
                        "position_ids": tuple(tuple(int(value) for value in row.tolist()) for row in position_ids),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                    }
                )
                logits = torch.full((2, 4, 16), -10.0, dtype=torch.float32)
                for row_index, row in enumerate(input_ids):
                    for node_index, token_id in enumerate(row.tolist()):
                        logits[row_index, node_index, (int(token_id) + 1) % 16] = 10.0
                return logits

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(
                encode=lambda text, add_special_tokens=False: [int(part) for part in text.split()],
                decode=lambda token_ids, skip_special_tokens=False: " ".join(str(token_id) for token_id in token_ids),
            ),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )

        outputs = runner.linear_verify_batch(
            [
                LinearForwardInput(prefix_ids=[1, 2, 3], draft_tokens=[4, 5], allow_bonus=True),
                LinearForwardInput(prefix_ids=[4, 5], draft_tokens=[6, 8], allow_bonus=True),
            ]
        )

        self.assertEqual(batch_engine.reset_calls, 1)
        self.assertEqual(len(batch_engine.prefill_calls), 2)
        self.assertEqual(len(batch_engine.forward_calls), 1)
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][0], (4, 5, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][1], (6, 8, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["position_ids"][0][:2], (3, 4))
        self.assertEqual(batch_engine.forward_calls[0]["position_ids"][1][:2], (2, 3))
        self.assertEqual(outputs[0].draft_target_tokens, [4, 5])
        self.assertEqual(outputs[0].bonus_token, 6)
        self.assertEqual(outputs[1].draft_target_tokens, [6, 7])
        self.assertIsNone(outputs[1].bonus_token)
        self.assertEqual(outputs[0].metadata["linear_forward_batch_kind"], "linear_tree_attention_batch_qwen3_graph")
        self.assertNotEqual(outputs[0].metadata["linear_forward_batch_kind"], "linear_prefix_step_qwen3_graph")
        self.assertTrue(outputs[0].metadata["single_pass_linear_verify"])
        self.assertTrue(outputs[0].metadata["causal_safe_prefix_batch"])
        self.assertTrue(outputs[0].metadata["explicit_kv_cache"])
        self.assertTrue(outputs[0].metadata["cuda_graph"])
        self.assertEqual(outputs[0].metadata["target_forward_call_count"], 1)
        self.assertEqual(outputs[0].metadata["shared_forward_id"], outputs[1].metadata["shared_forward_id"])

    def test_qwen3_graph_linear_verify_fails_when_tail_exceeds_captured_shape(self) -> None:
        """strict graph verifier 不应静默退回到 HF cached 或 eager path。"""
        import torch

        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(),
            model=SimpleNamespace(dtype=torch.float32),
            engine=SimpleNamespace(reset=lambda: None),
            max_len=8,
            max_graph_tokens=2,
            device="cpu",
        )

        with self.assertRaises(Qwen3GraphBackendUnavailable):
            runner.linear_verify_batch(
                [LinearForwardInput(prefix_ids=[1], draft_tokens=[2, 3, 4], allow_bonus=True)]
            )

    def test_qwen3_graph_tree_forward_batch_uses_batch_graph_engine(self) -> None:
        """qwen3_graph tree verify 应走固定 shape BatchGraphEngine，而不是 sequential fallback。"""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls: list[dict[str, object]] = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                tokens = [int(value) for value in input_ids[0].tolist()]
                self.prefill_calls.append(
                    {
                        "batch_idx": int(batch_idx),
                        "input_ids": tuple(tokens),
                        "position_ids": tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                    }
                )
                logits = torch.full((1, len(tokens), 32), -10.0, dtype=torch.float32)
                logits[0, -1, 7 + int(batch_idx)] = 10.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                self.forward_calls.append(
                    {
                        "input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids),
                        "position_ids": tuple(tuple(int(value) for value in row.tolist()) for row in position_ids),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                        "row0_node0_allowed": tuple(int(index) for index in torch.where(attention_mask[0, 0, 0] > 0)[0].tolist()),
                        "row0_node1_allowed": tuple(int(index) for index in torch.where(attention_mask[0, 0, 1] > 0)[0].tolist()),
                    }
                )
                logits = torch.full((2, 4, 32), -10.0, dtype=torch.float32)
                logits[0, 0, 8] = 10.0
                logits[0, 1, 9] = 10.0
                logits[1, 0, 10] = 10.0
                return logits

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )

        outputs = runner.tree_forward_batch(
            [
                TreeForwardInput(
                    prefix_ids=[1, 2],
                    nodes=[
                        TreeForwardNode(node_id=0, parent_id=None, token_id=7, depth=1),
                        TreeForwardNode(node_id=1, parent_id=0, token_id=8, depth=2),
                    ],
                ),
                TreeForwardInput(
                    prefix_ids=[3, 4, 5],
                    nodes=[TreeForwardNode(node_id=2, parent_id=None, token_id=9, depth=1)],
                ),
            ]
        )

        self.assertEqual(batch_engine.reset_calls, 1)
        self.assertEqual(len(batch_engine.prefill_calls), 2)
        self.assertEqual(len(batch_engine.forward_calls), 1)
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][0], (7, 8, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["position_ids"][0][:2], (2, 3))
        self.assertEqual(batch_engine.forward_calls[0]["cache_seq_indices"][:4], (2, 3, 7, 7))
        self.assertEqual(batch_engine.forward_calls[0]["row0_node0_allowed"], (0, 1, 2))
        self.assertEqual(batch_engine.forward_calls[0]["row0_node1_allowed"], (0, 1, 2, 3))
        self.assertEqual([choice.target_token_id for choice in outputs[0].choices], [7, 8])
        self.assertEqual([choice.parent_node_id for choice in outputs[0].choices], [None, 0])
        self.assertEqual([choice.prefix_len for choice in outputs[0].choices], [2, 3])
        self.assertEqual([choice.target_token_id for choice in outputs[1].choices], [8])
        self.assertEqual(outputs[0].metadata["tree_forward_batch_kind"], "tree_attention_batch_qwen3_graph")
        self.assertTrue(outputs[0].metadata["single_pass_tree_verify"])
        self.assertTrue(outputs[0].metadata["explicit_kv_cache"])
        self.assertTrue(outputs[0].metadata["cuda_graph"])

    def test_qwen3_graph_generate_tree_topk_batch_graph_uses_batch_graph_engine(self) -> None:
        """draft-side official batch tree expansion should use one BatchGraphEngine forward step."""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls: list[dict[str, object]] = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                self.prefill_calls.append(
                    {
                        "batch_idx": int(batch_idx),
                        "input_ids": tuple(int(value) for value in input_ids[0].tolist()),
                        "position_ids": tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                    }
                )
                logits = torch.full((1, input_ids.shape[1], 32), -10.0, dtype=torch.float32)
                logits[0, -1, 2 + (3 * int(batch_idx))] = 10.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                self.forward_calls.append(
                    {
                        "input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids),
                        "position_ids": tuple(tuple(int(value) for value in row.tolist()) for row in position_ids),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "attention_mask_shape": tuple(int(value) for value in attention_mask.shape),
                        "row0_allowed": tuple(int(index) for index in torch.where(attention_mask[0, 0, 0] > 0)[0].tolist()),
                        "row1_allowed": tuple(int(index) for index in torch.where(attention_mask[1, 0, 0] > 0)[0].tolist()),
                    }
                )
                logits = torch.full((2, 4, 32), -10.0, dtype=torch.float32)
                logits[0, 0, 4] = 10.0
                logits[1, 0, 8] = 10.0
                return logits

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )

        results = runner.generate_tree_topk_batch_graph(
            [
                {
                    "prefix_ids": [1, 11],
                    "max_depth": 2,
                    "max_branches": 1,
                    "max_nodes": 4,
                    "draft_batch_index": 1,
                    "request_id": "r0",
                    "runner_id": "draft0",
                    "metadata": {},
                },
                {
                    "prefix_ids": [3, 13],
                    "max_depth": 2,
                    "max_branches": 1,
                    "max_nodes": 4,
                    "draft_batch_index": 0,
                    "request_id": "r1",
                    "runner_id": "draft1",
                    "metadata": {},
                },
            ]
        )

        self.assertEqual(batch_engine.reset_calls, 1)
        self.assertEqual(len(batch_engine.prefill_calls), 2)
        self.assertEqual([call["batch_idx"] for call in batch_engine.prefill_calls], [1, 0])
        self.assertEqual(len(batch_engine.forward_calls), 1)
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][0], (2, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][1], (5, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["cache_seq_indices"][:4], (2, 7, 7, 7))
        self.assertEqual(batch_engine.forward_calls[0]["row0_allowed"], (0, 1, 2))
        self.assertEqual(batch_engine.forward_calls[0]["row1_allowed"], (0, 1, 2))
        self.assertEqual([node.token_id for node in results[0]["tree"].nodes], [5, 8])
        self.assertEqual([node.parent_id for node in results[0]["tree"].nodes], [None, 0])
        self.assertEqual([node.token_id for node in results[1]["tree"].nodes], [2, 4])
        self.assertTrue(results[0]["metadata"]["official_batch_tree_draft"])
        self.assertEqual(results[0]["metadata"]["tree_draft_backend"], "qwen3_batch_graph_official")
        self.assertEqual(results[0]["metadata"]["official_draft_batch_index"], 1)
        self.assertEqual(results[1]["metadata"]["official_draft_batch_index"], 0)

    def test_qwen3_graph_batch_tree_budget_trim_gathers_kv(self) -> None:
        """official batch tree trim must compact draft KV positions with BatchGraphEngine.gather."""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.gather_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                del position_ids, batch_idx, cache_seq_indices, attention_mask
                logits = torch.full((1, input_ids.shape[1], 32), -10.0, dtype=torch.float32)
                logits[0, -1, 2] = 10.0
                logits[0, -1, 3] = 9.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                del position_ids, cache_batch_indices, cache_seq_indices, attention_mask
                self.forward_input_ids = tuple(int(value) for value in input_ids[0].tolist())
                logits = torch.full((1, 4, 32), -10.0, dtype=torch.float32)
                logits[0, 0, 4] = 10.0
                logits[0, 1, 6] = 0.0
                return logits

            def gather(self, batch_idx, src_indices, dest_indices):
                self.gather_calls.append(
                    {
                        "batch_idx": int(batch_idx),
                        "src_indices": tuple(int(value) for value in src_indices.cpu().tolist()),
                        "dest_indices": tuple(int(value) for value in dest_indices.cpu().tolist()),
                    }
                )

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=1,
            device="cpu",
        )

        result = runner.generate_tree_topk_batch_graph(
            [
                {
                    "prefix_ids": [1],
                    "max_depth": 2,
                    "max_branches": 2,
                    "max_nodes": 2,
                    "request_id": "req-1",
                    "runner_id": "draft-graph",
                    "metadata": {},
                }
            ]
        )[0]

        self.assertEqual(batch_engine.forward_input_ids, (2, 3, 0, 0))
        self.assertEqual([node.token_id for node in result["tree"].nodes], [2, 4])
        self.assertEqual([node.parent_id for node in result["tree"].nodes], [None, 0])
        self.assertEqual(batch_engine.gather_calls, [{"batch_idx": 0, "src_indices": (1, 3), "dest_indices": (1, 2)}])
        self.assertTrue(result["metadata"]["official_budget_kv_gather"])
        gather_events = [
            event
            for event in result["metadata"]["draft_token_forward_events"]
            if event["phase"] == "draft.batch_graph_budget_gather"
        ]
        self.assertEqual(gather_events[0]["source_seq_indices"], [1, 3])
        self.assertEqual(gather_events[0]["dest_seq_indices"], [1, 2])

    def test_qwen3_graph_official_commit_gathers_batch_engine_kv(self) -> None:
        """acceptance commit should expose official gather to draft BatchGraphEngine."""
        import torch

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.gather_calls = []

            def gather(self, batch_idx, src_indices, dest_indices):
                self.gather_calls.append(
                    {
                        "batch_idx": int(batch_idx),
                        "src_indices": tuple(int(value) for value in src_indices.cpu().tolist()),
                        "dest_indices": tuple(int(value) for value in dest_indices.cpu().tolist()),
                    }
                )

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=SimpleNamespace(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )
        runner._official_draft_batch_indices["req-1"] = 1

        metadata = runner.official_specedge_commit_acceptance(
            request_id="req-1",
            batch_index=None,
            source_seq_indices=[0, 1, 4],
            dest_seq_indices=[0, 1, 2],
            prefix_ids=[10, 2, 9],
            retained_tree={"nodes": [{"node_id": 0}]},
            reused_proactive_tree=True,
        )

        self.assertEqual(batch_engine.gather_calls, [{"batch_idx": 1, "src_indices": (0, 1, 4), "dest_indices": (0, 1, 2)}])
        self.assertTrue(metadata["official_draft_kv_gather"])
        self.assertEqual(metadata["retained_tree_node_count"], 1)
        self.assertTrue(metadata["reused_proactive_tree"])

    def test_qwen3_graph_grows_official_state_without_reset_or_prefill(self) -> None:
        """persistent official grow should consume gathered KV instead of full-prefix prefill."""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, **kwargs):
                self.prefill_calls.append(kwargs)
                raise AssertionError("official persistent grow must not prefill")

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                call_index = len(self.forward_calls)
                self.forward_calls.append(
                    {
                        "input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids),
                        "position_ids": tuple(tuple(int(value) for value in row.tolist()) for row in position_ids),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "row1_allowed": tuple(int(index) for index in torch.where(attention_mask[1, 0, 0] > 0)[0].tolist()),
                    }
                )
                logits = torch.full((2, 4, 32), -10.0, dtype=torch.float32)
                logits[1, 0, 11 + call_index] = 10.0
                return logits

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )
        runner._official_draft_batch_indices["req-1"] = 1

        result = runner.grow_official_tree_batch_graph(
            [
                {
                    "prefix_ids": [10, 2, 9],
                    "tree": {"root_prefix_len": 3, "nodes": []},
                    "tree_node_statuses": {},
                    "needs_prefix_tail_forward": True,
                    "draft_batch_index": None,
                    "max_depth": 2,
                    "max_branches": 1,
                    "max_nodes": 4,
                    "request_id": "req-1",
                    "runner_id": "draft-graph",
                    "metadata": {},
                }
            ]
        )[0]

        self.assertEqual(batch_engine.reset_calls, 0)
        self.assertEqual(batch_engine.prefill_calls, [])
        self.assertEqual(len(batch_engine.forward_calls), 2)
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][1], (9, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["position_ids"][1][0], 2)
        self.assertEqual(batch_engine.forward_calls[0]["cache_seq_indices"][4:8], (2, 7, 7, 7))
        self.assertEqual(batch_engine.forward_calls[0]["row1_allowed"], (0, 1, 2))
        self.assertEqual(batch_engine.forward_calls[1]["input_ids"][1], (11, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[1]["position_ids"][1][0], 3)
        self.assertEqual(batch_engine.forward_calls[1]["cache_seq_indices"][4:8], (3, 7, 7, 7))
        self.assertEqual(batch_engine.forward_calls[1]["row1_allowed"], (0, 1, 2, 3))
        self.assertEqual([node.token_id for node in result["tree"].nodes], [11, 12])
        self.assertEqual([node.parent_id for node in result["tree"].nodes], [None, 0])
        self.assertTrue(result["metadata"]["official_persistent_kv_reused"])
        self.assertFalse(result["metadata"]["official_needs_prefix_tail_forward"])
        self.assertEqual(result["metadata"]["tree_draft_backend"], "qwen3_batch_graph_official_grow")
        self.assertEqual(result["metadata"]["tree_node_statuses"], {"0": 10, "1": 15})

    def test_qwen3_graph_official_state_grow_budget_trim_gathers_kv(self) -> None:
        """persistent official grow trim must keep CandidateTree indices aligned with draft KV."""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls = []
                self.forward_calls: list[dict[str, object]] = []
                self.gather_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, **kwargs):
                self.prefill_calls.append(kwargs)
                raise AssertionError("official persistent grow trim must not prefill")

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                del position_ids, cache_batch_indices, cache_seq_indices, attention_mask
                self.forward_calls.append(
                    {"input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids)}
                )
                logits = torch.full((2, 4, 32), -10.0, dtype=torch.float32)
                logits[1, 0, 4] = 10.0
                logits[1, 1, 6] = 0.0
                return logits

            def gather(self, batch_idx, src_indices, dest_indices):
                self.gather_calls.append(
                    {
                        "batch_idx": int(batch_idx),
                        "src_indices": tuple(int(value) for value in src_indices.cpu().tolist()),
                        "dest_indices": tuple(int(value) for value in dest_indices.cpu().tolist()),
                    }
                )

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-graph"),
                CandidateNode(1, None, 3, 1, -2.0, "draft-graph"),
            ],
        )

        result = runner.grow_official_tree_batch_graph(
            [
                {
                    "prefix_ids": [10],
                    "tree": tree.to_dict(),
                    "tree_node_statuses": {"0": 15, "1": 15},
                    "needs_prefix_tail_forward": False,
                    "draft_batch_index": 1,
                    "max_depth": 2,
                    "max_branches": 1,
                    "max_nodes": 2,
                    "request_id": "req-1",
                    "runner_id": "draft-graph",
                    "metadata": {},
                }
            ]
        )[0]

        self.assertEqual(batch_engine.reset_calls, 0)
        self.assertEqual(batch_engine.prefill_calls, [])
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][1], (2, 3, 0, 0))
        self.assertEqual([node.token_id for node in result["tree"].nodes], [2, 4])
        self.assertEqual([node.parent_id for node in result["tree"].nodes], [None, 0])
        self.assertEqual(batch_engine.gather_calls, [{"batch_idx": 1, "src_indices": (1, 3), "dest_indices": (1, 2)}])
        gather_events = [
            event
            for event in result["metadata"]["draft_token_forward_events"]
            if event["phase"] == "draft.batch_graph_official_grow_budget_gather"
        ]
        self.assertEqual(gather_events[0]["source_seq_indices"], [1, 3])
        self.assertEqual(gather_events[0]["dest_seq_indices"], [1, 2])

    def test_qwen3_graph_generates_official_proactive_without_reset_or_prefill(self) -> None:
        """official proactive should select/grow POST nodes on the persistent draft KV row."""
        import torch

        class FakeLinearEngine:
            def reset(self) -> None:
                return None

        class FakeBatchGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, **kwargs):
                self.prefill_calls.append(kwargs)
                raise AssertionError("official proactive graph must not prefill")

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                call_index = len(self.forward_calls)
                self.forward_calls.append(
                    {
                        "input_ids": tuple(tuple(int(value) for value in row.tolist()) for row in input_ids),
                        "position_ids": tuple(tuple(int(value) for value in row.tolist()) for row in position_ids),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "row1_allowed": tuple(int(index) for index in torch.where(attention_mask[1, 0, 0] > 0)[0].tolist()),
                    }
                )
                logits = torch.full((2, 4, 32), -10.0, dtype=torch.float32)
                logits[1, 0, 9 if call_index == 0 else 11] = 10.0
                if call_index == 1:
                    logits[1, 0, 12] = 0.0
                return logits

        batch_engine = FakeBatchGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
            model=SimpleNamespace(dtype=torch.float32),
            engine=FakeLinearEngine(),
            batch_engine=batch_engine,
            max_len=8,
            max_graph_tokens=4,
            max_graph_batch_size=2,
            device="cpu",
        )
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-graph")],
        )

        result = runner.generate_official_proactive_graph(
            {
                "prefix_ids": [10],
                "tree": tree.to_dict(),
                "tree_node_statuses": {"0": 10},
                "draft_batch_index": 1,
                "max_depth": 1,
                "max_branches": 2,
                "max_nodes": 2,
                "max_leaf_beams": 1,
                "root_top_k": 1,
                "prompt_len": 1,
                "max_new_tokens": 4,
                "request_id": "req-1",
                "runner_id": "draft-graph",
                "metadata": {},
            }
        )

        assert result is not None
        self.assertEqual(batch_engine.reset_calls, 0)
        self.assertEqual(batch_engine.prefill_calls, [])
        self.assertEqual(len(batch_engine.forward_calls), 2)
        self.assertEqual(batch_engine.forward_calls[0]["input_ids"][1], (2, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[0]["position_ids"][1][0], 1)
        self.assertEqual(batch_engine.forward_calls[0]["cache_seq_indices"][4:8], (1, 7, 7, 7))
        self.assertEqual(batch_engine.forward_calls[0]["row1_allowed"], (0, 1))
        self.assertEqual(batch_engine.forward_calls[1]["input_ids"][1], (9, 0, 0, 0))
        self.assertEqual(batch_engine.forward_calls[1]["position_ids"][1][0], 2)
        self.assertEqual(batch_engine.forward_calls[1]["cache_seq_indices"][4:8], (2, 7, 7, 7))
        self.assertEqual(batch_engine.forward_calls[1]["row1_allowed"], (0, 1, 2))
        self.assertEqual(result["parent_node_id"], 0)
        self.assertEqual(result["root_token_id"], 9)
        self.assertEqual(result["root_status"], 25)
        self.assertEqual([node.token_id for node in result["subtree"].nodes], [11])
        self.assertEqual(result["subtree_statuses"], {"0": 15})
        self.assertTrue(result["metadata"]["official_proactive_graph"])
        self.assertTrue(result["metadata"]["official_persistent_kv_reused"])
        self.assertTrue(result["metadata"]["official_proactive_budget_pruning"])
        self.assertEqual(result["metadata"]["tree_draft_backend"], "qwen3_batch_graph_official_proactive")

    def test_qwen3_graph_generate_tree_topk_graph_expands_frontier_with_kv_masks(self) -> None:
        """draft-side graph tree expansion should keep frontier KV positions and tree masks explicit."""
        import torch

        class FakeGraphEngine:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prefill_calls: list[dict[str, object]] = []
                self.forward_calls: list[dict[str, object]] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                self.prefill_calls.append(
                    {
                        "input_ids": tuple(int(value) for value in input_ids[0].tolist()),
                        "position_ids": tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                    }
                )
                logits = torch.full((1, input_ids.shape[1], 16), -10.0, dtype=torch.float32)
                logits[0, -1, 2] = 10.0
                logits[0, -1, 3] = 9.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                self.forward_calls.append(
                    {
                        "input_ids": tuple(int(value) for value in input_ids[0].tolist()),
                        "position_ids": tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_batch_indices": tuple(int(value) for value in cache_batch_indices.tolist()),
                        "cache_seq_indices": tuple(int(value) for value in cache_seq_indices.tolist()),
                        "row0_allowed": tuple(int(index) for index in torch.where(attention_mask[0, 0, 0] > 0)[0].tolist()),
                        "row1_allowed": tuple(int(index) for index in torch.where(attention_mask[0, 0, 1] > 0)[0].tolist()),
                    }
                )
                logits = torch.full((1, input_ids.shape[1], 16), -10.0, dtype=torch.float32)
                logits[0, 0, 4] = 10.0
                logits[0, 0, 5] = 9.0
                logits[0, 1, 6] = 10.0
                logits[0, 1, 7] = 9.0
                return logits

        engine = FakeGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(),
            model=SimpleNamespace(dtype=torch.float32),
            engine=engine,
            max_len=8,
            max_graph_tokens=4,
            device="cpu",
        )

        result = runner.generate_tree_topk_graph(
            prefix_ids=[1],
            max_depth=2,
            max_branches=2,
            max_nodes=6,
            request_id="req",
            runner_id="draft-graph",
        )

        tree = result["tree"]
        self.assertEqual(engine.reset_calls, 1)
        self.assertEqual(engine.prefill_calls[0]["input_ids"], (1,))
        self.assertEqual(engine.forward_calls[0]["input_ids"], (2, 3))
        self.assertEqual(engine.forward_calls[0]["cache_seq_indices"], (1, 2))
        self.assertEqual(engine.forward_calls[0]["row0_allowed"], (0, 1))
        self.assertEqual(engine.forward_calls[0]["row1_allowed"], (0, 2))
        self.assertEqual([node.token_id for node in tree.nodes], [2, 3, 4, 5, 6, 7])
        self.assertEqual([node.parent_id for node in tree.nodes], [None, None, 0, 0, 1, 1])
        self.assertTrue(result["metadata"]["graph_tree_draft"])
        self.assertEqual(result["metadata"]["tree_draft_backend"], "qwen3_graph_frontier")

    def test_qwen3_graph_generate_tree_topk_graph_uses_official_budget_pruning(self) -> None:
        """Graph tree draft should keep high-score incoming beams, not just BFS order."""
        import torch

        class FakeGraphEngine:
            def reset(self) -> None:
                return None

            def prefill(self, *, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
                del position_ids, batch_idx, cache_seq_indices, attention_mask
                logits = torch.full((1, input_ids.shape[1], 16), -10.0, dtype=torch.float32)
                logits[0, -1, 2] = 10.0
                logits[0, -1, 3] = 8.0
                return logits

            def forward(self, *, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
                del position_ids, cache_batch_indices, cache_seq_indices, attention_mask
                self.forward_input_ids = tuple(int(value) for value in input_ids[0].tolist())
                logits = torch.full((1, input_ids.shape[1], 16), -10.0, dtype=torch.float32)
                logits[0, 0, 4] = 10.0
                logits[0, 0, 5] = 0.0
                logits[0, 1, 6] = 10.0
                logits[0, 1, 7] = 9.0
                return logits

        engine = FakeGraphEngine()
        runner = Qwen3GraphCausalLMRunner(
            runner_id="graph",
            tokenizer=SimpleNamespace(),
            model=SimpleNamespace(dtype=torch.float32),
            engine=engine,
            max_len=8,
            max_graph_tokens=4,
            device="cpu",
        )

        result = runner.generate_tree_topk_graph(
            prefix_ids=[1],
            max_depth=2,
            max_branches=2,
            max_nodes=4,
            request_id="req",
            runner_id="draft-graph",
        )

        tree = result["tree"]
        self.assertEqual(engine.forward_input_ids, (2, 3))
        self.assertEqual([node.token_id for node in tree.nodes], [2, 3, 4, 6])
        self.assertEqual([node.parent_id for node in tree.nodes], [None, None, 0, 1])
        self.assertTrue(result["metadata"]["official_budget_pruning"])
        self.assertEqual(len({node.node_id for node in tree.nodes}), len(tree.nodes))

    def test_transformers_batch_next_token_buckets_by_prefix_length(self) -> None:
        """变长 next-token batch 不应通过 padding 改变 greedy 语义。"""
        import torch

        class FakeModel:
            config = SimpleNamespace(pad_token_id=0, eos_token_id=0)

            def __init__(self) -> None:
                self.calls: list[tuple[list[list[int]], bool, bool]] = []

            def __call__(self, *, input_ids, attention_mask=None, position_ids=None):
                rows = [[int(value) for value in row] for row in input_ids.tolist()]
                self.calls.append((rows, attention_mask is not None, position_ids is not None))
                logits = torch.full((len(rows), len(rows[0]), 16), -10.0, dtype=torch.float32)
                for row_index, row in enumerate(rows):
                    logits[row_index, -1, (row[-1] + 1) % 16] = 10.0
                return SimpleNamespace(logits=logits)

        model = FakeModel()
        runner = TransformersCausalLMRunner(
            runner_id="hf",
            tokenizer=SimpleNamespace(pad_token_id=0),
            model=model,
            max_len=16,
            device="cpu",
        )

        token_ids = runner.greedy_next_tokens([[1, 2], [3, 4, 5], [6, 7]])

        self.assertEqual(token_ids, [3, 6, 8])
        self.assertEqual(
            model.calls,
            [
                ([[1, 2], [6, 7]], False, False),
                ([[3, 4, 5]], False, False),
            ],
        )

    def test_transformers_linear_verify_batch_uses_causal_safe_prefixes(self) -> None:
        """linear verify 应只用每个位置真实可见的 prefix 做 batch。"""
        import torch

        class FakeModel:
            config = SimpleNamespace(pad_token_id=0, eos_token_id=0)

            def __init__(self) -> None:
                self.calls: list[list[list[int]]] = []

            def __call__(self, *, input_ids, attention_mask=None, position_ids=None):
                rows = [[int(value) for value in row] for row in input_ids.tolist()]
                self.calls.append(rows)
                logits = torch.full((len(rows), len(rows[0]), 16), -10.0, dtype=torch.float32)
                for row_index, row in enumerate(rows):
                    for position, token_id in enumerate(row):
                        logits[row_index, position, (token_id + 1) % 16] = 10.0
                return SimpleNamespace(logits=logits)

        model = FakeModel()
        runner = TransformersCausalLMRunner(
            runner_id="hf",
            tokenizer=SimpleNamespace(pad_token_id=0),
            model=model,
            max_len=16,
            device="cpu",
        )

        outputs = runner.linear_verify_batch(
            [
                LinearForwardInput(prefix_ids=[1, 2], draft_tokens=[3], allow_bonus=True),
                LinearForwardInput(prefix_ids=[4, 5, 6], draft_tokens=[7], allow_bonus=True),
                LinearForwardInput(prefix_ids=[8, 9], draft_tokens=[10], allow_bonus=False),
            ]
        )

        self.assertEqual(
            model.calls,
            [
                [[1, 2], [8, 9]],
                [[1, 2, 3], [4, 5, 6]],
                [[4, 5, 6, 7]],
            ],
        )
        self.assertEqual(outputs[0].draft_target_tokens, [3])
        self.assertEqual(outputs[0].bonus_token, 4)
        self.assertEqual(outputs[1].draft_target_tokens, [7])
        self.assertEqual(outputs[1].bonus_token, 8)
        self.assertEqual(outputs[2].draft_target_tokens, [10])
        self.assertIsNone(outputs[2].bonus_token)
        self.assertEqual(outputs[0].metadata["linear_forward_batch_kind"], "linear_prefix_batch")
        self.assertTrue(outputs[0].metadata["causal_safe_prefix_batch"])

    def test_cached_transformers_runner_reuses_longest_prefix_past(self) -> None:
        """HF cached runner 应复用最长 prefix cache，只 forward 新增 token。"""
        import torch

        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None, bool]] = []

            def __call__(self, *, input_ids, past_key_values=None, use_cache=False):
                input_tuple = tuple(int(value) for value in input_ids[0].tolist())
                self.calls.append((input_tuple, past_key_values, bool(use_cache)))
                full_prefix = tuple(past_key_values or ()) + input_tuple
                logits = torch.full((1, len(input_tuple), 6), -10.0, dtype=torch.float32)
                logits[0, -1, (full_prefix[-1] + 1) % 6] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=full_prefix)

        model = FakeModel()
        runner = CachedTransformersCausalLMRunner(
            runner_id="cached",
            tokenizer=object(),
            model=model,
            max_len=16,
            device="cpu",
        )

        first = runner.next_token_topk([1], 2)
        second = runner.next_token_topk([1, 2], 2)

        self.assertEqual([call[0] for call in model.calls], [(1,), (2,)])
        self.assertIsNone(model.calls[0][1])
        self.assertEqual(model.calls[1][1], (1,))
        self.assertTrue(all(call[2] for call in model.calls))
        self.assertEqual(first[0].token_id, 2)
        self.assertEqual(second[0].token_id, 3)
        capabilities = runner.backend_capabilities().to_dict()
        self.assertEqual(capabilities["backend_name"], "hf_cached")
        self.assertTrue(capabilities["supports_batched_next_token"])
        self.assertTrue(capabilities["supports_tree_attention"])
        self.assertTrue(capabilities["supports_kv_cache"])

    def test_cached_transformers_greedy_uses_kv_cache_with_explicit_positions(self) -> None:
        """target-only greedy baseline 在 cached runner 中使用显式 position 的 KV 路径。"""
        import torch

        class FakeModel:
            config = SimpleNamespace(pad_token_id=0, eos_token_id=0)

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(
                self,
                *,
                input_ids,
                past_key_values=None,
                use_cache=False,
                attention_mask=None,
                position_ids=None,
                cache_position=None,
            ):
                input_tuple = tuple(int(value) for value in input_ids[0].tolist())
                self.calls.append(
                    {
                        "input_ids": input_tuple,
                        "past": past_key_values,
                        "use_cache": bool(use_cache),
                        "attention_mask": None
                        if attention_mask is None
                        else tuple(int(value) for value in attention_mask.shape),
                        "position_ids": None
                        if position_ids is None
                        else tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_position": None
                        if cache_position is None
                        else tuple(int(value) for value in cache_position.tolist()),
                    }
                )
                logits = torch.full((1, len(input_tuple), 8), -10.0, dtype=torch.float32)
                logits[0, -1, (input_tuple[-1] + 1) % 8] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=input_tuple)

        model = FakeModel()
        runner = CachedTransformersCausalLMRunner(
            runner_id="cached",
            tokenizer=object(),
            model=model,
            max_len=16,
            device="cpu",
        )

        first = runner.greedy_next_token([1, 2, 3])
        second = runner.greedy_next_token([1, 2, 3, 4])

        self.assertEqual(first, 4)
        self.assertEqual(second, 5)
        self.assertEqual(model.calls[0]["input_ids"], (1, 2, 3))
        self.assertIsNone(model.calls[0]["past"])
        self.assertEqual(model.calls[0]["attention_mask"], (1, 1, 3, 3))
        self.assertEqual(model.calls[0]["position_ids"], (0, 1, 2))
        self.assertEqual(model.calls[0]["cache_position"], (0, 1, 2))
        self.assertEqual(model.calls[1]["input_ids"], (4,))
        self.assertEqual(model.calls[1]["past"], (1, 2, 3))
        self.assertEqual(model.calls[1]["attention_mask"], (1, 1, 1, 4))
        self.assertEqual(model.calls[1]["position_ids"], (3,))
        self.assertEqual(model.calls[1]["cache_position"], (3,))

    def test_cached_transformers_linear_verify_reuses_prefix_kv(self) -> None:
        """cached linear verify 应复用 prefix KV，并只 forward draft tail。"""
        import torch

        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None, bool]] = []

            def __call__(self, *, input_ids, past_key_values=None, use_cache=False):
                input_tuple = tuple(int(value) for value in input_ids[0].tolist())
                self.calls.append((input_tuple, past_key_values, bool(use_cache)))
                base = tuple(past_key_values or ())
                full_prefix = base + input_tuple
                logits = torch.full((1, len(input_tuple), 8), -10.0, dtype=torch.float32)
                for position in range(len(input_tuple)):
                    partial = base + input_tuple[: position + 1]
                    logits[0, position, (partial[-1] + 1) % 8] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=full_prefix)

        model = FakeModel()
        runner = CachedTransformersCausalLMRunner(
            runner_id="cached",
            tokenizer=object(),
            model=model,
            max_len=16,
            device="cpu",
        )

        first = runner.linear_verify_batch(
            [
                LinearForwardInput(
                    prefix_ids=[1],
                    draft_tokens=[2, 3],
                    allow_bonus=True,
                )
            ]
        )[0]
        second = runner.linear_verify_batch(
            [
                LinearForwardInput(
                    prefix_ids=[1, 2, 3],
                    draft_tokens=[4],
                    allow_bonus=True,
                )
            ]
        )[0]

        self.assertEqual(first.draft_target_tokens, [2, 3])
        self.assertEqual(first.bonus_token, 4)
        self.assertEqual(first.metadata["prefix_cache_hit"], False)
        self.assertEqual(first.metadata["prefix_forward_call_count"], 1)
        self.assertEqual(first.metadata["tail_forward_call_count"], 1)
        self.assertEqual(second.draft_target_tokens, [4])
        self.assertEqual(second.bonus_token, 5)
        self.assertEqual(second.metadata["prefix_cache_hit"], True)
        self.assertEqual(second.metadata["prefix_forward_call_count"], 0)
        self.assertEqual(second.metadata["tail_forward_call_count"], 1)
        self.assertEqual([call[0] for call in model.calls], [(1,), (2, 3), (4,)])
        self.assertIsNone(model.calls[0][1])
        self.assertEqual(model.calls[1][1], (1,))
        self.assertEqual(model.calls[2][1], (1, 2, 3))
        self.assertTrue(all(call[2] for call in model.calls))

    def test_cached_transformers_linear_verify_passes_explicit_tail_positions(self) -> None:
        """cached tail forward 应显式传入 attention/position/cache_position。"""
        import torch

        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(
                self,
                *,
                input_ids,
                past_key_values=None,
                use_cache=False,
                attention_mask=None,
                position_ids=None,
                cache_position=None,
            ):
                input_tuple = tuple(int(value) for value in input_ids[0].tolist())
                base = tuple(past_key_values or ())
                self.calls.append(
                    {
                        "input_ids": input_tuple,
                        "past": past_key_values,
                        "use_cache": bool(use_cache),
                        "attention_mask": None
                        if attention_mask is None
                        else tuple(int(value) for value in attention_mask.shape),
                        "position_ids": None
                        if position_ids is None
                        else tuple(int(value) for value in position_ids[0].tolist()),
                        "cache_position": None
                        if cache_position is None
                        else tuple(int(value) for value in cache_position.tolist()),
                    }
                )
                full_prefix = base + input_tuple
                logits = torch.full((1, len(input_tuple), 16), -10.0, dtype=torch.float32)
                for position in range(len(input_tuple)):
                    partial = base + input_tuple[: position + 1]
                    logits[0, position, (partial[-1] + 1) % 16] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=full_prefix)

        model = FakeModel()
        runner = CachedTransformersCausalLMRunner(
            runner_id="cached",
            tokenizer=object(),
            model=model,
            max_len=16,
            device="cpu",
        )

        output = runner.linear_verify_batch(
            [LinearForwardInput(prefix_ids=[1, 2, 3], draft_tokens=[4, 5], allow_bonus=True)]
        )[0]

        self.assertEqual(output.draft_target_tokens, [4, 5])
        self.assertEqual(output.bonus_token, 6)
        self.assertEqual(model.calls[0]["attention_mask"], (1, 1, 3, 3))
        self.assertEqual(model.calls[0]["position_ids"], (0, 1, 2))
        self.assertEqual(model.calls[0]["cache_position"], (0, 1, 2))
        self.assertEqual(model.calls[1]["attention_mask"], (1, 1, 2, 5))
        self.assertEqual(model.calls[1]["position_ids"], (3, 4))
        self.assertEqual(model.calls[1]["cache_position"], (3, 4))

    def test_cached_transformers_linear_verify_clones_mutable_prefix_cache(self) -> None:
        """HF mutable cache 复用前必须 clone，否则 prefix cache 会被 draft tail 污染。"""
        import torch

        class MutablePast:
            def __init__(self, tokens: tuple[int, ...]) -> None:
                self.tokens = list(tokens)

            def __deepcopy__(self, memo) -> "MutablePast":
                del memo
                return MutablePast(tuple(self.tokens))

        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []

            def __call__(self, *, input_ids, past_key_values=None, use_cache=False):
                del use_cache
                input_tuple = tuple(int(value) for value in input_ids[0].tolist())
                past_before = None if past_key_values is None else tuple(past_key_values.tokens)
                self.calls.append((input_tuple, past_before))
                if past_key_values is not None:
                    past_key_values.tokens.extend(input_tuple)
                    full_prefix = tuple(past_key_values.tokens)
                else:
                    full_prefix = input_tuple
                logits = torch.full((1, len(input_tuple), 16), -10.0, dtype=torch.float32)
                for position in range(len(input_tuple)):
                    partial = full_prefix[: len(full_prefix) - len(input_tuple) + position + 1]
                    logits[0, position, (partial[-1] + 1) % 16] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=MutablePast(full_prefix))

        model = FakeModel()
        runner = CachedTransformersCausalLMRunner(
            runner_id="cached",
            tokenizer=object(),
            model=model,
            max_len=16,
            device="cpu",
        )

        first = runner.linear_verify_batch(
            [LinearForwardInput(prefix_ids=[1, 2, 3], draft_tokens=[4, 5], allow_bonus=True)]
        )[0]
        second = runner.linear_verify_batch(
            [LinearForwardInput(prefix_ids=[1, 2, 3], draft_tokens=[4, 5], allow_bonus=True)]
        )[0]

        self.assertEqual(first.draft_target_tokens, [4, 5])
        self.assertEqual(first.bonus_token, 6)
        self.assertEqual(second.draft_target_tokens, [4, 5])
        self.assertEqual(second.bonus_token, 6)
        self.assertEqual(model.calls[1], ((4, 5), (1, 2, 3)))
        self.assertEqual(model.calls[2], ((4, 5), (1, 2, 3)))
        self.assertEqual(runner._prefix_cache[(1, 2, 3)][0].tokens, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
