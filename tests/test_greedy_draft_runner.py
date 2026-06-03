"""Step 2：GreedyDraftRunner 的最小行为测试。"""

import math
import unittest

from specplatform.core import CandidateNode, CandidateTree
from specplatform.draft import GreedyDraftRunner, TopKTreeDraftRunner
from specplatform.model import CausalLMRunner, ModelBackendCapabilities, ModelForwardInput, ModelForwardOutput, TopKToken


class ScriptedCausalLMRunner(CausalLMRunner):
    """测试内部使用的脚本化 causal LM。

    它不是生产 fake model，只是用一个 prefix -> next token 映射验证 draft runner
    是否按真实接口连续调用 greedy_next_token。
    """

    runner_id = "scripted-draft-model"
    max_len = 32

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        self.next_tokens_by_prefix = dict(next_tokens_by_prefix)
        self.seen_prefixes: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        """把空格分隔的数字文本编码成 token ids。"""
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 解码成空格分隔文本。"""
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """保留 ModelRunner 契约；本测试直接覆盖 next_token_logits。"""
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """根据完整 prefix 返回 one-hot 风格 logits。"""
        self.seen_prefixes.append(list(prefix_ids))
        token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
        logits = [-10.0] * 16
        logits[token_id] = 10.0
        return logits


class GreedyDraftRunnerTest(unittest.TestCase):
    """验证 draft runner 只生成 draft tokens，不越界做 proposal/verification/session 工作。"""

    def test_generate_tokens_extends_local_prefix_greedily(self) -> None:
        """runner 应该把上一步生成的 token 接到局部 prefix 后再预测下一步。"""
        model = ScriptedCausalLMRunner(
            {
                (1, 2): 4,
                (1, 2, 4): 7,
                (1, 2, 4, 7): 3,
            }
        )
        runner = GreedyDraftRunner(model=model, runner_id="draft-worker-0")

        generation = runner.generate_tokens(
            prefix_ids=[1, 2],
            max_tokens=3,
            request_id="request-1",
            metadata={"draft_budget_id": "budget-1"},
        )

        self.assertEqual(generation.tokens, [4, 7, 3])
        self.assertEqual(model.seen_prefixes, [[1, 2], [1, 2, 4], [1, 2, 4, 7]])
        self.assertEqual(generation.timing, {})
        self.assertEqual(generation.metadata["request_id"], "request-1")
        self.assertEqual(generation.metadata["runner_id"], "draft-worker-0")
        self.assertEqual(generation.metadata["prefix_ids"], [1, 2])
        self.assertEqual(generation.metadata["max_tokens"], 3)
        self.assertEqual(generation.metadata["draft_budget_id"], "budget-1")

    def test_generate_tokens_does_not_mutate_input_prefix(self) -> None:
        """调用者持有的 prefix 通常来自 GenerationSession，draft runner 不能原地修改它。"""
        model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        runner = GreedyDraftRunner(model=model)
        prefix_ids = [1]

        generation = runner.generate_tokens(prefix_ids=prefix_ids, max_tokens=2)

        self.assertEqual(generation.tokens, [2, 3])
        self.assertEqual(prefix_ids, [1])

    def test_generate_tokens_returns_empty_when_budget_is_zero(self) -> None:
        """零预算表示本轮不生成 draft token，也不应该触发模型调用。"""
        model = ScriptedCausalLMRunner({})
        runner = GreedyDraftRunner(model=model)

        generation = runner.generate_tokens(prefix_ids=[1, 2], max_tokens=0)

        self.assertEqual(generation.tokens, [])
        self.assertEqual(model.seen_prefixes, [])

    def test_generate_tokens_rejects_empty_prefix(self) -> None:
        """空 prefix 没有 causal context，应交给上游 session 初始化流程处理。"""
        model = ScriptedCausalLMRunner({})
        runner = GreedyDraftRunner(model=model)

        with self.assertRaises(ValueError):
            runner.generate_tokens(prefix_ids=[], max_tokens=1)

    def test_generate_tokens_until_confidence_drop_stops_after_trigger_token(self) -> None:
        """SLED dynamic drafting should include the low-confidence trigger token."""

        class ConfidenceScriptedRunner(ScriptedCausalLMRunner):
            def __init__(self) -> None:
                super().__init__({})

            def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
                self.seen_prefixes.append(list(prefix_ids))
                if tuple(prefix_ids) == (1,):
                    return [0.0, 4.0, 0.0, 0.0]
                if tuple(prefix_ids) == (1, 1):
                    return [0.0, 0.1, 0.0, 0.0]
                raise KeyError(tuple(prefix_ids))

        model = ConfidenceScriptedRunner()
        runner = GreedyDraftRunner(model=model, runner_id="edge-0")

        generation = runner.generate_tokens_until_confidence_drop(
            prefix_ids=[1],
            max_tokens=4,
            confidence_threshold=0.4,
            request_id="request-1",
        )

        self.assertEqual(generation.tokens, [1, 1])
        self.assertEqual(model.seen_prefixes, [[1], [1, 1]])
        self.assertEqual(generation.metadata["dynamic_stop_reason"], "confidence_below_threshold")
        self.assertTrue(generation.metadata["dynamic_drafting"])
        self.assertLess(generation.metadata["draft_confidences"][-1], 0.4)

    def test_topk_tree_runner_batches_frontier_by_depth(self) -> None:
        """SpecEdge tree draft should expand same-depth frontier with one top-k batch."""

        class BatchedTopKRunner(ScriptedCausalLMRunner):
            def __init__(self) -> None:
                super().__init__({})
                self.topk_by_prefix = {
                    (1,): [2, 3],
                    (1, 2): [4, 5],
                    (1, 3): [6, 7],
                }
                self.topk_batch_calls: list[list[list[int]]] = []

            def backend_capabilities(self) -> ModelBackendCapabilities:
                return ModelBackendCapabilities(
                    backend_name="batched-topk-scripted",
                    supports_batched_topk=True,
                )

            def next_token_topk_batch(self, prefix_ids_batch: list[list[int]], k: int) -> list[list[TopKToken]]:
                self.topk_batch_calls.append([list(prefix_ids) for prefix_ids in prefix_ids_batch])
                outputs: list[list[TopKToken]] = []
                for prefix_ids in prefix_ids_batch:
                    token_ids = self.topk_by_prefix[tuple(prefix_ids)][:k]
                    logprob = -math.log(max(1, len(token_ids)))
                    outputs.append(
                        [
                            TopKToken(token_id=token_id, logprob=logprob, rank=rank)
                            for rank, token_id in enumerate(token_ids)
                        ]
                    )
                return outputs

        model = BatchedTopKRunner()
        runner = TopKTreeDraftRunner(model=model, runner_id="draft-tree")

        generation = runner.generate_tree(
            prefix_ids=[1],
            max_depth=2,
            max_branches=2,
            max_nodes=6,
        )

        self.assertEqual(model.topk_batch_calls, [[[1]], [[1, 2], [1, 3]]])
        self.assertEqual([node.token_id for node in generation.tree.nodes], [2, 3, 4, 5, 6, 7])
        self.assertEqual([node.parent_id for node in generation.tree.nodes], [None, None, 0, 0, 1, 1])
        events = generation.metadata["draft_token_forward_events"]
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1]["shared_batch_event_id"], events[2]["shared_batch_event_id"])
        self.assertEqual(events[1]["topk_batch_kind"], "batched_topk")

    def test_topk_tree_runner_delegates_to_graph_tree_draft_when_available(self) -> None:
        """qwen3_graph draft workers should use the graph/KV tree generator boundary."""

        class GraphTreeRunner(ScriptedCausalLMRunner):
            def __init__(self) -> None:
                super().__init__({})
                self.graph_calls: list[dict[str, object]] = []

            def backend_capabilities(self) -> ModelBackendCapabilities:
                return ModelBackendCapabilities(
                    backend_name="qwen3_graph",
                    supports_kv_cache=True,
                    supports_cuda_graph=True,
                )

            def generate_tree_topk_graph(self, **kwargs):
                self.graph_calls.append(dict(kwargs))
                return {
                    "tree": CandidateTree(
                        root_prefix_len=len(kwargs["prefix_ids"]),
                        nodes=[
                            CandidateNode(
                                node_id=0,
                                parent_id=None,
                                token_id=9,
                                depth=1,
                                draft_logprob=-0.1,
                                draft_worker_id=kwargs["runner_id"],
                            )
                        ],
                    ),
                    "metadata": {
                        "graph_tree_draft": True,
                        "draft_token_forward_events": [{"phase": "draft.graph_topk"}],
                    },
                }

        model = GraphTreeRunner()
        runner = TopKTreeDraftRunner(model=model, runner_id="graph-worker")

        generation = runner.generate_tree(
            prefix_ids=[1, 2],
            max_depth=4,
            max_branches=2,
            max_nodes=8,
            request_id="req-1",
        )

        self.assertEqual(len(model.graph_calls), 1)
        self.assertEqual(model.graph_calls[0]["prefix_ids"], [1, 2])
        self.assertEqual(model.graph_calls[0]["max_depth"], 4)
        self.assertEqual(model.graph_calls[0]["runner_id"], "graph-worker")
        self.assertEqual([node.token_id for node in generation.tree.nodes], [9])
        self.assertTrue(generation.metadata["graph_tree_draft"])
        self.assertEqual(generation.metadata["tree_snapshot"]["nodes"][0]["token_id"], 9)


if __name__ == "__main__":
    unittest.main()
