"""Step 2：GreedyDraftRunner 的最小行为测试。"""

import unittest

from specplatform.draft import GreedyDraftRunner
from specplatform.model import CausalLMRunner, ModelForwardInput, ModelForwardOutput


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


if __name__ == "__main__":
    unittest.main()
