"""Step 1：真实 CausalLMRunner 接口的最小测试。"""

import unittest

from specplatform.model import CausalLMRunner, ModelForwardInput, ModelForwardOutput


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


if __name__ == "__main__":
    unittest.main()
