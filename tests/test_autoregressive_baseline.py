"""auto-regressive baseline 的最小行为测试。"""

import unittest

from specplatform.model import FakeDeterministicModelRunner, ModelForwardInput, ModelForwardOutput
from specplatform.runtime import GenerationSession, run_autoregressive_baseline


class RecordingFakeRunner(FakeDeterministicModelRunner):
    """记录 forward 输入，验证 baseline 每轮只喂 1 个 token。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.forward_inputs: list[list[int]] = []
        self.forward_positions: list[list[int] | None] = []

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """记录请求后复用 fake runner 的 deterministic forward。"""
        self.forward_inputs.append(list(request.input_ids))
        self.forward_positions.append(
            list(request.position_ids) if request.position_ids is not None else None
        )
        return super().forward(request)


class AutoRegressiveBaselineTest(unittest.TestCase):
    """验证普通逐 token 生成对照组。"""

    def test_baseline_generates_one_token_per_forward_until_max_new_tokens(self) -> None:
        """baseline 应逐 token forward，并在 max_new_tokens 停止。"""
        session = GenerationSession(
            request_id="req0",
            prompt_ids=[1, 2],
            max_new_tokens=3,
            max_len=16,
        )
        runner = RecordingFakeRunner(runner_id="target0", vocab_size=16)

        result = run_autoregressive_baseline(session=session, model_runner=runner)

        self.assertEqual(result.output_token_ids, [4, 7, 11])
        self.assertEqual(session.generated_ids, [4, 7, 11])
        self.assertEqual(result.step_count, 3)
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(runner.forward_inputs, [[2], [4], [7]])
        self.assertEqual(runner.forward_positions, [[1], [2], [3]])

    def test_baseline_stops_when_eos_is_generated(self) -> None:
        """生成 eos 时应提前结束 session。"""
        session = GenerationSession(
            request_id="req0",
            prompt_ids=[1, 2],
            max_new_tokens=4,
            max_len=16,
            eos_token_id=4,
        )
        runner = RecordingFakeRunner(runner_id="target0", vocab_size=16)

        result = run_autoregressive_baseline(session=session, model_runner=runner)

        self.assertEqual(result.output_token_ids, [4])
        self.assertTrue(session.is_finished)
        self.assertEqual(result.step_count, 1)
        self.assertEqual(result.stop_reason, "eos")


if __name__ == "__main__":
    unittest.main()
