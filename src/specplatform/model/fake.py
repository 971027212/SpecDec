from __future__ import annotations

"""确定性的 fake model runner。

它用简单公式生成 logits，方便测试生成流程。这里不模拟真实概率分布，
只提供稳定、可预测的 token 选择结果。
"""

import time
from dataclasses import dataclass

from specplatform.model.base import ModelForwardInput, ModelForwardOutput, ModelRunner


@dataclass
class FakeDeterministicModelRunner(ModelRunner):
    """根据 token_id 和 position 选择固定 preferred token 的 fake runner。"""

    runner_id: str = "fake"
    vocab_size: int = 16
    max_len: int = 128
    timing_ms: float = 0.1

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """为每个输入 token 返回一行 deterministic logits。"""
        start_ns = time.perf_counter_ns()
        logits: list[list[float]] = []
        for offset, token_id in enumerate(request.input_ids):
            position = request.position_ids[offset] if request.position_ids else offset
            logits.append(self._logits_for(int(token_id), int(position)))
        end_ns = start_ns + int(round(self.timing_ms * 1_000_000))
        return ModelForwardOutput(
            logits=logits,
            timing_ms=self.timing_ms,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def _logits_for(self, token_id: int, position: int) -> list[float]:
        """构造一行 logits，让 argmax token 可预测。"""
        preferred = (token_id + 1 + position) % self.vocab_size
        second = (preferred + 1) % self.vocab_size
        third = (preferred + 2) % self.vocab_size
        logits = [-8.0] * self.vocab_size
        logits[preferred] = 8.0
        logits[second] = 5.0
        logits[third] = 3.0
        return logits
