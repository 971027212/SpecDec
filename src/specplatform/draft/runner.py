from __future__ import annotations

"""Phase 1 的 fake draft runner。

当前实现继承 deterministic fake model，仅提供 encode/decode 便利方法。
Step 2 会在这个边界附近补“一次生成多个 draft tokens”的能力。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.model import ModelForwardInput
from specplatform.model.fake import FakeDeterministicModelRunner


@dataclass(frozen=True)
class FakeDraftGeneration:
    """一次 fake draft 连续生成的结果。

    tokens 是后续 CandidateProposal.tokens 的来源；timing 字段只记录 draft
    runner 自己的 forward 信息，不参与接受/拒绝决策。
    """

    tokens: list[int] = field(default_factory=list)
    forward_timing_ms: list[float] = field(default_factory=list)
    forward_intervals_ns: list[dict[str, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeDraftRunner(FakeDeterministicModelRunner):
    """用于测试 draft 阶段的 fake runner。"""

    def generate_tokens(
        self,
        *,
        prefix_ids: list[int],
        max_tokens: int,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FakeDraftGeneration:
        """基于当前 prefix 连续生成最多 max_tokens 个 draft token。

        这里是 Step 2 的核心：draft runner 负责多步 forward；method 只负责把
        这些 draft token 包装成 CandidateProposal。
        """
        if not prefix_ids:
            raise ValueError("FakeDraftRunner.generate_tokens requires a non-empty prefix.")
        if max_tokens <= 0:
            return FakeDraftGeneration(metadata=dict(metadata or {}))

        tokens: list[int] = []
        forward_timing_ms: list[float] = []
        forward_intervals_ns: list[dict[str, int]] = []
        cursor = int(prefix_ids[-1])
        for offset in range(max_tokens):
            output = self.forward(
                ModelForwardInput(
                    input_ids=[cursor],
                    position_ids=[len(prefix_ids) + offset - 1],
                    metadata={
                        **dict(metadata or {}),
                        "request_id": request_id,
                        "draft_runner_id": self.runner_id,
                    },
                )
            )
            token_id = _argmax(output.logits[0])
            tokens.append(token_id)
            forward_timing_ms.append(output.timing_ms)
            if output.start_ns is not None and output.end_ns is not None:
                forward_intervals_ns.append(
                    {"start_ns": output.start_ns, "end_ns": output.end_ns}
                )
            cursor = token_id

        return FakeDraftGeneration(
            tokens=tokens,
            forward_timing_ms=forward_timing_ms,
            forward_intervals_ns=forward_intervals_ns,
            metadata={
                **dict(metadata or {}),
                "request_id": request_id,
                "draft_runner_id": self.runner_id,
                "prefix_ids": list(prefix_ids),
            },
        )

    def encode(self, text: str) -> list[int]:
        """把短文本稳定映射成 fake token ids，便于测试。"""
        return [max(1, ord(char) % max(2, self.vocab_size)) for char in text[:8]] or [1]

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 转成可读字符串；仅用于 fake 调试。"""
        return " ".join(str(token_id) for token_id in token_ids)


def _argmax(values: list[float]) -> int:
    """返回 logits 最大值所在的 token id。"""
    return max(range(len(values)), key=lambda index: values[index])
