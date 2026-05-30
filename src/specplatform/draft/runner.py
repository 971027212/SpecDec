from __future__ import annotations

"""Phase 1 的 fake draft runner。

当前实现继承 deterministic fake model，仅提供 encode/decode 便利方法。
Step 2 会在这个边界附近补“一次生成多个 draft tokens”的能力。
"""

from dataclasses import dataclass

from specplatform.model.fake import FakeDeterministicModelRunner


@dataclass
class FakeDraftRunner(FakeDeterministicModelRunner):
    """用于测试 draft 阶段的 fake runner。"""

    def encode(self, text: str) -> list[int]:
        """把短文本稳定映射成 fake token ids，便于测试。"""
        return [max(1, ord(char) % max(2, self.vocab_size)) for char in text[:8]] or [1]

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 转成可读字符串；仅用于 fake 调试。"""
        return " ".join(str(token_id) for token_id in token_ids)
