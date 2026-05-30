from __future__ import annotations

from dataclasses import dataclass

from specplatform.model.fake import FakeDeterministicModelRunner


@dataclass
class FakeDraftRunner(FakeDeterministicModelRunner):
    def encode(self, text: str) -> list[int]:
        return [max(1, ord(char) % max(2, self.vocab_size)) for char in text[:8]] or [1]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)
