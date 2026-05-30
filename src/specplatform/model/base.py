from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelForwardInput:
    input_ids: list[int]
    position_ids: list[int] | None = None
    cache_indices: list[int] | None = None
    attention_mask: list[list[int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelForwardOutput:
    logits: list[list[float]]
    timing_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRunner(ABC):
    runner_id: str
    max_len: int

    def prefill(self, input_ids: list[int]) -> None:
        return None

    @abstractmethod
    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        raise NotImplementedError

    def gather_kv(self, src_indices: list[int], dest_indices: list[int]) -> None:
        return None

    def reset(self, request_id: str | None = None) -> None:
        return None
