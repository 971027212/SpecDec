from __future__ import annotations

"""模型 runner 的最小抽象。

后续真实 Torch/HTTP/Graph backend 都应该适配到这个边界附近；runtime 和
method 不应该直接依赖具体深度学习框架。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelForwardInput:
    """一次模型 forward 的输入。"""

    input_ids: list[int]
    position_ids: list[int] | None = None
    cache_indices: list[int] | None = None
    attention_mask: list[list[int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelForwardOutput:
    """一次模型 forward 的输出。"""

    logits: list[list[float]]
    timing_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRunner(ABC):
    """模型 runner 抽象基类。"""

    runner_id: str
    max_len: int

    def prefill(self, input_ids: list[int]) -> None:
        """可选 prefill 钩子；fake runner 当前不需要缓存初始化。"""
        return None

    @abstractmethod
    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """对输入 token 执行一次 forward，返回 logits。"""
        raise NotImplementedError

    def gather_kv(self, src_indices: list[int], dest_indices: list[int]) -> None:
        """可选 KV cache 重排钩子；真实 backend 后续再实现。"""
        return None

    def reset(self, request_id: str | None = None) -> None:
        """可选状态清理钩子；fake runner 当前无内部状态。"""
        return None
