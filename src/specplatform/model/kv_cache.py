from __future__ import annotations

"""通用 Torch KV cache 容器。

这个模块先提供平台级 KV cache 边界和 gather/reorder 语义；Qwen3 graph
backend 后续会把具体模型 forward 接到这个边界上。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TorchKVCache:
    """按 [layer, batch, kv_head, seq, head_dim] 存储的 KV cache。"""

    num_layers: int
    batch_size: int
    num_key_value_heads: int
    max_len: int
    head_dim: int
    device: str
    dtype: Any

    def __post_init__(self) -> None:
        """分配 K/V cache 张量。"""
        import torch

        shape = (
            int(self.num_layers),
            int(self.batch_size),
            int(self.num_key_value_heads),
            int(self.max_len),
            int(self.head_dim),
        )
        self.k_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)

    def clear(self, batch_indices: Any | None = None) -> None:
        """清空全部或指定 batch 的 KV cache。"""
        if batch_indices is None:
            self.k_cache.zero_()
            self.v_cache.zero_()
            return
        batch_indices = self._tensor_indices(batch_indices)
        self.k_cache[:, batch_indices, ...].zero_()
        self.v_cache[:, batch_indices, ...].zero_()

    def gather(self, batch_idx: int, src_indices: Any, dest_indices: Any) -> None:
        """把同一 batch 内的 seq 位置从 src 重排到 dest。"""
        src = self._tensor_indices(src_indices)
        dest = self._tensor_indices(dest_indices)
        if src.numel() != dest.numel():
            raise ValueError("src_indices and dest_indices must have the same length.")
        if src.numel() == 0:
            return
        if src.dtype == self._torch().bool:
            src = src.nonzero(as_tuple=True)[0]
        self.k_cache[:, batch_idx, :, dest, :] = self.k_cache[:, batch_idx, :, src, :]
        self.v_cache[:, batch_idx, :, dest, :] = self.v_cache[:, batch_idx, :, src, :]
        tail_start = int(dest.max().item()) + 1
        if tail_start < self.max_len:
            self.k_cache[:, batch_idx, :, tail_start:, :].zero_()
            self.v_cache[:, batch_idx, :, tail_start:, :].zero_()

    def update(
        self,
        *,
        layer_idx: int,
        batch_indices: Any,
        seq_indices: Any,
        key_states: Any,
        value_states: Any,
    ) -> None:
        """写入一层的 K/V states。"""
        batch = self._tensor_indices(batch_indices)
        seq = self._tensor_indices(seq_indices)
        self.k_cache[layer_idx, batch, :, seq, :] = key_states
        self.v_cache[layer_idx, batch, :, seq, :] = value_states

    def _tensor_indices(self, indices: Any) -> Any:
        """把 Python/list/tensor 索引规范到 cache device 上的 long tensor。"""
        torch = self._torch()
        if hasattr(indices, "to"):
            return indices.to(device=self.k_cache.device)
        return torch.tensor(indices, device=self.k_cache.device, dtype=torch.long)

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch
