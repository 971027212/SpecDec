from __future__ import annotations

"""Model runner loader with backend selection."""

from typing import Literal

from specplatform.model.qwen3_graph import load_qwen3_graph_or_fallback
from specplatform.model.transformers import CachedTransformersCausalLMRunner, TransformersCausalLMRunner


ModelBackendName = Literal["hf_eager", "eager", "hf_cached", "cached", "qwen3_graph", "graph"]


def load_causal_lm_runner(
    model_path: str,
    *,
    runner_id: str,
    backend: str = "hf_eager",
    device: str | None = "cuda",
    torch_dtype: str | None = "auto",
    device_map: str | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    allow_fallback: bool = True,
    max_graph_len: int | None = None,
    max_graph_tokens: int | None = None,
    max_graph_batch_size: int | None = None,
) -> TransformersCausalLMRunner:
    """Load a CausalLMRunner using a named backend."""
    normalized = str(backend or "hf_eager").lower()
    if normalized in {"hf_eager", "eager"}:
        return TransformersCausalLMRunner.from_pretrained(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
    if normalized in {"hf_cached", "cached"}:
        return CachedTransformersCausalLMRunner.from_pretrained(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
    if normalized in {"qwen3_graph", "graph"}:
        return load_qwen3_graph_or_fallback(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            allow_fallback=allow_fallback,
            max_graph_len=max_graph_len,
            max_graph_tokens=max_graph_tokens,
            max_graph_batch_size=max_graph_batch_size,
        )
    raise ValueError(f"Unsupported model backend: {backend}")
