"""model runner 抽象出口。

model 层只描述模型 forward/prefill/reset 的接口。真实模型加载和推理实现
会在后续步骤接入，但 generation loop 和 acceptance 决策仍不属于 model 层。
"""

from specplatform.model.base import (
    CausalLMRunner,
    LinearForwardInput,
    LinearForwardOutput,
    ModelBackendCapabilities,
    ModelForwardInput,
    ModelForwardOutput,
    ModelRunner,
    TopKToken,
    TreeForwardChoice,
    TreeForwardInput,
    TreeForwardNode,
    TreeForwardOutput,
)
from specplatform.model.kv_cache import TorchKVCache
from specplatform.model.loader import load_causal_lm_runner
from specplatform.model.qwen3_graph import (
    Qwen3GraphBackendUnavailable,
    Qwen3GraphCausalLMRunner,
    qwen3_graph_fallback_capabilities,
)
from specplatform.model.transformers import CachedTransformersCausalLMRunner, TransformersCausalLMRunner

__all__ = [
    "CausalLMRunner",
    "LinearForwardInput",
    "LinearForwardOutput",
    "ModelBackendCapabilities",
    "ModelForwardInput",
    "ModelForwardOutput",
    "ModelRunner",
    "TorchKVCache",
    "TopKToken",
    "TreeForwardChoice",
    "TreeForwardInput",
    "TreeForwardNode",
    "TreeForwardOutput",
    "CachedTransformersCausalLMRunner",
    "TransformersCausalLMRunner",
    "Qwen3GraphBackendUnavailable",
    "Qwen3GraphCausalLMRunner",
    "load_causal_lm_runner",
    "qwen3_graph_fallback_capabilities",
]
