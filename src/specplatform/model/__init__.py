"""model runner 抽象出口。

model 层只描述模型 forward/prefill/reset 的接口。真实模型加载和推理实现
会在后续步骤接入，但 generation loop 和 acceptance 决策仍不属于 model 层。
"""

from specplatform.model.base import CausalLMRunner, ModelForwardInput, ModelForwardOutput, ModelRunner

__all__ = [
    "CausalLMRunner",
    "ModelForwardInput",
    "ModelForwardOutput",
    "ModelRunner",
]
