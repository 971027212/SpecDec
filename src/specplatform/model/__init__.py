"""model runner 抽象出口。

model 层只描述模型 forward/prefill/reset 的接口和 fake 实现，不拥有
generation loop，也不决定 speculative acceptance。
"""

from specplatform.model.base import ModelForwardInput, ModelForwardOutput, ModelRunner
from specplatform.model.fake import FakeDeterministicModelRunner

__all__ = [
    "FakeDeterministicModelRunner",
    "ModelForwardInput",
    "ModelForwardOutput",
    "ModelRunner",
]
