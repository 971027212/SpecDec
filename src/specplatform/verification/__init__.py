"""verification backend 出口。

verification 层只验证 proposal 并返回 VerificationResult，不决定最终接受哪些
token，也不修改 GenerationSession。
"""

from specplatform.verification.base import VerifierBackend

__all__ = [
    "VerifierBackend",
]
