"""verification backend 出口。

verification 层只验证 proposal 并返回 VerificationResult，不决定最终接受哪些
token，也不修改 GenerationSession。
"""

from specplatform.verification.base import VerifierBackend
from specplatform.verification.http_client import HttpLinearVerifierClient
from specplatform.verification.linear import LinearVerifier
from specplatform.verification.schema import LinearVerifyRequest, LinearVerifyResponse

__all__ = [
    "HttpLinearVerifierClient",
    "LinearVerifier",
    "LinearVerifyRequest",
    "LinearVerifyResponse",
    "VerifierBackend",
]
