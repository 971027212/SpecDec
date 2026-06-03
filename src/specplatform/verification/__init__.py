"""verification backend 出口。

verification 层只验证 proposal 并返回 VerificationResult，不决定最终接受哪些
token，也不修改 GenerationSession。
"""

from specplatform.verification.base import VerifierBackend
from specplatform.verification.generation import HttpGreedyGeneratorClient
from specplatform.verification.http_client import (
    HttpLinearVerifierClient,
    HttpLinearVerifierPoolClient,
    HttpTreeVerifierClient,
    TransportProfile,
)
from specplatform.verification.linear import LinearVerifier
from specplatform.verification.schema import (
    BatchVerifyItem,
    BatchVerifyRequest,
    BatchVerifyResponse,
    BatchVerifyResultItem,
    GreedyGenerateRequest,
    GreedyGenerateResponse,
    LinearVerifyRequest,
    LinearVerifyResponse,
    TreeVerifyRequest,
    TreeVerifyResponse,
)
from specplatform.verification.tree import TreeVerifier

__all__ = [
    "BatchVerifyItem",
    "BatchVerifyRequest",
    "BatchVerifyResponse",
    "BatchVerifyResultItem",
    "GreedyGenerateRequest",
    "GreedyGenerateResponse",
    "HttpGreedyGeneratorClient",
    "HttpLinearVerifierClient",
    "HttpLinearVerifierPoolClient",
    "HttpTreeVerifierClient",
    "LinearVerifier",
    "LinearVerifyRequest",
    "LinearVerifyResponse",
    "TreeVerifier",
    "TreeVerifyRequest",
    "TreeVerifyResponse",
    "TransportProfile",
    "VerifierBackend",
]
