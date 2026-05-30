from __future__ import annotations

"""verifier backend 抽象。

真实 HTTP/Torch/A100/3090 verifier 后续都应该实现这个接口。runtime 只依赖
VerifierBackend，不感知后端部署位置。
"""

from abc import ABC, abstractmethod

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult


class VerifierBackend(ABC):
    """proposal verifier 抽象基类。"""

    backend_name: str

    @abstractmethod
    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """验证单个 proposal；只返回事实结果，不写 session。"""
        raise NotImplementedError

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        """默认逐个验证 proposal；真实后端可覆盖成批量实现。"""
        return [self.verify_proposal(proposal, context) for proposal in proposals]
