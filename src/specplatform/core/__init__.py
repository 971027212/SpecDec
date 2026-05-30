"""core 公共数据模型出口。

core 只定义跨模块共享的数据结构，不执行 request loop、不调用 verifier、
不记录 timing/metrics 副作用。
"""

from specplatform.core.candidate import CandidateProposal, ProposalShape
from specplatform.core.context import RuntimeContext
from specplatform.core.plan import DraftBudget, DraftJob, ExecutablePlan, PlanHints, VerifyBatch
from specplatform.core.result import AcceptResult, VerificationResult
from specplatform.core.target import (
    DEFAULT_TARGET_PLACEMENT,
    SUPPORTED_TARGET_PLACEMENTS,
    TargetPlacement,
    TargetPlacementConfig,
    normalize_target_placement,
)
from specplatform.core.types import CandidateNode, CandidateTree, PhaseEvent

__all__ = [
    "AcceptResult",
    "CandidateNode",
    "CandidateProposal",
    "CandidateTree",
    "DEFAULT_TARGET_PLACEMENT",
    "DraftBudget",
    "DraftJob",
    "ExecutablePlan",
    "PhaseEvent",
    "PlanHints",
    "ProposalShape",
    "RuntimeContext",
    "SUPPORTED_TARGET_PLACEMENTS",
    "TargetPlacement",
    "TargetPlacementConfig",
    "VerificationResult",
    "VerifyBatch",
    "normalize_target_placement",
]
