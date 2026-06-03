"""method 策略出口。

methods 层只定义候选生成策略、接受策略和 planning hints，不拥有完整
request loop，也不直接写 metrics/timing。
"""

from specplatform.methods.base import (
    AcceptancePolicy,
    CandidateStrategy,
    PlanningPolicy,
    ProactiveDraftPolicy,
    ReconcilePolicy,
    ReconcileResult,
)
from specplatform.methods.dip_sd import DiPSDPlanningPolicy
from specplatform.methods.linear import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy
from specplatform.methods.sled import (
    SLEDAsyncDraftPolicy,
    SLEDAsyncReconcilePolicy,
    SLEDDynamicCandidateStrategy,
    SLEDPlanningPolicy,
)
from specplatform.methods.specedge_tree import (
    SpecEdgeOfficialAcceptancePolicy,
    SpecEdgeOfficialCandidateStrategy,
    SpecEdgeOfficialProactiveDraftPolicy,
    SpecEdgeOfficialReconcilePolicy,
    SpecEdgePipelinePlanningPolicy,
    SpecEdgeProactiveDraftPolicy,
    SpecEdgeReconcilePolicy,
    SpecEdgeTreeAcceptancePolicy,
    SpecEdgeTreeCandidateStrategy,
)
from specplatform.methods.specedge_official import (
    OfficialAcceptReorder,
    OfficialDraftBeam,
    OfficialProactiveDraftRecord,
    OfficialSpecEdgeDraftState,
    OfficialSpecEdgeSlot,
    OfficialTreeStatus,
)

__all__ = [
    "AcceptancePolicy",
    "CandidateStrategy",
    "DiPSDPlanningPolicy",
    "GreedyPrefixAcceptancePolicy",
    "LinearCandidateStrategy",
    "OfficialAcceptReorder",
    "OfficialDraftBeam",
    "OfficialProactiveDraftRecord",
    "OfficialSpecEdgeDraftState",
    "OfficialSpecEdgeSlot",
    "OfficialTreeStatus",
    "PlanningPolicy",
    "ProactiveDraftPolicy",
    "ReconcilePolicy",
    "ReconcileResult",
    "SLEDAsyncDraftPolicy",
    "SLEDAsyncReconcilePolicy",
    "SLEDDynamicCandidateStrategy",
    "SLEDPlanningPolicy",
    "SpecEdgeOfficialAcceptancePolicy",
    "SpecEdgeOfficialCandidateStrategy",
    "SpecEdgeOfficialProactiveDraftPolicy",
    "SpecEdgeOfficialReconcilePolicy",
    "SpecEdgePipelinePlanningPolicy",
    "SpecEdgeProactiveDraftPolicy",
    "SpecEdgeReconcilePolicy",
    "SpecEdgeTreeAcceptancePolicy",
    "SpecEdgeTreeCandidateStrategy",
]
