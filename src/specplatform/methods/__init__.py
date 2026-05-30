"""method 策略出口。

methods 层只定义候选生成策略、接受策略和 planning hints，不拥有完整
request loop，也不直接写 metrics/timing。
"""

from specplatform.methods.base import AcceptancePolicy, CandidateStrategy, PlanningPolicy

__all__ = [
    "AcceptancePolicy",
    "CandidateStrategy",
    "PlanningPolicy",
]
