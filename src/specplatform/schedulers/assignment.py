from __future__ import annotations

"""scheduler assignment 兼容出口。

当前只重新导出 round-robin scheduler；后续如果有复杂 worker assignment，
仍应保持“只计划、不执行”的边界。
"""

from specplatform.schedulers.policies import BatchAssignmentPolicy, PreferredBatchAssignmentPolicy
from specplatform.schedulers.request_scheduler import RoundRobinRequestScheduler, SchedulerResources

__all__ = [
    "BatchAssignmentPolicy",
    "PreferredBatchAssignmentPolicy",
    "RoundRobinRequestScheduler",
    "SchedulerResources",
]
