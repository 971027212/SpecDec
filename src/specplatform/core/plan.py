from __future__ import annotations

"""scheduler 输出的可执行计划数据模型。

这里描述 draft job、verify batch 和 planning hint，但不执行任何模型调用。
scheduler 负责创建这些对象，runtime 负责解释并执行它们。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DraftBudget:
    """单个 draft job 允许使用的 draft token/分支预算。"""

    max_tokens: int
    max_branches: int = 1
    timeout_ms: float | None = None


@dataclass(frozen=True)
class DraftJob:
    """把一个 request 分配给一个 draft worker 的执行单元。"""

    request_id: str
    worker_id: str
    budget: DraftBudget
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyBatch:
    """一次 verifier batch 计划。

    scheduler 只决定哪些 request 同批验证；proposal_ids 由 runtime 在 draft
    完成后回填。
    """

    batch_id: str
    request_ids: list[str]
    proposal_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanHints:
    """method/planning policy 给 scheduler 的非强制提示。"""

    draft_lengths: dict[str, int] = field(default_factory=dict)
    preferred_batches: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutablePlan:
    """runtime 可以执行的一轮计划。"""

    draft_jobs: list[DraftJob]
    verify_batches: list[VerifyBatch]
    hints: PlanHints | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
