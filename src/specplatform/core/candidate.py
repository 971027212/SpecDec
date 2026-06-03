from __future__ import annotations

"""候选 token/proposal 数据模型。

CandidateProposal 是 draft 阶段和 verification 阶段之间的契约：
draft/method 负责构造它，verifier 只消费它，core 本身不生成也不验证 token。
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from specplatform.core.types import CandidateTree


ProposalShape = Literal["linear", "tree"]


@dataclass(frozen=True)
class CandidateProposal:
    """一次 draft 候选结果。

    linear proposal 用 tokens 表示一条候选链；tree proposal 用 CandidateTree
    表示多分支候选。当前最小闭环先使用 linear。
    """

    proposal_id: str
    request_id: str
    worker_id: str | None
    shape: ProposalShape
    tokens: list[int] = field(default_factory=list)
    tree: CandidateTree | None = None
    draft_length: int = 0
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 proposal shape 和承载字段一致，避免下游猜测数据形态。"""
        if self.shape == "linear" and self.tree is not None:
            raise ValueError("linear proposals cannot carry a candidate tree.")
        if self.shape == "tree" and self.tree is None:
            raise ValueError("tree proposals require a candidate tree.")
        if self.shape == "tree" and self.tree is not None:
            self.tree.validate()
