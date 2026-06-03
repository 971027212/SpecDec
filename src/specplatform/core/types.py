from __future__ import annotations

"""跨模块共享的基础类型。

这里放候选树节点和 PhaseEvent 等公共结构；它们只承载数据，不触发模型调用、
调度或指标写盘。
"""

from dataclasses import asdict, dataclass, field
from time import time_ns
from typing import Any


@dataclass(frozen=True)
class CandidateNode:
    """候选树中的一个 draft token 节点。"""

    node_id: int
    parent_id: int | None
    token_id: int
    depth: int
    draft_logprob: float | None
    draft_worker_id: str

    def to_dict(self) -> dict[str, Any]:
        """转换成 tree verify schema 可直接序列化的节点字典。"""
        return {
            "node_id": int(self.node_id),
            "parent_id": None if self.parent_id is None else int(self.parent_id),
            "token_id": int(self.token_id),
            "depth": int(self.depth),
            "draft_logprob": None if self.draft_logprob is None else float(self.draft_logprob),
            "draft_worker_id": str(self.draft_worker_id),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateNode":
        """从 JSON 友好的节点字典恢复 CandidateNode。"""
        parent_id = payload.get("parent_id")
        draft_logprob = payload.get("draft_logprob")
        return cls(
            node_id=int(payload["node_id"]),
            parent_id=None if parent_id is None else int(parent_id),
            token_id=int(payload["token_id"]),
            depth=int(payload["depth"]),
            draft_logprob=None if draft_logprob is None else float(draft_logprob),
            draft_worker_id=str(payload.get("draft_worker_id", "")),
        )


@dataclass
class CandidateTree:
    """树形 speculative decoding 候选结构。

    当前最小闭环先走 linear proposal；这个结构为后续 tree draft 保留边界。
    """

    root_prefix_len: int
    nodes: list[CandidateNode] = field(default_factory=list)

    def validate(self) -> None:
        """校验候选树拓扑：节点唯一、父节点先出现、深度连续。"""
        if self.root_prefix_len < 1:
            raise ValueError("CandidateTree root_prefix_len must be positive.")
        seen: dict[int, CandidateNode] = {}
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError(f"Duplicate candidate node id: {node.node_id}")
            if node.depth < 1:
                raise ValueError(f"Candidate node depth must be positive: {node.node_id}")
            if node.parent_id is None:
                if node.depth != 1:
                    raise ValueError(f"Root child depth must be 1: {node.node_id}")
            else:
                parent = seen.get(node.parent_id)
                if parent is None:
                    raise ValueError(f"Parent must appear before child: {node.node_id}")
                if node.depth != parent.depth + 1:
                    raise ValueError(f"Candidate node depth must follow parent depth: {node.node_id}")
            seen[node.node_id] = node

    def to_dict(self) -> dict[str, Any]:
        """转换成 tree verify schema 可直接序列化的字典。"""
        return {
            "root_prefix_len": int(self.root_prefix_len),
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateTree":
        """从 JSON 友好的字典恢复 CandidateTree，并校验拓扑。"""
        tree = cls(
            root_prefix_len=int(payload["root_prefix_len"]),
            nodes=[CandidateNode.from_dict(node) for node in payload.get("nodes", [])],
        )
        tree.validate()
        return tree

    def nodes_by_id(self) -> dict[int, CandidateNode]:
        """返回 node_id 到节点的映射。"""
        self.validate()
        return {node.node_id: node for node in self.nodes}

    def children_by_parent(self) -> dict[int | None, list[CandidateNode]]:
        """按 parent_id 分组，保留节点原始顺序。"""
        self.validate()
        grouped: dict[int | None, list[CandidateNode]] = {}
        for node in self.nodes:
            grouped.setdefault(node.parent_id, []).append(node)
        return grouped


@dataclass
class PhaseEvent:
    """metrics/timing 输出的统一事件结构。"""

    run_id: str
    request_id: str
    method: str
    phase: str
    duration_ms: float | None = None
    event_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    plan_id: str | None = None
    phase_category: str | None = None
    event_scope: str = "system"
    span_kind: str = "leaf"
    measured_duration_ms: float | None = None
    attributed_duration_ms: float | None = None
    session_id: str | None = None
    worker_id: str | None = None
    batch_id: str | None = None
    proposal_id: str | None = None
    shared: bool = False
    attribution: str | None = None
    device: str | None = None
    draft_worker_id: str | None = None
    target_backend: str | None = None
    round: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    timestamp_ns: int = field(default_factory=time_ns)
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """补齐派生字段，并校验时间区间合法。"""
        if self.phase_category is None:
            self.phase_category = _phase_category(self.phase)
        if self.duration_ms is None and self.attributed_duration_ms is None:
            raise ValueError("PhaseEvent requires duration_ms or attributed_duration_ms.")
        if self.attributed_duration_ms is None:
            self.attributed_duration_ms = float(self.duration_ms or 0.0)
        if self.duration_ms is None:
            self.duration_ms = float(self.attributed_duration_ms)
        if self.measured_duration_ms is None:
            self.measured_duration_ms = float(self.attributed_duration_ms)
        if self.start_ns is None and self.end_ns is None:
            return
        if self.start_ns is None:
            self.start_ns = self.end_ns - int(round(float(self.measured_duration_ms) * 1_000_000))
        if self.end_ns is None:
            self.end_ns = self.start_ns + int(round(float(self.measured_duration_ms) * 1_000_000))
        if self.end_ns < self.start_ns:
            raise ValueError("Phase end_ns cannot be earlier than start_ns")

    def to_dict(self) -> dict[str, Any]:
        """转成 artifact writer 可直接序列化的字典。"""
        return asdict(self)


def _phase_category(phase: str) -> str:
    """根据 phase 前缀推导粗粒度类别。"""
    prefix = phase.split(".", maxsplit=1)[0]
    known = {
        "runtime",
        "scheduler",
        "draft",
        "verify",
        "accept",
        "session",
        "request",
        "setup",
        "artifact",
        "plot",
    }
    return prefix if prefix in known else "runtime"
