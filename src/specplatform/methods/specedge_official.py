from __future__ import annotations

"""Official SpecEdge state primitives.

These classes intentionally live beside the existing stateless SpecEdge
proposal path.  They model the official persistent tree/status/gather boundary
without changing SLED, target-only, or the current runtime loop yet.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from math import log
from typing import Any

from specplatform.core import CandidateNode, CandidateProposal, CandidateTree
from specplatform.model import TopKToken, TreeForwardInput, TreeForwardNode


class OfficialTreeStatus(IntEnum):
    """Node states used by the official SpecEdge tree implementation."""

    PROMPT = 0
    GENERATED = 5
    PROCESSED = 10
    CANDIDATE = 15
    POST_CANDIDATE = 20
    POST_PROCESSED = 25


@dataclass(frozen=True)
class OfficialDraftBeam:
    """One candidate beam selected for a graph draft step."""

    request_id: str
    node_id: int
    token_id: int
    position: int
    parent_id: int | None
    depth: int
    draft_logprob: float


@dataclass(frozen=True)
class OfficialAcceptReorder:
    """Result of compacting a slot after target verification."""

    request_id: str
    emitted_tokens: list[int]
    source_seq_indices: list[int]
    dest_seq_indices: list[int]
    bonus_token: int | None = None
    reused_proactive_tree: bool = False
    retained_tree_node_count: int = 0


@dataclass(frozen=True)
class OfficialProactiveDraftRecord:
    """Metadata for a proactive subtree appended with POST_* statuses."""

    parent_node_id: int
    root_node_id: int
    root_token_id: int
    node_ids: list[int]


@dataclass
class OfficialSpecEdgeSlot:
    """Persistent per-request tree state, matching official BatchTree semantics."""

    request_id: str
    prefix_ids: list[int]
    draft_worker_id: str
    draft_batch_index: int | None = None
    initial_prefix_len: int | None = None
    needs_prefix_tail_forward: bool = False
    nodes: list[CandidateNode] = field(default_factory=list)
    statuses: dict[int, OfficialTreeStatus] = field(default_factory=dict)
    next_node_id: int = 0
    proactive_record: OfficialProactiveDraftRecord | None = None

    def __post_init__(self) -> None:
        self.prefix_ids = [int(token_id) for token_id in self.prefix_ids]
        if not self.prefix_ids:
            raise ValueError("OfficialSpecEdgeSlot requires a non-empty prefix.")
        if self.initial_prefix_len is None:
            self.initial_prefix_len = len(self.prefix_ids)

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_ids)

    def add_root_candidates(self, topk: list[TopKToken]) -> list[int]:
        """Add initial root candidates from prefill logits."""
        return self.add_children(parent_id=None, topk=topk, depth=1, parent_logprob=0.0)

    def add_children(
        self,
        *,
        parent_id: int | None,
        topk: list[TopKToken],
        depth: int | None = None,
        parent_logprob: float | None = None,
        status: OfficialTreeStatus = OfficialTreeStatus.CANDIDATE,
    ) -> list[int]:
        """Append child nodes without budget pruning."""
        if parent_id is not None:
            parent = self.node_by_id(int(parent_id))
            depth = int(parent.depth) + 1 if depth is None else int(depth)
            parent_logprob = float(parent.draft_logprob or 0.0) if parent_logprob is None else float(parent_logprob)
        else:
            depth = 1 if depth is None else int(depth)
            parent_logprob = 0.0 if parent_logprob is None else float(parent_logprob)
        added: list[int] = []
        for candidate in topk:
            node = CandidateNode(
                node_id=self.next_node_id,
                parent_id=None if parent_id is None else int(parent_id),
                token_id=int(candidate.token_id),
                depth=int(depth),
                draft_logprob=float(parent_logprob) + float(candidate.logprob),
                draft_worker_id=self.draft_worker_id,
            )
            self.nodes.append(node)
            self.statuses[int(node.node_id)] = status
            added.append(int(node.node_id))
            self.next_node_id += 1
        self.to_candidate_tree().validate()
        return added

    def replace_tree(
        self,
        tree: CandidateTree,
        *,
        prefix_ids: list[int] | None = None,
        statuses: dict[int, OfficialTreeStatus | int] | None = None,
    ) -> None:
        """Replace the draft tree while keeping the request slot persistent."""
        if prefix_ids is not None:
            self.prefix_ids = [int(token_id) for token_id in prefix_ids]
        if tree.root_prefix_len != len(self.prefix_ids):
            raise ValueError("Official SpecEdge tree root_prefix_len must match slot prefix length.")
        tree.validate()
        self.nodes = [
            CandidateNode(
                node_id=int(node.node_id),
                parent_id=None if node.parent_id is None else int(node.parent_id),
                token_id=int(node.token_id),
                depth=int(node.depth),
                draft_logprob=node.draft_logprob,
                draft_worker_id=str(node.draft_worker_id or self.draft_worker_id),
            )
            for node in tree.nodes
        ]
        node_ids = {int(node.node_id) for node in self.nodes}
        status_map = {
            int(node_id): OfficialTreeStatus(int(status))
            for node_id, status in dict(statuses or {}).items()
            if int(node_id) in node_ids
        }
        self.statuses = {
            int(node.node_id): status_map.get(int(node.node_id), OfficialTreeStatus.CANDIDATE)
            for node in self.nodes
        }
        self.next_node_id = (max(node_ids) + 1) if node_ids else 0
        self.proactive_record = None
        self.needs_prefix_tail_forward = False

    def select_candidate_beams(self, max_beams: int) -> list[OfficialDraftBeam]:
        """Select top candidate nodes and mark them PROCESSED."""
        candidates = [
            node
            for node in self.nodes
            if self.statuses.get(int(node.node_id)) == OfficialTreeStatus.CANDIDATE
        ]
        candidates.sort(key=_node_score_key, reverse=True)
        selected = candidates[: max(0, int(max_beams))]
        for node in selected:
            self.statuses[int(node.node_id)] = OfficialTreeStatus.PROCESSED
        return [
            OfficialDraftBeam(
                request_id=self.request_id,
                node_id=int(node.node_id),
                token_id=int(node.token_id),
                position=self.position_for_node(int(node.node_id)),
                parent_id=None if node.parent_id is None else int(node.parent_id),
                depth=int(node.depth),
                draft_logprob=float(node.draft_logprob or 0.0),
            )
            for node in selected
        ]

    def add_budgeted_children(
        self,
        *,
        children_by_parent: dict[int, list[TopKToken]],
        max_budget: int,
        decay_factor: float = log(0.9),
        score_floor: float = -10.0,
    ) -> list[int]:
        """Add children using the official budget-bucket threshold rule."""
        incoming: list[dict[str, Any]] = []
        for parent_id, topk in children_by_parent.items():
            parent = self.node_by_id(int(parent_id))
            parent_logprob = float(parent.draft_logprob or 0.0)
            for candidate in topk:
                incoming.append(
                    {
                        "parent_id": int(parent_id),
                        "token_id": int(candidate.token_id),
                        "depth": int(parent.depth) + 1,
                        "draft_logprob": parent_logprob + float(decay_factor) + float(candidate.logprob),
                        "draft_worker_id": self.draft_worker_id,
                    }
                )
        kept = _official_budget_children(self.nodes, incoming, max_nodes=max_budget, score_floor=score_floor)
        added: list[int] = []
        for child in kept:
            node = CandidateNode(
                node_id=self.next_node_id,
                parent_id=int(child["parent_id"]),
                token_id=int(child["token_id"]),
                depth=int(child["depth"]),
                draft_logprob=float(child["draft_logprob"]),
                draft_worker_id=str(child["draft_worker_id"]),
            )
            self.nodes.append(node)
            self.statuses[int(node.node_id)] = OfficialTreeStatus.CANDIDATE
            added.append(int(node.node_id))
            self.next_node_id += 1
        self.trim_budget(max_budget)
        return [node_id for node_id in added if node_id in self.statuses]

    def trim_budget(self, max_budget: int) -> None:
        """Trim candidate tree to budget while preserving ancestors."""
        max_budget = int(max_budget)
        if max_budget <= 0:
            self.nodes.clear()
            self.statuses.clear()
            return
        if len(self.nodes) <= max_budget:
            return
        kept_nodes = _trim_nodes_with_ancestors(self.nodes, max_nodes=max_budget)
        kept_ids = {int(node.node_id) for node in kept_nodes}
        self.nodes = kept_nodes
        self.statuses = {
            node_id: status
            for node_id, status in self.statuses.items()
            if int(node_id) in kept_ids
        }
        if self.proactive_record is not None:
            kept_proactive_ids = [
                node_id
                for node_id in self.proactive_record.node_ids
                if int(node_id) in kept_ids
            ]
            if self.proactive_record.root_node_id not in kept_ids:
                self.proactive_record = None
            else:
                self.proactive_record = OfficialProactiveDraftRecord(
                    parent_node_id=self.proactive_record.parent_node_id,
                    root_node_id=self.proactive_record.root_node_id,
                    root_token_id=self.proactive_record.root_token_id,
                    node_ids=kept_proactive_ids,
                )
        self.to_candidate_tree().validate()

    def apply_acceptance(self, accepted_node_ids: list[int], bonus_token: int | None = None) -> OfficialAcceptReorder:
        """Compact the slot after verification, like official gather/reorder."""
        accepted_node_ids = [int(node_id) for node_id in accepted_node_ids]
        self._validate_accepted_path(accepted_node_ids)
        accepted_tokens = [int(self.node_by_id(node_id).token_id) for node_id in accepted_node_ids]
        proactive_match = self._proactive_matches_acceptance(accepted_node_ids, bonus_token)
        proactive_descendants: list[CandidateNode] = []
        if proactive_match and self.proactive_record is not None:
            children_by_parent = self.to_candidate_tree().children_by_parent()
            proactive_descendants = _descendants(self.proactive_record.root_node_id, children_by_parent)
        emitted = [*accepted_tokens]
        if bonus_token is not None:
            emitted.append(int(bonus_token))
        source_seq_indices = list(range(len(self.prefix_ids)))
        source_seq_indices.extend(self.position_for_node(node_id) for node_id in accepted_node_ids)
        if proactive_match and self.proactive_record is not None:
            source_seq_indices.append(self.position_for_node(self.proactive_record.root_node_id))
            source_seq_indices.extend(self.position_for_node(int(node.node_id)) for node in proactive_descendants)
        dest_seq_indices = list(range(len(source_seq_indices)))
        self.prefix_ids = [*self.prefix_ids, *emitted]
        if proactive_match and self.proactive_record is not None:
            self._retain_proactive_descendants(self.proactive_record.root_node_id)
            self.needs_prefix_tail_forward = False
        else:
            self.nodes.clear()
            self.statuses.clear()
            self.next_node_id = 0
            self.proactive_record = None
            self.needs_prefix_tail_forward = bool(emitted)
        return OfficialAcceptReorder(
            request_id=self.request_id,
            emitted_tokens=emitted,
            source_seq_indices=source_seq_indices,
            dest_seq_indices=dest_seq_indices,
            bonus_token=None if bonus_token is None else int(bonus_token),
            reused_proactive_tree=bool(proactive_match),
            retained_tree_node_count=len(self.nodes),
        )

    def add_proactive_subtree(
        self,
        *,
        parent_node_id: int,
        root_token_id: int,
        root_logprob: float,
        subtree: CandidateTree,
        subtree_statuses: dict[int, OfficialTreeStatus | int] | None = None,
        root_status: OfficialTreeStatus | int = OfficialTreeStatus.POST_CANDIDATE,
    ) -> OfficialProactiveDraftRecord:
        """Append a proactive tree under a leaf using official POST_* statuses."""
        parent = self.node_by_id(int(parent_node_id))
        self._discard_existing_proactive()
        root = CandidateNode(
            node_id=self.next_node_id,
            parent_id=int(parent.node_id),
            token_id=int(root_token_id),
            depth=int(parent.depth) + 1,
            draft_logprob=float(root_logprob),
            draft_worker_id=self.draft_worker_id,
        )
        self.nodes.append(root)
        self.statuses[int(root.node_id)] = OfficialTreeStatus(int(root_status))
        self.next_node_id += 1
        added_ids = [int(root.node_id)]

        subtree.validate()
        status_map = {
            int(node_id): OfficialTreeStatus(int(status))
            for node_id, status in dict(subtree_statuses or {}).items()
        }
        id_map: dict[int, int] = {}
        for source_node in subtree.nodes:
            parent_id = (
                int(root.node_id)
                if source_node.parent_id is None
                else id_map[int(source_node.parent_id)]
            )
            node = CandidateNode(
                node_id=self.next_node_id,
                parent_id=parent_id,
                token_id=int(source_node.token_id),
                depth=int(root.depth) + int(source_node.depth),
                draft_logprob=source_node.draft_logprob,
                draft_worker_id=str(source_node.draft_worker_id or self.draft_worker_id),
            )
            self.nodes.append(node)
            source_status = status_map.get(int(source_node.node_id), OfficialTreeStatus.CANDIDATE)
            self.statuses[int(node.node_id)] = _post_status_for(source_status)
            id_map[int(source_node.node_id)] = int(node.node_id)
            added_ids.append(int(node.node_id))
            self.next_node_id += 1

        self.proactive_record = OfficialProactiveDraftRecord(
            parent_node_id=int(parent.node_id),
            root_node_id=int(root.node_id),
            root_token_id=int(root.token_id),
            node_ids=added_ids,
        )
        self.to_candidate_tree().validate()
        return self.proactive_record

    def to_candidate_tree(self) -> CandidateTree:
        tree = CandidateTree(root_prefix_len=len(self.prefix_ids), nodes=list(self.nodes))
        tree.validate()
        return tree

    def to_tree_forward_input(self) -> TreeForwardInput:
        return TreeForwardInput(
            prefix_ids=list(self.prefix_ids),
            nodes=[
                TreeForwardNode(
                    node_id=int(node.node_id),
                    parent_id=None if node.parent_id is None else int(node.parent_id),
                    token_id=int(node.token_id),
                    depth=int(node.depth),
                )
                for node in self.nodes
            ],
        )

    def to_candidate_proposal(
        self,
        *,
        proposal_id: str,
        allow_bonus: bool,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateProposal:
        tree = self.to_candidate_tree()
        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=self.request_id,
            worker_id=self.draft_worker_id,
            shape="tree",
            tokens=[int(node.token_id) for node in tree.nodes],
            tree=tree,
            draft_length=len(tree.nodes),
            metadata={
                **dict(metadata or {}),
                "prefix_ids": list(self.prefix_ids),
                "allow_bonus": bool(allow_bonus),
                "official_specedge_state": True,
                "official_needs_prefix_tail_forward": bool(self.needs_prefix_tail_forward),
                "official_proactive_record": None
                if self.proactive_record is None
                else {
                    "parent_node_id": self.proactive_record.parent_node_id,
                    "root_node_id": self.proactive_record.root_node_id,
                    "root_token_id": self.proactive_record.root_token_id,
                    "node_ids": list(self.proactive_record.node_ids),
                },
                "tree_node_statuses": {str(node_id): int(status) for node_id, status in self.statuses.items()},
            },
        )

    def node_by_id(self, node_id: int) -> CandidateNode:
        for node in self.nodes:
            if int(node.node_id) == int(node_id):
                return node
        raise KeyError(f"Unknown official SpecEdge node id: {node_id}")

    def position_for_node(self, node_id: int) -> int:
        for index, node in enumerate(self.nodes):
            if int(node.node_id) == int(node_id):
                return len(self.prefix_ids) + index
        raise KeyError(f"Unknown official SpecEdge node id: {node_id}")

    def _validate_accepted_path(self, accepted_node_ids: list[int]) -> None:
        expected_parent: int | None = None
        for node_id in accepted_node_ids:
            node = self.node_by_id(node_id)
            if node.parent_id != expected_parent:
                raise ValueError("accepted_node_ids must be one contiguous root-to-leaf path.")
            expected_parent = int(node.node_id)

    def _proactive_matches_acceptance(self, accepted_node_ids: list[int], bonus_token: int | None) -> bool:
        if self.proactive_record is None or bonus_token is None or not accepted_node_ids:
            return False
        return (
            int(self.proactive_record.parent_node_id) == int(accepted_node_ids[-1])
            and int(self.proactive_record.root_token_id) == int(bonus_token)
        )

    def _retain_proactive_descendants(self, proactive_root_node_id: int) -> None:
        children_by_parent = self.to_candidate_tree().children_by_parent()
        root = self.node_by_id(int(proactive_root_node_id))
        descendants = _descendants(root.node_id, children_by_parent)
        id_map = {int(node.node_id): index for index, node in enumerate(descendants)}
        retained: list[CandidateNode] = []
        statuses: dict[int, OfficialTreeStatus] = {}
        for node in descendants:
            new_node_id = id_map[int(node.node_id)]
            retained.append(
                CandidateNode(
                    node_id=new_node_id,
                    parent_id=None if int(node.parent_id) == int(root.node_id) else id_map[int(node.parent_id)],
                    token_id=int(node.token_id),
                    depth=max(1, int(node.depth) - int(root.depth)),
                    draft_logprob=node.draft_logprob,
                    draft_worker_id=str(node.draft_worker_id),
                )
            )
            statuses[new_node_id] = _normal_status_for(self.statuses.get(int(node.node_id), OfficialTreeStatus.CANDIDATE))
        self.nodes = retained
        self.statuses = statuses
        self.next_node_id = len(retained)
        self.proactive_record = None
        self.to_candidate_tree().validate()

    def _discard_existing_proactive(self) -> None:
        if self.proactive_record is None:
            return
        proactive_ids = {int(node_id) for node_id in self.proactive_record.node_ids}
        self.nodes = [node for node in self.nodes if int(node.node_id) not in proactive_ids]
        self.statuses = {
            int(node_id): status
            for node_id, status in self.statuses.items()
            if int(node_id) not in proactive_ids
        }
        self.proactive_record = None
        self.to_candidate_tree().validate()


@dataclass
class OfficialSpecEdgeDraftState:
    """Persistent request-slot registry for official SpecEdge drafting."""

    max_batch_size: int
    draft_worker_id: str
    slots: dict[str, OfficialSpecEdgeSlot] = field(default_factory=dict)

    def add_request(
        self,
        request_id: str,
        prefix_ids: list[int],
        *,
        draft_worker_id: str | None = None,
        draft_batch_index: int | None = None,
    ) -> OfficialSpecEdgeSlot:
        if request_id in self.slots:
            raise ValueError(f"Duplicate official SpecEdge request id: {request_id}")
        if len(self.slots) >= int(self.max_batch_size):
            raise ValueError("OfficialSpecEdgeDraftState has no empty batch slot.")
        worker_id = str(draft_worker_id or self.draft_worker_id)
        batch_index = self._assign_batch_index(
            worker_id=worker_id,
            draft_batch_index=draft_batch_index,
            request_id=str(request_id),
        )
        slot = OfficialSpecEdgeSlot(
            request_id=str(request_id),
            prefix_ids=list(prefix_ids),
            draft_worker_id=worker_id,
            draft_batch_index=batch_index,
        )
        self.slots[slot.request_id] = slot
        return slot

    def ensure_request(
        self,
        request_id: str,
        prefix_ids: list[int],
        *,
        draft_worker_id: str | None = None,
        draft_batch_index: int | None = None,
    ) -> OfficialSpecEdgeSlot:
        """Return an existing slot, or recreate it if the committed prefix moved."""
        request_id = str(request_id)
        prefix = [int(token_id) for token_id in prefix_ids]
        slot = self.slots.get(request_id)
        if slot is not None and slot.prefix_ids == prefix:
            worker_id = str(draft_worker_id or slot.draft_worker_id)
            worker_changed = worker_id != str(slot.draft_worker_id)
            if draft_worker_id is not None:
                slot.draft_worker_id = worker_id
            if draft_batch_index is not None or slot.draft_batch_index is None or worker_changed:
                slot.draft_batch_index = self._assign_batch_index(
                    worker_id=worker_id,
                    draft_batch_index=draft_batch_index,
                    request_id=request_id,
                )
            return slot
        if slot is not None:
            self.remove_request(request_id)
        return self.add_request(
            request_id,
            prefix,
            draft_worker_id=draft_worker_id,
            draft_batch_index=draft_batch_index,
        )

    def remove_request(self, request_id: str) -> None:
        self.slots.pop(str(request_id), None)

    def empty_slots(self) -> int:
        return max(0, int(self.max_batch_size) - len(self.slots))

    def slot(self, request_id: str) -> OfficialSpecEdgeSlot:
        try:
            return self.slots[str(request_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown official SpecEdge request id: {request_id}") from exc

    def select_candidate_beams(self, max_beams_per_request: int) -> list[OfficialDraftBeam]:
        beams: list[OfficialDraftBeam] = []
        for slot in self.slots.values():
            beams.extend(slot.select_candidate_beams(max_beams=max_beams_per_request))
        return beams

    def _assign_batch_index(
        self,
        *,
        worker_id: str,
        draft_batch_index: int | None,
        request_id: str,
    ) -> int:
        if draft_batch_index is not None:
            batch_index = int(draft_batch_index)
            if batch_index < 0 or batch_index >= int(self.max_batch_size):
                raise ValueError(
                    f"OfficialSpecEdgeDraftState batch index {batch_index} is outside max_batch_size={self.max_batch_size}."
                )
            if batch_index in self._used_batch_indices(worker_id=worker_id, excluding_request_id=request_id):
                raise ValueError(
                    f"OfficialSpecEdgeDraftState batch index {batch_index} is already used by worker {worker_id}."
                )
            return batch_index
        used = self._used_batch_indices(worker_id=worker_id, excluding_request_id=request_id)
        for batch_index in range(int(self.max_batch_size)):
            if batch_index not in used:
                return batch_index
        raise ValueError(f"OfficialSpecEdgeDraftState has no empty batch row for worker {worker_id}.")

    def _used_batch_indices(self, *, worker_id: str, excluding_request_id: str | None = None) -> set[int]:
        used: set[int] = set()
        for request_id, slot in self.slots.items():
            if excluding_request_id is not None and str(request_id) == str(excluding_request_id):
                continue
            if str(slot.draft_worker_id) != str(worker_id):
                continue
            if slot.draft_batch_index is not None:
                used.add(int(slot.draft_batch_index))
        return used


def _official_budget_children(
    nodes: list[CandidateNode],
    children: list[dict[str, Any]],
    *,
    max_nodes: int,
    score_floor: float,
) -> list[dict[str, Any]]:
    if not children:
        return []
    if len(nodes) + len(children) <= int(max_nodes):
        return list(children)
    scores = [
        float("-inf") if node.draft_logprob is None else float(node.draft_logprob)
        for node in nodes
    ]
    scores.extend(float(child["draft_logprob"]) for child in children)
    scores.sort(reverse=True)
    threshold_index = min(max(1, int(max_nodes)), len(scores)) - 1
    threshold = max(float(scores[threshold_index]), float(score_floor))
    return [child for child in children if float(child["draft_logprob"]) >= threshold]


def _trim_nodes_with_ancestors(nodes: list[CandidateNode], *, max_nodes: int) -> list[CandidateNode]:
    if len(nodes) <= int(max_nodes):
        return list(nodes)
    nodes_by_id = {int(node.node_id): node for node in nodes}
    selected: set[int] = set()
    for node in sorted(nodes, key=_node_score_key, reverse=True):
        lineage: list[CandidateNode] = []
        current: CandidateNode | None = node
        while current is not None and int(current.node_id) not in selected:
            lineage.append(current)
            current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
        missing = [item for item in reversed(lineage) if int(item.node_id) not in selected]
        if len(selected) + len(missing) > int(max_nodes):
            continue
        selected.update(int(item.node_id) for item in missing)
        if len(selected) >= int(max_nodes):
            break
    return [node for node in nodes if int(node.node_id) in selected]


def _node_score_key(node: CandidateNode) -> tuple[float, int]:
    return (
        float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
        -int(node.node_id),
    )


def _post_status_for(status: OfficialTreeStatus) -> OfficialTreeStatus:
    if status == OfficialTreeStatus.PROCESSED:
        return OfficialTreeStatus.POST_PROCESSED
    return OfficialTreeStatus.POST_CANDIDATE


def _normal_status_for(status: OfficialTreeStatus) -> OfficialTreeStatus:
    if status == OfficialTreeStatus.POST_PROCESSED:
        return OfficialTreeStatus.PROCESSED
    if status == OfficialTreeStatus.POST_CANDIDATE:
        return OfficialTreeStatus.CANDIDATE
    return OfficialTreeStatus(status)


def _descendants(
    root_node_id: int,
    children_by_parent: dict[int | None, list[CandidateNode]],
) -> list[CandidateNode]:
    result: list[CandidateNode] = []
    frontier = list(children_by_parent.get(int(root_node_id), []))
    while frontier:
        node = frontier.pop(0)
        result.append(node)
        frontier.extend(children_by_parent.get(int(node.node_id), []))
    return result
