from __future__ import annotations

"""linear verifier 的 HTTP 数据契约。

这个文件只定义 request/response schema，不执行模型、不做 acceptance。
3090 client 和 A100 service 都使用同一份契约，避免两端字段悄悄漂移。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import CandidateTree


@dataclass(frozen=True)
class LinearVerifyRequest:
    """POST /verify_linear 的请求体。"""

    request_id: str
    proposal_id: str
    prefix_ids: list[int]
    draft_tokens: list[int]
    eos_token_ids: list[int] = field(default_factory=list)
    allow_bonus: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "prefix_ids": list(self.prefix_ids),
            "draft_tokens": list(self.draft_tokens),
            "eos_token_ids": list(self.eos_token_ids),
            "allow_bonus": bool(self.allow_bonus),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearVerifyRequest":
        """从 HTTP JSON body 构造请求对象，并做最小类型规范化。"""
        return cls(
            request_id=str(payload["request_id"]),
            proposal_id=str(payload["proposal_id"]),
            prefix_ids=[int(token_id) for token_id in payload["prefix_ids"]],
            draft_tokens=[int(token_id) for token_id in payload["draft_tokens"]],
            eos_token_ids=[int(token_id) for token_id in payload.get("eos_token_ids", [])],
            allow_bonus=bool(payload.get("allow_bonus", True)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class LinearVerifyResponse:
    """POST /verify_linear 的响应体。"""

    request_id: str
    proposal_id: str
    accepted_prefix_len: int
    verified_tokens: list[int]
    bonus_token: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "accepted_prefix_len": int(self.accepted_prefix_len),
            "verified_tokens": list(self.verified_tokens),
            "bonus_token": self.bonus_token,
            "metadata": dict(self.metadata),
            "timing": dict(self.timing),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearVerifyResponse":
        """从 HTTP JSON response 构造响应对象。"""
        bonus_token = payload.get("bonus_token")
        return cls(
            request_id=str(payload["request_id"]),
            proposal_id=str(payload["proposal_id"]),
            accepted_prefix_len=int(payload["accepted_prefix_len"]),
            verified_tokens=[int(token_id) for token_id in payload.get("verified_tokens", [])],
            bonus_token=None if bonus_token is None else int(bonus_token),
            metadata=dict(payload.get("metadata", {})),
            timing=dict(payload.get("timing", {})),
        )


@dataclass(frozen=True)
class TreeVerifyRequest:
    """POST /verify_tree 的请求体。"""

    request_id: str
    proposal_id: str
    prefix_ids: list[int]
    tree: CandidateTree
    eos_token_ids: list[int] = field(default_factory=list)
    allow_bonus: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        self.tree.validate()
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "prefix_ids": list(self.prefix_ids),
            "tree": self.tree.to_dict(),
            "eos_token_ids": list(self.eos_token_ids),
            "allow_bonus": bool(self.allow_bonus),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TreeVerifyRequest":
        """从 HTTP JSON body 构造请求对象，并校验 tree 拓扑。"""
        return cls(
            request_id=str(payload["request_id"]),
            proposal_id=str(payload["proposal_id"]),
            prefix_ids=[int(token_id) for token_id in payload["prefix_ids"]],
            tree=CandidateTree.from_dict(dict(payload["tree"])),
            eos_token_ids=[int(token_id) for token_id in payload.get("eos_token_ids", [])],
            allow_bonus=bool(payload.get("allow_bonus", True)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class TreeVerifyResponse:
    """POST /verify_tree 的响应体。"""

    request_id: str
    proposal_id: str
    accepted_node_ids: list[int]
    target_choices: list[dict[str, Any]]
    bonus_token: int | None = None
    rejected_node_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "accepted_node_ids": [int(node_id) for node_id in self.accepted_node_ids],
            "target_choices": [dict(choice) for choice in self.target_choices],
            "bonus_token": self.bonus_token,
            "rejected_node_ids": [int(node_id) for node_id in self.rejected_node_ids],
            "metadata": dict(self.metadata),
            "timing": dict(self.timing),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TreeVerifyResponse":
        """从 HTTP JSON response 构造响应对象。"""
        bonus_token = payload.get("bonus_token")
        return cls(
            request_id=str(payload["request_id"]),
            proposal_id=str(payload["proposal_id"]),
            accepted_node_ids=[int(node_id) for node_id in payload.get("accepted_node_ids", [])],
            target_choices=[dict(choice) for choice in payload.get("target_choices", [])],
            bonus_token=None if bonus_token is None else int(bonus_token),
            rejected_node_ids=[int(node_id) for node_id in payload.get("rejected_node_ids", [])],
            metadata=dict(payload.get("metadata", {})),
            timing=dict(payload.get("timing", {})),
        )


@dataclass(frozen=True)
class GreedyGenerateRequest:
    """POST /generate_greedy 的 target-only 请求体。"""

    request_id: str
    prefix_ids: list[int]
    max_new_tokens: int
    eos_token_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "prefix_ids": list(self.prefix_ids),
            "max_new_tokens": int(self.max_new_tokens),
            "eos_token_ids": list(self.eos_token_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GreedyGenerateRequest":
        """从 HTTP JSON body 构造 target-only 请求对象。"""
        return cls(
            request_id=str(payload["request_id"]),
            prefix_ids=[int(token_id) for token_id in payload["prefix_ids"]],
            max_new_tokens=int(payload["max_new_tokens"]),
            eos_token_ids=[int(token_id) for token_id in payload.get("eos_token_ids", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class GreedyGenerateResponse:
    """POST /generate_greedy 的 target-only 响应体。"""

    request_id: str
    generated_tokens: list[int]
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "generated_tokens": list(self.generated_tokens),
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
            "timing": dict(self.timing),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GreedyGenerateResponse":
        """从 HTTP JSON response 构造 target-only 响应对象。"""
        return cls(
            request_id=str(payload["request_id"]),
            generated_tokens=[int(token_id) for token_id in payload.get("generated_tokens", [])],
            stop_reason=None if payload.get("stop_reason") is None else str(payload["stop_reason"]),
            metadata=dict(payload.get("metadata", {})),
            timing=dict(payload.get("timing", {})),
        )


@dataclass(frozen=True)
class BatchVerifyItem:
    """Batch verify 中的一条请求，保留原单请求 schema。"""

    kind: str
    request: LinearVerifyRequest | TreeVerifyRequest

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "request": self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchVerifyItem":
        kind = str(payload["kind"])
        request_payload = dict(payload["request"])
        if kind == "linear":
            request: LinearVerifyRequest | TreeVerifyRequest = LinearVerifyRequest.from_dict(request_payload)
        elif kind == "tree":
            request = TreeVerifyRequest.from_dict(request_payload)
        else:
            raise ValueError(f"Unsupported batch verify item kind: {kind}")
        return cls(kind=kind, request=request)


@dataclass(frozen=True)
class BatchVerifyRequest:
    """POST /verify_*_batch 的请求体。"""

    batch_id: str
    items: list[BatchVerifyItem]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": str(self.batch_id),
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchVerifyRequest":
        return cls(
            batch_id=str(payload["batch_id"]),
            items=[BatchVerifyItem.from_dict(dict(item)) for item in payload.get("items", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class BatchVerifyResultItem:
    """Batch verify 中的一条响应，保留原单响应 schema。"""

    kind: str
    response: LinearVerifyResponse | TreeVerifyResponse

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "response": self.response.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchVerifyResultItem":
        kind = str(payload["kind"])
        response_payload = dict(payload["response"])
        if kind == "linear":
            response: LinearVerifyResponse | TreeVerifyResponse = LinearVerifyResponse.from_dict(response_payload)
        elif kind == "tree":
            response = TreeVerifyResponse.from_dict(response_payload)
        else:
            raise ValueError(f"Unsupported batch verify result kind: {kind}")
        return cls(kind=kind, response=response)


@dataclass(frozen=True)
class BatchVerifyResponse:
    """POST /verify_*_batch 的响应体。"""

    batch_id: str
    results: list[BatchVerifyResultItem]
    metadata: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": str(self.batch_id),
            "results": [result.to_dict() for result in self.results],
            "metadata": dict(self.metadata),
            "timing": dict(self.timing),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchVerifyResponse":
        return cls(
            batch_id=str(payload["batch_id"]),
            results=[BatchVerifyResultItem.from_dict(dict(result)) for result in payload.get("results", [])],
            metadata=dict(payload.get("metadata", {})),
            timing=dict(payload.get("timing", {})),
        )
