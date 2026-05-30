from __future__ import annotations

"""linear verifier 的 HTTP 数据契约。

这个文件只定义 request/response schema，不执行模型、不做 acceptance。
3090 client 和 A100 service 都使用同一份契约，避免两端字段悄悄漂移。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LinearVerifyRequest:
    """POST /verify_linear 的请求体。"""

    request_id: str
    proposal_id: str
    prefix_ids: list[int]
    draft_tokens: list[int]
    eos_token_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "prefix_ids": list(self.prefix_ids),
            "draft_tokens": list(self.draft_tokens),
            "eos_token_ids": list(self.eos_token_ids),
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
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class LinearVerifyResponse:
    """POST /verify_linear 的响应体。"""

    accepted_prefix_len: int
    verified_tokens: list[int]
    bonus_token: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""
        return {
            "accepted_prefix_len": int(self.accepted_prefix_len),
            "verified_tokens": list(self.verified_tokens),
            "bonus_token": self.bonus_token,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearVerifyResponse":
        """从 HTTP JSON response 构造响应对象。"""
        bonus_token = payload.get("bonus_token")
        return cls(
            accepted_prefix_len=int(payload["accepted_prefix_len"]),
            verified_tokens=[int(token_id) for token_id in payload.get("verified_tokens", [])],
            bonus_token=None if bonus_token is None else int(bonus_token),
            metadata=dict(payload.get("metadata", {})),
        )
