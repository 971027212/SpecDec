from __future__ import annotations

"""target/verifier 放置配置。

target placement 是配置边界，不是 runtime 的方法分支。runtime 仍只依赖
VerifierBackend 抽象，后续真实 A100/3090 后端会藏在 backend 实现后面。
"""

from dataclasses import dataclass
from typing import Any, Literal


TargetPlacement = Literal["a100", "3090"]
DEFAULT_TARGET_PLACEMENT: TargetPlacement = "a100"
SUPPORTED_TARGET_PLACEMENTS: tuple[TargetPlacement, ...] = ("a100", "3090")


@dataclass(frozen=True)
class TargetPlacementConfig:
    """target/verifier 放置位置和 backend 元信息。"""

    placement: TargetPlacement = DEFAULT_TARGET_PLACEMENT
    backend: str | None = None
    host: str | None = None
    device: str | None = None

    @classmethod
    def from_backend_info(cls, backend_info: dict[str, Any] | None) -> "TargetPlacementConfig":
        """从 RuntimeContext.backend_info 中读取并规范化 target placement。"""
        info = dict(backend_info or {})
        return cls(
            placement=normalize_target_placement(info.get("target_placement")),
            backend=_optional_str(info.get("target_backend")),
            host=_optional_str(info.get("target_host")),
            device=_optional_str(info.get("target_device")),
        )

    def to_backend_info(self) -> dict[str, str]:
        """把配置转回可序列化的 backend_info 字典。"""
        payload = {"target_placement": self.placement}
        if self.backend is not None:
            payload["target_backend"] = self.backend
        if self.host is not None:
            payload["target_host"] = self.host
        if self.device is not None:
            payload["target_device"] = self.device
        return payload


def normalize_target_placement(value: Any | None) -> TargetPlacement:
    """校验 target placement；空值按默认 A100 处理。"""
    if value is None or value == "":
        return DEFAULT_TARGET_PLACEMENT
    placement = str(value).lower()
    if placement not in SUPPORTED_TARGET_PLACEMENTS:
        allowed = ", ".join(SUPPORTED_TARGET_PLACEMENTS)
        raise ValueError(f"Unsupported target placement: {value!r}. Expected one of: {allowed}.")
    return placement  # type: ignore[return-value]


def _optional_str(value: Any | None) -> str | None:
    """把可选配置值转换成字符串，None 保持 None。"""
    if value is None:
        return None
    return str(value)
