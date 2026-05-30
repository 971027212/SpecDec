from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TargetPlacement = Literal["a100", "3090"]
DEFAULT_TARGET_PLACEMENT: TargetPlacement = "a100"
SUPPORTED_TARGET_PLACEMENTS: tuple[TargetPlacement, ...] = ("a100", "3090")


@dataclass(frozen=True)
class TargetPlacementConfig:
    placement: TargetPlacement = DEFAULT_TARGET_PLACEMENT
    backend: str | None = None
    host: str | None = None
    device: str | None = None

    @classmethod
    def from_backend_info(cls, backend_info: dict[str, Any] | None) -> "TargetPlacementConfig":
        info = dict(backend_info or {})
        return cls(
            placement=normalize_target_placement(info.get("target_placement")),
            backend=_optional_str(info.get("target_backend")),
            host=_optional_str(info.get("target_host")),
            device=_optional_str(info.get("target_device")),
        )

    def to_backend_info(self) -> dict[str, str]:
        payload = {"target_placement": self.placement}
        if self.backend is not None:
            payload["target_backend"] = self.backend
        if self.host is not None:
            payload["target_host"] = self.host
        if self.device is not None:
            payload["target_device"] = self.device
        return payload


def normalize_target_placement(value: Any | None) -> TargetPlacement:
    if value is None or value == "":
        return DEFAULT_TARGET_PLACEMENT
    placement = str(value).lower()
    if placement not in SUPPORTED_TARGET_PLACEMENTS:
        allowed = ", ".join(SUPPORTED_TARGET_PLACEMENTS)
        raise ValueError(f"Unsupported target placement: {value!r}. Expected one of: {allowed}.")
    return placement  # type: ignore[return-value]


def _optional_str(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value)
