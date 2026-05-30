from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from specplatform.core.target import TargetPlacementConfig


Clock = Callable[[], int]
DebugLogger = Callable[[str], None]


@dataclass(frozen=True)
class RuntimeContext:
    method_config: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    tokenizer: Any | None = None
    seed: int | None = None
    clock: Clock = time.perf_counter_ns
    debug_logger: DebugLogger | None = None
    backend_info: dict[str, Any] = field(default_factory=dict)

    @property
    def target_placement(self) -> TargetPlacementConfig:
        return TargetPlacementConfig.from_backend_info(self.backend_info)
