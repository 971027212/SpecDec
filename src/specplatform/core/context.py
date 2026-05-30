from __future__ import annotations

"""运行上下文数据。

RuntimeContext 只携带配置、随机种子、时钟和 backend 元信息。它故意不暴露
engine/verifier/metrics recorder，避免 method 或 core 绕过统一 runtime 边界。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from specplatform.core.target import TargetPlacementConfig


Clock = Callable[[], int]
DebugLogger = Callable[[str], None]


@dataclass(frozen=True)
class RuntimeContext:
    """一次运行共享的只读上下文。"""

    method_config: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    tokenizer: Any | None = None
    seed: int | None = None
    clock: Clock = time.perf_counter_ns
    debug_logger: DebugLogger | None = None
    backend_info: dict[str, Any] = field(default_factory=dict)

    @property
    def target_placement(self) -> TargetPlacementConfig:
        """把 backend_info 规范成 target/verifier 放置配置。"""
        return TargetPlacementConfig.from_backend_info(self.backend_info)
