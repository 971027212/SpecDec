from __future__ import annotations

"""PhaseEvent 记录器。

EventLogger 只做轻量校验和收集，真正的耗时测量由 timing 模块提供。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from specplatform.core import PhaseEvent


@dataclass
class EventLogger:
    """一次 runtime run 中的内存事件缓冲区。"""

    events: list[PhaseEvent] = field(default_factory=list)

    def record(self, event: PhaseEvent) -> None:
        """记录单个 PhaseEvent，并拒绝负 duration。"""
        if event.duration_ms < 0:
            raise ValueError("Phase duration cannot be negative")
        self.events.append(event)

    def write_jsonl(self, path: str | Path) -> None:
        """把所有事件写成 JSONL。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
