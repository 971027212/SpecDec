from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from specplatform.core import PhaseEvent


@dataclass
class EventLogger:
    events: list[PhaseEvent] = field(default_factory=list)

    def record(self, event: PhaseEvent) -> None:
        if event.duration_ms < 0:
            raise ValueError("Phase duration cannot be negative")
        self.events.append(event)

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

