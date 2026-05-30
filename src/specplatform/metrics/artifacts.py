from __future__ import annotations

"""metrics artifact 写盘工具。

这里把 runtime 产生的 PhaseEvent / request result 写成 CSV/JSON，不反向影响
runtime 或 method 的决策。
"""

import csv
import json
from pathlib import Path
from typing import Any

from specplatform.core import PhaseEvent
from specplatform.timing.summary import summarize_timing_events


def write_phase_events_csv(events: list[PhaseEvent], path: str | Path) -> None:
    """写逐事件明细 CSV。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "span_id",
        "parent_span_id",
        "run_id",
        "method",
        "plan_id",
        "phase",
        "phase_category",
        "event_scope",
        "span_kind",
        "request_id",
        "session_id",
        "worker_id",
        "batch_id",
        "proposal_id",
        "shared",
        "attribution",
        "round",
        "tokens_in",
        "tokens_out",
        "start_ns",
        "end_ns",
        "measured_duration_ms",
        "attributed_duration_ms",
        "duration_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({name: getattr(event, name) for name in fieldnames})


def write_phase_summary_csv(events: list[PhaseEvent], path: str | Path) -> None:
    """写按视图聚合后的 phase summary CSV。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "summary_view",
            "method",
            "phase",
            "phase_category",
            "event_scope",
            "span_kind",
            "count",
            "total_measured_duration_ms",
            "total_attributed_duration_ms",
            "mean_measured_duration_ms",
            "mean_attributed_duration_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summarize_timing_events(events):
            writer.writerow(row.to_dict())


def write_request_results_json(results: list[Any], path: str | Path) -> None:
    """写每个 request 的输出 token/proposal 摘要 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "request_id": result.request_id,
            "output_token_ids": list(result.output_token_ids),
            "proposals": list(result.proposals),
            "stop_reason": result.stop_reason,
        }
        for result in results
    ]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
