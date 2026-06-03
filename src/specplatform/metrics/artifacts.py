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
        "metadata",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = {name: getattr(event, name) for name in fieldnames if name != "metadata"}
            row["metadata"] = json.dumps(event.metadata, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


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


def write_tree_snapshots_jsonl(events: list[PhaseEvent], path: str | Path) -> int:
    """从 PhaseEvent metadata 中提取每轮紧凑 tree snapshot。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accept_by_proposal: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.phase != "accept.apply" or not event.proposal_id:
            continue
        metadata = dict(event.metadata or {})
        if "accepted_node_ids" in metadata or "rejected_node_ids" in metadata:
            accept_by_proposal[event.proposal_id] = {
                "accepted_node_ids": list(metadata.get("accepted_node_ids", [])),
                "rejected_node_ids": list(metadata.get("rejected_node_ids", [])),
                "accepted_count": metadata.get("accepted_count"),
                "rejected_count": metadata.get("rejected_count"),
                "has_bonus": metadata.get("has_bonus"),
            }

    rows: list[dict[str, Any]] = []
    for event in events:
        metadata = dict(event.metadata or {})
        tree_snapshot = metadata.get("tree_snapshot")
        if not isinstance(tree_snapshot, dict):
            continue
        proposal_id = str(event.proposal_id or "")
        rows.append(
            {
                "run_id": event.run_id,
                "method": event.method,
                "round": event.round,
                "request_id": event.request_id,
                "worker_id": event.worker_id,
                "batch_id": event.batch_id,
                "proposal_id": proposal_id,
                "tree": tree_snapshot,
                **accept_by_proposal.get(proposal_id, {}),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)
