from __future__ import annotations

"""Timing 图表输出。

绘图模块只消费 PhaseEvent artifact，不参与 runtime 决策。matplotlib 延迟导入，
让没有绘图依赖的单元测试和核心 runtime import 保持轻量。
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from specplatform.core import PhaseEvent


DEFAULT_PLOT_FORMATS = ("png", "svg")
SINGLE_RESULT_CHART_NAMES = (
    "timeline_gantt",
    "compact_timeline_distribution",
    "phase_breakdown",
    "round_waterfall",
    "overlap_concurrency",
    "worker_batch_lanes",
    "network_breakdown",
    "proactive_reuse_chart",
    "http_verify_breakdown",
)
TIMING_CHART_MODES = {
    "single_result": SINGLE_RESULT_CHART_NAMES,
}
COMPACT_TIMELINE_BUCKETS = (
    ("scheduler.plan", ("scheduler.plan",)),
    ("draft.busy", ("draft.generate", "draft.proactive", "draft.reuse_proactive")),
    ("verify.batch_total", ("verify.batch_total",)),
    ("accept.apply", ("accept.apply",)),
    ("session.append", ("session.append",)),
)
MAIN_TIMELINE_PHASES = tuple(
    phase
    for _bucket, phases in COMPACT_TIMELINE_BUCKETS
    for phase in phases
)
COMPACT_TIMELINE_BUCKET_NAMES = tuple(bucket for bucket, _phases in COMPACT_TIMELINE_BUCKETS)
COMPACT_TIMELINE_PHASE_TO_BUCKET = {
    phase: bucket
    for bucket, phases in COMPACT_TIMELINE_BUCKETS
    for phase in phases
}


def write_timing_charts(
    events: list[PhaseEvent],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = DEFAULT_PLOT_FORMATS,
    mode: str = "single_result",
) -> dict[str, list[str]]:
    """写 timing 诊断图和 audit report。"""
    if mode not in TIMING_CHART_MODES:
        raise ValueError(f"Unsupported timing chart mode: {mode}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit_paths = write_timing_audit(events, output_path)

    plt = _load_pyplot()
    written: dict[str, list[str]] = {"audit": audit_paths}
    chart_builders = {
        "timeline_gantt": _build_timeline_gantt,
        "compact_timeline_distribution": _build_compact_timeline_distribution,
        "phase_breakdown": _build_phase_breakdown,
        "round_waterfall": _build_round_waterfall,
        "overlap_concurrency": _build_overlap_concurrency,
        "worker_batch_lanes": _build_worker_batch_lanes,
        "network_breakdown": _build_network_breakdown,
        "proactive_reuse_chart": _build_proactive_reuse_chart,
        "http_verify_breakdown": _build_http_verify_breakdown,
    }
    for chart_name in TIMING_CHART_MODES[mode]:
        builder = chart_builders[chart_name]
        fig = builder(plt, events)
        written[chart_name] = _save_figure(fig, output_path / chart_name, formats)
        plt.close(fig)
    return written


def write_timing_audit(events: list[PhaseEvent], output_dir: str | Path) -> list[str]:
    """只写 timing audit report，不重新渲染图表。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit = build_timing_audit(events)
    json_path = output_path / "timing_audit.json"
    text_path = output_path / "timing_audit.txt"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(_audit_text(audit), encoding="utf-8")
    return [str(json_path), str(text_path)]


def build_timing_audit(events: list[PhaseEvent]) -> dict[str, Any]:
    """检查 timing artifact 的基本一致性。"""
    warnings: list[str] = []
    bad_duration_events = [
        event.event_id
        for event in events
        if event.start_ns is not None and event.end_ns is not None and event.end_ns < event.start_ns
    ]
    if bad_duration_events:
        warnings.append(f"negative_or_reversed_duration_events={bad_duration_events}")

    rounds: list[dict[str, Any]] = []
    by_round: dict[int, list[PhaseEvent]] = defaultdict(list)
    for event in events:
        if event.round is not None:
            by_round[int(event.round)].append(event)
        if event.phase == "verify.http_total":
            residual = event.metadata.get("network_or_queue_residual_ms")
            if residual is not None and float(residual) < 0:
                warnings.append(
                    f"round={event.round} proposal={event.proposal_id} has negative network_or_queue_residual_ms={residual}"
                )

    for round_id in sorted(by_round):
        group = by_round[round_id]
        leaf_events = [
            event
            for event in group
            if event.event_scope == "system" and event.span_kind == "leaf" and _has_bounds(event)
        ]
        round_total = next((event for event in group if event.phase == "runtime.round_total"), None)
        leaf_sum_ms = sum(float(event.measured_duration_ms or 0.0) for event in leaf_events)
        leaf_window_ms = 0.0
        uncovered_ms = None
        coverage_ratio = None
        if leaf_events:
            leaf_window_ms = (
                max(int(event.end_ns) for event in leaf_events)
                - min(int(event.start_ns) for event in leaf_events)
            ) / 1_000_000
        if round_total is not None:
            total_ms = float(round_total.measured_duration_ms or 0.0)
            coverage_ratio = leaf_sum_ms / total_ms if total_ms else None
            uncovered_ms = total_ms - leaf_window_ms
        rounds.append(
            {
                "round": round_id,
                "system_leaf_count": len(leaf_events),
                "system_leaf_sum_ms": leaf_sum_ms,
                "system_leaf_window_ms": leaf_window_ms,
                "round_total_ms": None if round_total is None else float(round_total.measured_duration_ms or 0.0),
                "leaf_sum_to_round_total_ratio": coverage_ratio,
                "uncovered_wall_ms": uncovered_ms,
            }
        )

    return {
        "event_count": len(events),
        "system_leaf_count": len(
            [event for event in events if event.event_scope == "system" and event.span_kind == "leaf"]
        ),
        "system_detail_count": len(
            [event for event in events if event.event_scope == "system" and event.span_kind == "detail"]
        ),
        "warnings": warnings,
        "rounds": rounds,
    }


def _load_pyplot() -> Any:
    """导入 matplotlib pyplot，并给无 GUI 服务器设置安全 backend/cache。"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/specdec_matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "matplotlib is required for timing charts. Run this in the edge-specdec environment "
            "or install matplotlib."
        ) from exc
    return plt


def _build_timeline_gantt(plt: Any, events: list[PhaseEvent]) -> Any:
    selected = [
        event
        for event in events
        if event.event_scope == "system"
        and event.span_kind in {"setup", "leaf", "detail"}
        and _has_bounds(event)
    ]
    selected.sort(key=lambda event: (int(event.start_ns), str(event.phase), str(event.proposal_id)))
    height = max(4.0, 0.28 * len(selected) + 1.6)
    fig, ax = plt.subplots(figsize=(13, height))
    if not selected:
        ax.text(0.5, 0.5, "No timing events", ha="center", va="center")
        ax.set_axis_off()
        return fig
    base_ns = min(int(event.start_ns) for event in selected)
    colors = _phase_colors(plt)
    labels: list[str] = []
    for y, event in enumerate(selected):
        start_ms = (int(event.start_ns) - base_ns) / 1_000_000
        duration_ms = (int(event.end_ns) - int(event.start_ns)) / 1_000_000
        ax.barh(
            y,
            duration_ms,
            left=start_ms,
            color=_color_for(colors, _phase_root(event.phase)),
            alpha=0.55 if event.span_kind == "detail" else 0.9,
            edgecolor="black",
            linewidth=0.25,
        )
        round_label = "-" if event.round is None else str(event.round)
        labels.append(f"r{round_label} {event.phase} {event.proposal_id or event.batch_id or event.worker_id or ''}".strip())
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative time (ms)")
    ax.set_title("Timing timeline (system leaf/detail spans)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_phase_breakdown(plt: Any, events: list[PhaseEvent]) -> Any:
    selected = [
        event
        for event in events
        if event.event_scope == "system" and event.span_kind == "leaf"
    ]
    totals: dict[str, float] = defaultdict(float)
    for event in selected:
        totals[event.phase] += float(event.measured_duration_ms or 0.0)
    phases = sorted(totals, key=lambda phase: totals[phase])
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.45 * len(phases) + 1.4)))
    if not phases:
        ax.text(0.5, 0.5, "No system leaf events", ha="center", va="center")
        ax.set_axis_off()
        return fig
    values = [totals[phase] for phase in phases]
    ax.barh(phases, values, color="#4C78A8")
    total = sum(values)
    for index, value in enumerate(values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(value, index, f" {value:.1f} ms ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("measured duration (ms)")
    ax.set_title("System leaf phase breakdown")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_compact_timeline_distribution(plt: Any, events: list[PhaseEvent]) -> Any:
    selected = _main_leaf_events(events)
    by_round: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    round_totals: dict[int, float] = {}
    for event in selected:
        if event.round is not None:
            by_round[int(event.round)][_compact_timeline_bucket(event.phase)] += float(event.measured_duration_ms or 0.0)
    for event in events:
        if event.round is not None and event.phase == "runtime.round_total":
            round_totals[int(event.round)] = float(event.measured_duration_ms or 0.0)

    rounds = sorted(by_round)
    height = max(4.2, 0.46 * (len(rounds) + 1) + 1.7)
    fig, ax = plt.subplots(figsize=(12, height))
    if not rounds:
        ax.text(0.5, 0.5, "No main system leaf events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    colors = _phase_colors(plt)
    row_labels = [f"round {round_id}" for round_id in rounds]
    total_by_phase = {phase: 0.0 for phase in COMPACT_TIMELINE_BUCKET_NAMES}
    max_width = 0.0
    for y, round_id in enumerate(rounds):
        left = 0.0
        row_total = sum(by_round[round_id].values())
        max_width = max(max_width, row_total, round_totals.get(round_id, 0.0))
        for phase in COMPACT_TIMELINE_BUCKET_NAMES:
            value = by_round[round_id].get(phase, 0.0)
            total_by_phase[phase] += value
            if value <= 0:
                continue
            ax.barh(
                y,
                value,
                left=left,
                color=_color_for(colors, _phase_root(phase)),
                edgecolor="white",
                linewidth=0.5,
                label=phase if y == 0 else None,
            )
            left += value
        round_total = round_totals.get(round_id, row_total)
        draft_pct = _pct(by_round[round_id].get("draft.busy", 0.0), row_total)
        verify_pct = _pct(by_round[round_id].get("verify.batch_total", 0.0), row_total)
        ax.text(
            left + max_width * 0.015,
            y,
            f"wall {round_total:.1f} ms | busy {row_total:.1f} ms | draft {draft_pct:.0f}% verify {verify_pct:.0f}%",
            va="center",
            fontsize=8,
        )

    total_y = len(rounds) + 0.7
    left = 0.0
    grand_total = sum(total_by_phase.values())
    max_width = max(max_width, grand_total)
    for phase in COMPACT_TIMELINE_BUCKET_NAMES:
        value = total_by_phase[phase]
        if value <= 0:
            continue
        ax.barh(
            total_y,
            value,
            left=left,
            color=_color_for(colors, _phase_root(phase)),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
        )
        if value / grand_total >= 0.04:
            ax.text(left + value / 2, total_y, f"{_pct(value, grand_total):.0f}%", ha="center", va="center", fontsize=8)
        left += value
    ax.text(left + max_width * 0.015, total_y, f"{grand_total:.1f} ms total", va="center", fontsize=8)

    ax.set_yticks([*range(len(rounds)), total_y])
    ax.set_yticklabels([*row_labels, "total distribution"])
    ax.invert_yaxis()
    ax.set_xlim(0, max_width * 1.36 if max_width else 1.0)
    ax.set_xlabel("measured busy duration (ms)")
    ax.set_title("Compact timeline distribution (main system leaf phase buckets)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _build_round_waterfall(plt: Any, events: list[PhaseEvent]) -> Any:
    leaf_by_round: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    round_totals: dict[int, float] = {}
    for event in events:
        if event.round is None:
            continue
        round_id = int(event.round)
        if event.event_scope == "system" and event.span_kind == "leaf":
            leaf_by_round[round_id][event.phase] += float(event.measured_duration_ms or 0.0)
        if event.phase == "runtime.round_total":
            round_totals[round_id] = float(event.measured_duration_ms or 0.0)
    rounds = sorted(set(leaf_by_round) | set(round_totals))
    phases = sorted({phase for rows in leaf_by_round.values() for phase in rows})
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * len(rounds) + 1.5)))
    if not rounds:
        ax.text(0.5, 0.5, "No round timing", ha="center", va="center")
        ax.set_axis_off()
        return fig
    colors = _phase_colors(plt)
    for y, round_id in enumerate(rounds):
        left = 0.0
        for phase in phases:
            value = leaf_by_round[round_id].get(phase, 0.0)
            if value <= 0:
                continue
            ax.barh(y, value, left=left, color=_color_for(colors, _phase_root(phase)), edgecolor="white", linewidth=0.4)
            left += value
        if round_id in round_totals:
            ax.plot(round_totals[round_id], y, marker="|", markersize=14, color="black")
    ax.set_yticks(range(len(rounds)))
    ax.set_yticklabels([f"round {round_id}" for round_id in rounds])
    ax.invert_yaxis()
    ax.set_xlabel("duration (ms)")
    ax.set_title("Round waterfall (system leaf phases; black marker = round_total)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_overlap_concurrency(plt: Any, events: list[PhaseEvent]) -> Any:
    selected = [
        event
        for event in events
        if event.event_scope == "system" and event.span_kind == "leaf" and _has_bounds(event)
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if not selected:
        ax.text(0.5, 0.5, "No system leaf events", ha="center", va="center")
        ax.set_axis_off()
        return fig
    base_ns = min(int(event.start_ns) for event in selected)
    boundaries = sorted({int(event.start_ns) for event in selected} | {int(event.end_ns) for event in selected})
    phases = sorted({_phase_root(event.phase) for event in selected})
    xs: list[float] = []
    series: dict[str, list[int]] = {phase: [] for phase in phases}
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) // 2
        xs.append((left - base_ns) / 1_000_000)
        for phase in phases:
            series[phase].append(
                sum(
                    1
                    for event in selected
                    if _phase_root(event.phase) == phase
                    and int(event.start_ns) <= midpoint < int(event.end_ns)
                )
            )
    if not xs:
        ax.text(0.5, 0.5, "No overlap intervals", ha="center", va="center")
        ax.set_axis_off()
        return fig
    colors = _phase_colors(plt)
    ax.stackplot(xs, [series[phase] for phase in phases], labels=phases, colors=[_color_for(colors, phase) for phase in phases])
    ax.set_xlabel("relative time (ms)")
    ax.set_ylabel("active system leaf spans")
    ax.set_title("Overlap / concurrency view")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_worker_batch_lanes(plt: Any, events: list[PhaseEvent]) -> Any:
    lane_phases = {
        "dip_sd.solver",
        "planner.hints",
        "scheduler.plan",
        "draft.generate",
        "draft.proactive",
        "draft.reuse_proactive",
        "verify.batch_total",
        "accept.apply",
        "pipeline.planner_wait",
        "pipeline.draft_ready_wait",
        "pipeline.reconcile",
        "session.append",
    }
    selected = [
        event
        for event in events
        if event.event_scope == "system"
        and event.span_kind == "leaf"
        and _has_bounds(event)
        and event.phase in lane_phases
    ]
    selected.sort(key=lambda event: (int(event.start_ns), str(event.phase)))
    lanes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in selected:
        lanes[_pipeline_lane_label(event)].append(_lane_block_from_event(event))
    lanes = {
        lane: _merge_lane_blocks(blocks)
        for lane, blocks in lanes.items()
    }

    lane_names = sorted(
        lanes,
        key=lambda label: (
            0 if label.startswith("3090") else 1 if label.startswith("A100") else 2,
            label,
        ),
    )
    round_totals = [
        event
        for event in events
        if event.phase == "runtime.round_total" and event.span_kind == "aggregate" and _has_bounds(event)
    ]
    fig, ax = plt.subplots(figsize=(13, max(4.0, 0.72 * len(lane_names) + 1.8)))
    if not lane_names:
        ax.text(0.5, 0.5, "No worker/batch/request lane events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    base_candidates = [int(block["start_ns"]) for blocks in lanes.values() for block in blocks]
    base_candidates.extend(int(event.start_ns) for event in round_totals)
    base_ns = min(base_candidates)
    colors = _phase_colors(plt)
    for event in sorted(round_totals, key=lambda item: int(item.start_ns)):
        start_ms = (int(event.start_ns) - base_ns) / 1_000_000
        end_ms = (int(event.end_ns) - base_ns) / 1_000_000
        ax.axvspan(
            start_ms,
            end_ms,
            color=_color_for(colors, "runtime"),
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(start_ms, color="#666666", linestyle="--", linewidth=0.6, alpha=0.35)
        if event.round is not None:
            ax.text(start_ms, -0.55, f"r{event.round}", fontsize=8, color="#555555", ha="left", va="bottom")
    for y, lane in enumerate(lane_names):
        for block in lanes[lane]:
            start_ms = (int(block["start_ns"]) - base_ns) / 1_000_000
            duration_ms = (int(block["end_ns"]) - int(block["start_ns"])) / 1_000_000
            ax.barh(
                y,
                duration_ms,
                left=start_ms,
                color=_color_for(colors, _phase_root(str(block["phase"]))),
                alpha=0.92,
                edgecolor="black",
                linewidth=0.35,
                zorder=2,
            )
            label = _lane_block_label(block)
            if label and duration_ms >= 180.0:
                ax.text(
                    start_ms + duration_ms / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    clip_on=True,
                )
    ax.set_yticks(range(len(lane_names)))
    ax.set_yticklabels(lane_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("relative time (ms)")
    ax.set_title("Pipeline lanes (main leaf spans + round background)")
    ax.grid(axis="x", alpha=0.25)
    _add_phase_legend(ax, colors, ["draft", "verify", "scheduler", "accept", "session", "runtime"])
    fig.tight_layout()
    return fig


def _build_http_verify_breakdown(plt: Any, events: list[PhaseEvent]) -> Any:
    http_events = [
        event
        for event in events
        if event.phase == "verify.http_total" and event.span_kind == "detail"
    ]
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * len(http_events) + 1.5)))
    if not http_events:
        ax.text(0.5, 0.5, "No verify.http_total detail events", ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels: list[str] = []
    http_values: list[float] = []
    server_values: list[float] = []
    target_values: list[float] = []
    residual_values: list[float] = []
    for event in sorted(http_events, key=lambda item: (item.round if item.round is not None else -1, item.proposal_id or "")):
        response_timing = dict(event.metadata.get("response_timing") or {})
        http_ms = float(event.measured_duration_ms or 0.0)
        server_ms = float(response_timing.get("server_total_ms") or 0.0)
        target_ms = float(response_timing.get("target_forward_total_ms") or 0.0)
        residual_ms = max(0.0, float(event.metadata.get("network_or_queue_residual_ms") or 0.0))
        labels.append(f"r{event.round} {event.proposal_id or ''}".strip())
        http_values.append(http_ms)
        server_values.append(server_ms)
        target_values.append(target_ms)
        residual_values.append(residual_ms)
    y_positions = range(len(labels))
    height = 0.18
    offsets = [-1.5 * height, -0.5 * height, 0.5 * height, 1.5 * height]
    ax.barh([y + offsets[0] for y in y_positions], http_values, height=height, label="client http_total", color="#4C78A8")
    ax.barh([y + offsets[1] for y in y_positions], server_values, height=height, label="A100 server_total", color="#F58518")
    ax.barh([y + offsets[2] for y in y_positions], target_values, height=height, label="target forward", color="#54A24B")
    ax.barh([y + offsets[3] for y in y_positions], residual_values, height=height, label="network/queue residual", color="#E45756")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("duration (ms)")
    ax.set_title("HTTP verify breakdown")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_network_breakdown(plt: Any, events: list[PhaseEvent]) -> Any:
    """展示一次 verify HTTP 内 server、传输建模和剩余网络/排队成本。"""
    http_events = [
        event
        for event in events
        if event.phase == "verify.http_total" and event.span_kind == "detail"
    ]
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.42 * len(http_events) + 1.5)))
    if not http_events:
        ax.text(0.5, 0.5, "No verify.http_total detail events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels: list[str] = []
    rows: list[dict[str, float]] = []
    for event in sorted(http_events, key=lambda item: (item.round if item.round is not None else -1, item.proposal_id or "")):
        metadata = dict(event.metadata or {})
        timing = dict(metadata.get("response_timing") or {})
        server_ms = _server_total_ms(timing)
        upload_ms = float(metadata.get("modeled_upload_ms") or 0.0)
        downlink_ms = float(metadata.get("modeled_downlink_ms") or 0.0)
        serialize_ms = float(metadata.get("client_serialize_ms") or 0.0)
        deserialize_ms = float(metadata.get("client_deserialize_ms") or 0.0)
        residual_ms = max(0.0, float(metadata.get("network_or_queue_residual_ms") or 0.0) - upload_ms - downlink_ms)
        labels.append(f"r{event.round} {event.proposal_id or ''}".strip())
        rows.append(
            {
                "serialize": serialize_ms,
                "modeled upload": upload_ms,
                "A100 server": server_ms,
                "modeled downlink": downlink_ms,
                "deserialize": deserialize_ms,
                "residual": residual_ms,
            }
        )

    colors = {
        "serialize": "#4C78A8",
        "modeled upload": "#72B7B2",
        "A100 server": "#F58518",
        "modeled downlink": "#54A24B",
        "deserialize": "#B279A2",
        "residual": "#E45756",
    }
    y_positions = range(len(labels))
    lefts = [0.0 for _ in labels]
    for component in ["serialize", "modeled upload", "A100 server", "modeled downlink", "deserialize", "residual"]:
        values = [row[component] for row in rows]
        ax.barh(
            list(y_positions),
            values,
            left=lefts,
            label=component,
            color=colors[component],
            edgecolor="white",
            linewidth=0.35,
        )
        lefts = [left + value for left, value in zip(lefts, values)]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("duration (ms)")
    ax.set_title("Network / HTTP verify breakdown")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_proactive_reuse_chart(plt: Any, events: list[PhaseEvent]) -> Any:
    """展示 SpecEdge proactive draft 在 reconcile 后复用与丢弃的 token 数。"""
    reconcile_events = [
        event
        for event in events
        if event.phase == "pipeline.reconcile" and event.event_scope == "system"
    ]
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * len(reconcile_events) + 1.5)))
    if not reconcile_events:
        ax.text(0.5, 0.5, "No pipeline.reconcile events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels: list[str] = []
    reused: list[int] = []
    discarded: list[int] = []
    for event in sorted(reconcile_events, key=lambda item: (item.round if item.round is not None else -1, item.request_id or "")):
        metadata = dict(event.metadata or {})
        labels.append(f"r{event.round} {event.request_id or ''}".strip())
        reused.append(int(metadata.get("reused_token_count") or 0))
        discarded.append(int(metadata.get("discarded_token_count") or 0))

    y_positions = list(range(len(labels)))
    ax.barh(y_positions, reused, label="reused", color="#54A24B")
    ax.barh(y_positions, discarded, left=reused, label="discarded", color="#E45756")
    for index, (reuse_count, discard_count) in enumerate(zip(reused, discarded)):
        total = reuse_count + discard_count
        if total:
            ax.text(total + max(1.0, max(reused + discarded) * 0.02), index, f"{reuse_count}/{total}", va="center", fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("proactive draft tokens")
    ax.set_title("Proactive reuse vs discard")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _save_figure(fig: Any, path_prefix: Path, formats: tuple[str, ...]) -> list[str]:
    written: list[str] = []
    for file_format in formats:
        path = path_prefix.with_suffix(f".{file_format}")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        written.append(str(path))
    return written


def _audit_text(audit: dict[str, Any]) -> str:
    lines = [
        f"event_count: {audit['event_count']}",
        f"system_leaf_count: {audit['system_leaf_count']}",
        f"system_detail_count: {audit['system_detail_count']}",
        "warnings:",
    ]
    warnings = audit.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("rounds:")
    for row in audit.get("rounds", []):
        lines.append(
            "- round {round}: leaf_sum={system_leaf_sum_ms:.3f}ms "
            "round_total={round_total_ms} uncovered_wall={uncovered_wall_ms}".format(**row)
        )
    return "\n".join(lines) + "\n"


def _phase_colors(plt: Any) -> dict[str, Any]:
    palette = plt.get_cmap("tab10").colors
    names = [
        "setup",
        "scheduler",
        "draft",
        "verify",
        "accept",
        "session",
        "pipeline",
        "planner",
        "dip_sd",
        "request",
        "runtime",
        "artifact",
        "plot",
    ]
    return {name: palette[index % len(palette)] for index, name in enumerate(names)}


def _color_for(colors: dict[str, Any], phase: str) -> Any:
    return colors.get(phase, "#9D9D9D")


def _phase_root(phase: str) -> str:
    return phase.split(".", maxsplit=1)[0]


def _has_bounds(event: PhaseEvent) -> bool:
    return event.start_ns is not None and event.end_ns is not None


def _server_total_ms(timing: dict[str, Any]) -> float:
    for key in ("server_batch_total_ms", "server_total_ms", "server_batch_verify_ms", "verify_total_ms"):
        value = timing.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _main_leaf_events(events: list[PhaseEvent]) -> list[PhaseEvent]:
    """选择主推理阶段，排除 detail/aggregate/attribution/setup/artifact/plot。"""
    return [
        event
        for event in events
        if event.event_scope == "system"
        and event.span_kind == "leaf"
        and event.phase in MAIN_TIMELINE_PHASES
    ]


def _compact_timeline_bucket(phase: str) -> str:
    """Map detailed runtime phases into compact timeline buckets."""
    return COMPACT_TIMELINE_PHASE_TO_BUCKET.get(phase, phase)


def _pct(value: float, total: float) -> float:
    return 100.0 * float(value) / float(total) if total else 0.0


def _lane_label(event: PhaseEvent) -> str:
    """把事件归到 worker/batch/request 泳道。"""
    worker = event.worker_id or event.draft_worker_id
    if worker:
        return f"worker:{worker}"
    if event.batch_id or event.phase.startswith("verify."):
        return f"batch:{event.batch_id or event.request_id or 'unbatched'}"
    if event.request_id:
        return f"request:{event.request_id}"
    return "system"


def _pipeline_lane_label(event: PhaseEvent) -> str:
    """把主阶段事件归到少数几条可读 pipeline 泳道。"""
    worker = event.worker_id or event.draft_worker_id or "draft-worker"
    if event.phase.startswith("draft."):
        return f"3090 draft worker ({worker})"
    if event.phase == "verify.batch_total" or event.phase.startswith("verify."):
        return "A100 verify batch"
    return "request/control"


def _lane_block_from_event(event: PhaseEvent) -> dict[str, Any]:
    return {
        "phase": event.phase,
        "round": event.round,
        "start_ns": int(event.start_ns or 0),
        "end_ns": int(event.end_ns or 0),
        "count": 1,
        "metadata": dict(event.metadata or {}),
    }


def _merge_lane_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一轮同一阶段的相邻小块，让泳道图表达 wave 而不是 request 碎片。"""
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda block: (int(block["start_ns"]), str(block["phase"])))
    merged: list[dict[str, Any]] = []
    max_gap_ns = 5_000_000
    for block in sorted_blocks:
        if (
            merged
            and merged[-1]["phase"] == block["phase"]
            and merged[-1]["round"] == block["round"]
            and int(block["start_ns"]) - int(merged[-1]["end_ns"]) <= max_gap_ns
        ):
            merged[-1]["end_ns"] = max(int(merged[-1]["end_ns"]), int(block["end_ns"]))
            merged[-1]["count"] = int(merged[-1]["count"]) + int(block["count"])
            merged[-1]["metadata"] = _merge_block_metadata(
                dict(merged[-1].get("metadata") or {}),
                dict(block.get("metadata") or {}),
            )
            continue
        merged.append(dict(block))
    return merged


def _merge_block_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key == "proposal_ids" and isinstance(value, list):
            existing = merged.get(key)
            if isinstance(existing, list):
                merged[key] = [*existing, *value]
            else:
                merged[key] = list(value)
        elif key not in merged:
            merged[key] = value
    return merged


def _lane_bar_label(event: PhaseEvent) -> str:
    """返回泳道条内部的短标签；窄条会在调用方被跳过。"""
    labels = {
        "scheduler.plan": "plan",
        "draft.generate": "draft",
        "draft.proactive": "proactive",
        "draft.reuse_proactive": "reuse",
        "accept.apply": "accept",
        "pipeline.reconcile": "reconcile",
        "session.append": "append",
    }
    if event.phase == "verify.batch_total":
        proposal_ids = event.metadata.get("proposal_ids")
        batch_size = len(proposal_ids) if isinstance(proposal_ids, list) else event.metadata.get("batch_size")
        return f"verify n={batch_size}" if batch_size else "verify"
    return labels.get(event.phase, event.phase.split(".", maxsplit=1)[-1])


def _lane_block_label(block: dict[str, Any]) -> str:
    phase = str(block["phase"])
    count = int(block.get("count") or 1)
    labels = {
        "scheduler.plan": "plan",
        "draft.generate": "draft",
        "draft.proactive": "proactive",
        "draft.reuse_proactive": "reuse",
        "accept.apply": "accept",
        "pipeline.reconcile": "reconcile",
        "session.append": "append",
    }
    if phase == "verify.batch_total":
        metadata = dict(block.get("metadata") or {})
        proposal_ids = metadata.get("proposal_ids")
        batch_size = len(proposal_ids) if isinstance(proposal_ids, list) else metadata.get("batch_size")
        return f"verify n={batch_size}" if batch_size else "verify"
    label = labels.get(phase, phase.split(".", maxsplit=1)[-1])
    if phase.startswith("draft.") and count > 1:
        return f"{label} n={count}"
    return label


def _add_phase_legend(ax: Any, colors: dict[str, Any], phases: list[str]) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=_color_for(colors, phase), edgecolor="black", label=phase) for phase in phases]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(handles), 6),
        fontsize=8,
        frameon=True,
    )
