from __future__ import annotations

"""Latency calibration helpers for DiP-SD artifacts.

The DiP-SD paper relies on affine latency models fitted from profiling.  This
module keeps that fitting logic inside the DiP-SD package so experiment
artifacts can report how the paper model lines up with the current platform
run without leaking method-specific assumptions into SpecEdge or SLED.
"""

from dataclasses import asdict
from math import sqrt
import json
from pathlib import Path
from typing import Any

from specplatform.core import PhaseEvent
from specplatform.methods.dip_sd.model import (
    DiPSDModelConfig,
    draft_compute_intensity,
    verify_compute_intensity,
)


def calibration_from_events(
    events: list[PhaseEvent],
    *,
    model_config: DiPSDModelConfig | None = None,
) -> dict[str, Any]:
    config = model_config or DiPSDModelConfig()
    observations = _observations_from_events(events, config)
    fit_rows = _fit_rows(observations)
    return {
        "model_config": asdict(config),
        "observation_count": len(observations),
        "observations": observations,
        "fits": fit_rows,
        "recommended_method_config": _recommended_method_config(fit_rows),
    }


def recommended_method_config_from_profile(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("recommended_method_config")
    if not isinstance(raw, dict):
        raise ValueError(f"DiP-SD calibration profile has no recommended_method_config: {path}")
    result: dict[str, float] = {}
    for key in ("dip_sd_draft_c", "dip_sd_draft_beta", "dip_sd_verify_c", "dip_sd_verify_beta"):
        value = raw.get(key)
        if value is not None:
            result[key] = float(value)
    if not result:
        raise ValueError(f"DiP-SD calibration profile has no usable recommended values: {path}")
    return result


def _observations_from_events(events: list[PhaseEvent], config: DiPSDModelConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_verify_events: set[str] = set()
    for event in events:
        metadata = dict(event.metadata or {})
        if event.phase == "draft.generate":
            rows.extend(_draft_observations(event, metadata, config))
        elif event.phase == "verify.http_total":
            rows.extend(_verify_forward_observations(event, metadata, config, seen_verify_events))
        elif event.phase == "verify.batch_total":
            rows.append(_verify_batch_observation(event, metadata, config))
    return rows


def _draft_observations(
    event: PhaseEvent,
    metadata: dict[str, Any],
    config: DiPSDModelConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_events = [
        dict(item)
        for item in metadata.get("draft_token_forward_events", [])
        if isinstance(item, dict)
    ]
    if token_events:
        for index, token_event in enumerate(token_events):
            prefix_len = _optional_int(token_event.get("prefix_len"))
            duration_ms = _optional_float(token_event.get("duration_ms"))
            if prefix_len is None or duration_ms is None:
                continue
            rows.append(
                _draft_row(
                    event,
                    metadata,
                    config,
                    prefix_len=prefix_len,
                    duration_ms=duration_ms,
                    token_count=1,
                    source="draft.token_forward",
                    source_index=index,
                )
            )
        return rows

    prefix_ids = metadata.get("prefix_ids") if isinstance(metadata.get("prefix_ids"), list) else []
    prefix_len = len(prefix_ids)
    duration_ms = _optional_float(event.measured_duration_ms or event.duration_ms)
    token_count = _optional_int(metadata.get("max_tokens")) or _optional_int(metadata.get("draft_length")) or 1
    if duration_ms is None:
        return rows
    rows.append(
        _draft_row(
            event,
            metadata,
            config,
            prefix_len=prefix_len,
            duration_ms=duration_ms / max(1, token_count),
            token_count=token_count,
            source="draft.generate_average",
            source_index=0,
        )
    )
    return rows


def _draft_row(
    event: PhaseEvent,
    metadata: dict[str, Any],
    config: DiPSDModelConfig,
    *,
    prefix_len: int,
    duration_ms: float,
    token_count: int,
    source: str,
    source_index: int,
) -> dict[str, Any]:
    x_value = draft_compute_intensity(prefix_len, config)
    return {
        "kind": "draft",
        "source": source,
        "source_index": source_index,
        "round_id": event.round,
        "worker_id": event.worker_id or metadata.get("worker_id") or metadata.get("runner_id"),
        "request_id": event.request_id or metadata.get("request_id"),
        "batch_id": event.batch_id,
        "prefix_len": prefix_len,
        "draft_len": 1,
        "batch_size": 1,
        "token_count": int(token_count),
        "x_value": float(x_value),
        "duration_ms": float(duration_ms),
        "model_path": metadata.get("model_path"),
        "backend": metadata.get("backend"),
        "device": metadata.get("device"),
    }


def _verify_forward_observations(
    event: PhaseEvent,
    metadata: dict[str, Any],
    config: DiPSDModelConfig,
    seen: set[str],
) -> list[dict[str, Any]]:
    timing = dict(metadata.get("response_timing") or {})
    rows: list[dict[str, Any]] = []
    for index, forward_event in enumerate(timing.get("target_forward_events") or []):
        if not isinstance(forward_event, dict):
            continue
        kind = str(forward_event.get("kind") or "")
        if "linear" not in kind:
            continue
        event_key = str(
            forward_event.get("shared_batch_event_id")
            or (
                forward_event.get("start_ns"),
                forward_event.get("end_ns"),
                forward_event.get("batch_index"),
                kind,
            )
        )
        if event_key in seen:
            continue
        seen.add(event_key)
        duration_ms = _optional_float(forward_event.get("shared_duration_ms") or forward_event.get("duration_ms"))
        if duration_ms is None:
            continue
        batch_size = _optional_int(forward_event.get("batch_size")) or _optional_int(timing.get("batch_size")) or 1
        draft_len = (
            _optional_int(forward_event.get("verified_token_count"))
            or _optional_int(forward_event.get("draft_token_count"))
            or 1
        )
        prefix_len = _optional_int(forward_event.get("prefix_len"))
        if prefix_len is None:
            graph_prefill = dict(forward_event.get("metadata") or {}).get("graph_prefill_token_count")
            prefix_len = _optional_int(graph_prefill) or 0
        rows.append(
            _verify_row(
                event,
                config,
                source="verify.target_forward",
                source_index=index,
                batch_size=batch_size,
                draft_len=draft_len,
                prefix_len=prefix_len,
                duration_ms=duration_ms,
                backend=metadata.get("backend_name"),
            )
        )
    return rows


def _verify_batch_observation(
    event: PhaseEvent,
    metadata: dict[str, Any],
    config: DiPSDModelConfig,
) -> dict[str, Any]:
    request_ids = metadata.get("request_ids") if isinstance(metadata.get("request_ids"), list) else []
    return _verify_row(
        event,
        config,
        source="verify.batch_total",
        source_index=0,
        batch_size=max(1, len(request_ids)),
        draft_len=_optional_int(metadata.get("max_draft_len")) or 1,
        prefix_len=_optional_int(metadata.get("max_prefix_len")) or 0,
        duration_ms=float(event.measured_duration_ms or event.duration_ms or 0.0),
        backend=metadata.get("backend_name"),
    )


def _verify_row(
    event: PhaseEvent,
    config: DiPSDModelConfig,
    *,
    source: str,
    source_index: int,
    batch_size: int,
    draft_len: int,
    prefix_len: int,
    duration_ms: float,
    backend: Any,
) -> dict[str, Any]:
    x_value = float(batch_size) * verify_compute_intensity(draft_len, prefix_len, config)
    return {
        "kind": "verify",
        "source": source,
        "source_index": source_index,
        "round_id": event.round,
        "worker_id": None,
        "request_id": event.request_id,
        "batch_id": event.batch_id,
        "prefix_len": int(prefix_len),
        "draft_len": int(draft_len),
        "batch_size": int(batch_size),
        "token_count": int(batch_size) * int(draft_len),
        "x_value": x_value,
        "duration_ms": float(duration_ms),
        "model_path": None,
        "backend": backend,
        "device": None,
    }


def _fit_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in observations:
        kind = str(row.get("kind") or "unknown")
        group = str(row.get("worker_id") or "global") if kind == "draft" else str(row.get("source") or "verify")
        groups.setdefault((kind, group), []).append(row)
        if kind == "draft" and group != "global":
            groups.setdefault((kind, "global"), []).append(row)
    fits: list[dict[str, Any]] = []
    for (kind, group), rows in sorted(groups.items()):
        fit = _fit_affine([float(row["x_value"]) for row in rows], [float(row["duration_ms"]) for row in rows])
        fits.append(
            {
                "kind": kind,
                "group": group,
                "observation_count": len(rows),
                "c": fit["slope"],
                "beta": fit["intercept"],
                "mae_ms": fit["mae"],
                "rmse_ms": fit["rmse"],
                "r2": fit["r2"],
                "fit_status": fit["status"],
                "x_min": fit["x_min"],
                "x_max": fit["x_max"],
                "duration_min_ms": fit["y_min"],
                "duration_max_ms": fit["y_max"],
            }
        )
    return fits


def _fit_affine(xs: list[float], ys: list[float]) -> dict[str, Any]:
    n = len(xs)
    if n == 0:
        return _empty_fit("no_observations")
    mean_y = sum(ys) / n
    if n < 2 or max(xs) == min(xs):
        slope = 0.0
        intercept = mean_y
        status = "underdetermined_constant"
    else:
        mean_x = sum(xs) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        status = "least_squares"
        if slope < 0.0:
            slope = 0.0
            intercept = mean_y
            status = "least_squares_negative_slope_clamped"
        elif intercept < 0.0:
            slope = sum(x * y for x, y in zip(xs, ys)) / max(1e-12, sum(x * x for x in xs))
            intercept = 0.0
            status = "least_squares_negative_intercept_zeroed"
    preds = [slope * x + intercept for x in xs]
    abs_errors = [abs(y - pred) for y, pred in zip(ys, preds)]
    squared_errors = [(y - pred) ** 2 for y, pred in zip(ys, preds)]
    total_var = sum((y - mean_y) ** 2 for y in ys)
    residual = sum(squared_errors)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "mae": float(sum(abs_errors) / n),
        "rmse": float(sqrt(residual / n)),
        "r2": None if total_var <= 0 else float(1.0 - residual / total_var),
        "status": status,
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
        "y_min": float(min(ys)),
        "y_max": float(max(ys)),
    }


def _empty_fit(status: str) -> dict[str, Any]:
    return {
        "slope": None,
        "intercept": None,
        "mae": None,
        "rmse": None,
        "r2": None,
        "status": status,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    }


def _recommended_method_config(fits: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    draft_global = next((fit for fit in fits if fit.get("kind") == "draft" and fit.get("group") == "global"), None)
    draft_any = next((fit for fit in fits if fit.get("kind") == "draft"), None)
    verify_forward = next((fit for fit in fits if fit.get("kind") == "verify" and fit.get("group") == "verify.target_forward"), None)
    verify_any = next((fit for fit in fits if fit.get("kind") == "verify"), None)
    draft_fit = draft_global or draft_any
    verify_fit = verify_forward or verify_any
    if draft_fit and draft_fit.get("c") is not None:
        result["dip_sd_draft_c"] = draft_fit.get("c")
        result["dip_sd_draft_beta"] = draft_fit.get("beta")
    if verify_fit and verify_fit.get("c") is not None:
        result["dip_sd_verify_c"] = verify_fit.get("c")
        result["dip_sd_verify_beta"] = verify_fit.get("beta")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
