from __future__ import annotations

"""Shared planning math for edge-side scheduling policies.

This module intentionally contains method-neutral helpers only.  SLED and
DiP-SD can both use resource metadata, depth clamping and simple ready-time
batching without depending on each other's method packages.
"""

from math import ceil
from typing import Any

from specplatform.core import RuntimeContext


def clamp_depth(value: int | float, *, min_depth: int, max_depth: int, remaining: Any = None) -> int:
    depth = max(int(min_depth), min(int(max_depth), int(round(value))))
    if remaining is not None:
        depth = min(depth, max(0, int(remaining)))
    return depth


def timing_config(context: RuntimeContext, default_server_ms: float, default_network_ms: float) -> dict[str, float]:
    return {
        "server_verify_ms": float(context.method_config.get("estimated_server_verify_ms", default_server_ms)),
        "network_residual_ms": float(context.method_config.get("estimated_network_residual_ms", default_network_ms)),
        "server_batch_per_request_ms": float(context.method_config.get("estimated_server_batch_per_request_ms", 0.0)),
        "server_batch_per_token_ms": float(context.method_config.get("estimated_server_batch_per_token_ms", 0.0)),
    }


def assignment_score_and_tail(
    ready_time_ms: dict[str, float],
    draft_lengths: dict[str, int],
    expected_accept_tokens: dict[str, float],
    *,
    batch_size: int,
    timing: dict[str, float],
) -> tuple[float, float]:
    batches = batches_by_ready_time(list(ready_time_ms), batch_size=batch_size, ready_time_ms=ready_time_ms)
    server_available_ms = 0.0
    tail_ms = 0.0
    network_ms = max(0.0, timing["network_residual_ms"])
    for batch in batches:
        if not batch:
            continue
        batch_ready_ms = max(ready_time_ms[request_id] for request_id in batch)
        server_start_ms = max(server_available_ms, batch_ready_ms)
        server_available_ms = server_start_ms + estimated_server_batch_ms(batch, draft_lengths, timing)
        tail_ms = max(tail_ms, server_available_ms + network_ms)
    expected_tokens = sum(max(0.01, expected_accept_tokens.get(request_id, 0.0)) for request_id in ready_time_ms)
    return tail_ms / max(expected_tokens, 0.01), tail_ms


def candidate_batch_sizes(
    *,
    request_count: int,
    worker_count: int,
    configured_batch_size: int,
    context: RuntimeContext,
    key_prefix: str,
) -> list[int]:
    if request_count <= 0:
        return [1]
    raw = context.method_config.get(f"{key_prefix}_candidate_batch_sizes")
    if raw:
        values = [
            int(part)
            for part in (raw if isinstance(raw, list) else str(raw).split(","))
            if str(part).strip()
        ]
    else:
        max_batches = int(context.method_config.get(f"{key_prefix}_max_batch_count", request_count) or request_count)
        values = [
            ceil(request_count / batch_count)
            for batch_count in range(1, max(1, min(request_count, max_batches)) + 1)
        ]
        values.extend([configured_batch_size, worker_count, ceil(request_count / 2), request_count])
    return sorted({max(1, min(request_count, int(value))) for value in values if int(value) > 0})


def estimated_server_batch_ms(
    batch: list[str],
    draft_lengths: dict[str, int],
    timing: dict[str, float],
) -> float:
    return (
        float(timing["server_verify_ms"])
        + float(timing.get("server_batch_per_request_ms") or 0.0) * max(0, len(batch) - 1)
        + float(timing.get("server_batch_per_token_ms") or 0.0)
        * sum(max(0, int(draft_lengths.get(request_id, 0))) for request_id in batch)
    )


def expected_accept_tokens(depth: int, acceptance_rate: float) -> float:
    depth = max(0, int(depth))
    q = max(0.0, min(float(acceptance_rate), 1.0))
    if depth <= 0:
        return 1.0
    if q >= 0.999:
        return float(depth)
    return sum(q**index for index in range(1, depth + 1))


def batch_size(
    context: RuntimeContext,
    key: str,
    configured: int | None,
    *,
    request_count: int,
    worker_count: int,
) -> int:
    raw = int(context.method_config.get(key, configured or 0) or 0)
    if raw > 0:
        return max(1, min(request_count, raw))
    return max(1, min(request_count, max(worker_count, ceil(request_count / 2))))


def batches_by_ready_time(
    request_ids: list[str],
    *,
    batch_size: int,
    ready_time_ms: dict[str, float],
) -> list[list[str]]:
    ordered = sorted(request_ids, key=lambda request_id: ready_time_ms.get(request_id, 0.0))
    return [
        ordered[index : index + max(1, batch_size)]
        for index in range(0, len(ordered), max(1, batch_size))
    ]


def resource_worker_metadata(resources: Any) -> dict[str, dict[str, Any]]:
    if isinstance(resources, dict):
        raw = resources.get("draft_worker_metadata", {})
    else:
        raw = getattr(resources, "draft_worker_metadata", {})
    if not isinstance(raw, dict):
        return {}
    return {str(worker_id): dict(metadata or {}) for worker_id, metadata in raw.items()}


def speed_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    profile = metadata.get("speed_profile")
    if isinstance(profile, dict):
        return dict(profile)
    config = metadata.get("config")
    if isinstance(config, dict):
        profile = config.get("speed_profile")
        if isinstance(profile, dict):
            return dict(profile)
    return {}


def worker_acceptance_rate(
    metadata: dict[str, Any],
    *,
    context: RuntimeContext,
    default_rate: float,
    key_prefix: str,
) -> float:
    profile = speed_profile(metadata)
    for key in ("expected_acceptance", "acceptance_rate", "quality"):
        if profile.get(key) is not None:
            return float(profile[key])
    return float(context.method_config.get(f"{key_prefix}_alpha", default_rate))


def worker_ms_per_token(metadata: dict[str, Any], *, default_ms: float) -> float:
    profile = speed_profile(metadata)
    tokens_per_second = profile.get("tokens_per_second")
    if tokens_per_second:
        return 1000.0 / max(float(tokens_per_second), 1e-6)
    relative_speed = float(profile.get("relative_speed") or 1.0)
    return float(default_ms) / max(relative_speed, 1e-6)


def worker_latency_ms(metadata: dict[str, Any]) -> float:
    profile = speed_profile(metadata)
    return float(profile.get("latency_ms") or 0.0)


def resource_worker_ids(resources: Any) -> list[str]:
    if isinstance(resources, dict):
        return [str(worker_id) for worker_id in resources.get("draft_worker_ids", [])]
    return [str(worker_id) for worker_id in getattr(resources, "draft_worker_ids", [])]


# Backwards-compatible private aliases for tests and transitional imports.
_assignment_score_and_tail = assignment_score_and_tail
_batch_size = batch_size
_batches_by_ready_time = batches_by_ready_time
_clamp_depth = clamp_depth
_resource_worker_ids = resource_worker_ids
_resource_worker_metadata = resource_worker_metadata
_speed_profile = speed_profile
_worker_latency_ms = worker_latency_ms
_worker_ms_per_token = worker_ms_per_token
