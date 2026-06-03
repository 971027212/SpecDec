from __future__ import annotations

"""DiP-SD paper cost model.

The formulas here mirror the optimization model in the paper: distributed
local drafting, central batch verification, communication delay, pipeline span
and edge-server KV memory feasibility.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiPSDUserParams:
    request_id: str
    prefix_len: int
    acceptance: float
    comm_latency_ms: float
    draft_c: float
    draft_beta: float
    remaining_tokens: int | None = None
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiPSDModelConfig:
    draft_blocks: int = 28
    draft_hidden: int = 2048
    draft_ffn_hidden: int = 6144
    verify_blocks: int = 64
    verify_hidden: int = 5120
    verify_ffn_hidden: int = 25600
    verify_c: float = 1.2077e-11
    verify_beta: float = 95.1074
    memory_cap_bytes: float = 8.0e10


@dataclass(frozen=True)
class DiPSDPaperDefaults:
    user_count: int = 6
    prefix_len: int = 512
    acceptance: float = 0.78
    comm_latency_ms: float = 3.0
    draft_c: float = 4.0305e-11
    draft_beta: float = 33.8151
    initial_draft_length: int = 7
    max_draft_length: int = 20


@dataclass(frozen=True)
class DiPSDBatchMetrics:
    stage_index: int
    request_ids: tuple[str, ...]
    batch_size: int
    max_draft_len: int
    max_prefix_len: int
    draft_complete_ms: float
    verify_ms: float
    stage_duration_ms: float
    memory_bytes: float
    memory_feasible: bool


@dataclass(frozen=True)
class DiPSDScheduleEvaluation:
    feasible: bool
    throughput_tokens_per_ms: float
    expected_tokens: float
    pipeline_span_ms: float
    verification_sum_ms: float
    max_draft_verify_ms: float
    batch_metrics: tuple[DiPSDBatchMetrics, ...]
    reason: str | None = None


def expected_accepted_tokens(draft_length: int, acceptance: float) -> float:
    """Paper utility u_m(l_m), including the guaranteed bonus token."""
    length = max(0, int(draft_length))
    alpha = max(0.0, min(float(acceptance), 1.0))
    if alpha >= 0.999999:
        return float(length + 1)
    return (1.0 - alpha ** (length + 1)) / max(1e-12, 1.0 - alpha)


def draft_compute_intensity(prefix_len: int, config: DiPSDModelConfig) -> float:
    i = max(0, int(prefix_len))
    return 4.0 * config.draft_blocks * config.draft_hidden * (
        2.0 * config.draft_hidden + i + 1.0 + config.draft_ffn_hidden
    )


def verify_compute_intensity(max_draft_len: int, max_prefix_len: int, config: DiPSDModelConfig) -> float:
    length = max(0, int(max_draft_len))
    prefix = max(0, int(max_prefix_len))
    return 4.0 * config.verify_blocks * config.verify_hidden * length * (
        2.0 * config.verify_hidden + prefix + length + config.verify_ffn_hidden
    )


def affine_latency_ms(batch_size: int, compute_intensity: float, c: float, beta: float) -> float:
    return float(c) * max(1, int(batch_size)) * float(compute_intensity) + float(beta)


def draft_latency_ms(user: DiPSDUserParams, draft_length: int, config: DiPSDModelConfig) -> float:
    per_token_ms = affine_latency_ms(
        1,
        draft_compute_intensity(user.prefix_len, config),
        user.draft_c,
        user.draft_beta,
    )
    return max(0, int(draft_length)) * per_token_ms


def verify_latency_ms(
    *,
    batch_size: int,
    max_draft_len: int,
    max_prefix_len: int,
    config: DiPSDModelConfig,
) -> float:
    return affine_latency_ms(
        batch_size,
        verify_compute_intensity(max_draft_len, max_prefix_len, config),
        config.verify_c,
        config.verify_beta,
    )


def verify_parameter_memory_bytes(config: DiPSDModelConfig) -> float:
    return config.verify_blocks * (
        8.0 * (config.verify_hidden**2)
        + 4.0 * config.verify_hidden * config.verify_ffn_hidden
    )


def verify_kv_memory_bytes(*, batch_size: int, max_prefix_len: int, config: DiPSDModelConfig) -> float:
    return 4.0 * config.verify_blocks * config.verify_hidden * max(1, int(batch_size)) * max(0, int(max_prefix_len))


def verify_memory_bytes(*, batch_size: int, max_prefix_len: int, config: DiPSDModelConfig) -> float:
    return verify_parameter_memory_bytes(config) + verify_kv_memory_bytes(
        batch_size=batch_size,
        max_prefix_len=max_prefix_len,
        config=config,
    )


def evaluate_schedule(
    *,
    users: list[DiPSDUserParams],
    batches: list[list[str]],
    draft_lengths: dict[str, int],
    config: DiPSDModelConfig,
) -> DiPSDScheduleEvaluation:
    users_by_id = {user.request_id: user for user in users}
    if not users:
        return DiPSDScheduleEvaluation(
            feasible=False,
            throughput_tokens_per_ms=0.0,
            expected_tokens=0.0,
            pipeline_span_ms=0.0,
            verification_sum_ms=0.0,
            max_draft_verify_ms=0.0,
            batch_metrics=(),
            reason="no_users",
        )
    if any(not batch for batch in batches):
        return _infeasible("empty_batch")

    metrics: list[DiPSDBatchMetrics] = []
    expected_tokens = 0.0
    verification_sum_ms = 0.0
    max_draft_verify_ms = 0.0
    for stage_index, batch in enumerate(batches):
        try:
            batch_users = [users_by_id[request_id] for request_id in batch]
        except KeyError as exc:
            return _infeasible(f"unknown_request_id:{exc.args[0]}")
        lengths = [max(1, int(draft_lengths.get(user.request_id, 1))) for user in batch_users]
        max_draft_len = max(lengths)
        max_prefix_len = max(max(0, int(user.prefix_len)) for user in batch_users)
        draft_complete_ms = max(
            draft_latency_ms(user, length, config) + max(0.0, float(user.comm_latency_ms))
            for user, length in zip(batch_users, lengths)
        )
        verify_ms = verify_latency_ms(
            batch_size=len(batch_users),
            max_draft_len=max_draft_len,
            max_prefix_len=max_prefix_len,
            config=config,
        )
        memory_bytes = verify_memory_bytes(
            batch_size=len(batch_users),
            max_prefix_len=max_prefix_len,
            config=config,
        )
        memory_feasible = memory_bytes <= float(config.memory_cap_bytes)
        verification_sum_ms += verify_ms
        max_draft_verify_ms = max(max_draft_verify_ms, draft_complete_ms + verify_ms)
        for user, length in zip(batch_users, lengths):
            expected_tokens += expected_accepted_tokens(length, user.acceptance)
        metrics.append(
            DiPSDBatchMetrics(
                stage_index=stage_index,
                request_ids=tuple(batch),
                batch_size=len(batch_users),
                max_draft_len=max_draft_len,
                max_prefix_len=max_prefix_len,
                draft_complete_ms=draft_complete_ms,
                verify_ms=verify_ms,
                stage_duration_ms=verify_ms,
                memory_bytes=memory_bytes,
                memory_feasible=memory_feasible,
            )
        )

    if not all(metric.memory_feasible for metric in metrics):
        return DiPSDScheduleEvaluation(
            feasible=False,
            throughput_tokens_per_ms=0.0,
            expected_tokens=expected_tokens,
            pipeline_span_ms=0.0,
            verification_sum_ms=verification_sum_ms,
            max_draft_verify_ms=max_draft_verify_ms,
            batch_metrics=tuple(metrics),
            reason="memory_infeasible",
        )

    span_ms = max(verification_sum_ms, max_draft_verify_ms)
    throughput = expected_tokens / max(span_ms, 1e-12)
    return DiPSDScheduleEvaluation(
        feasible=True,
        throughput_tokens_per_ms=throughput,
        expected_tokens=expected_tokens,
        pipeline_span_ms=span_ms,
        verification_sum_ms=verification_sum_ms,
        max_draft_verify_ms=max_draft_verify_ms,
        batch_metrics=tuple(metrics),
    )


def _infeasible(reason: str) -> DiPSDScheduleEvaluation:
    return DiPSDScheduleEvaluation(
        feasible=False,
        throughput_tokens_per_ms=0.0,
        expected_tokens=0.0,
        pipeline_span_ms=0.0,
        verification_sum_ms=0.0,
        max_draft_verify_ms=0.0,
        batch_metrics=(),
        reason=reason,
    )
