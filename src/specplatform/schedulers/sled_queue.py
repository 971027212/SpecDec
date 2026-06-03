from __future__ import annotations

"""SLED-style multi-device arrival queue and static verification batch planner."""

import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class PoissonArrivalConfig:
    """Independent Poisson request arrivals for each edge device."""

    device_count: int
    arrival_rate_per_device_s: float
    duration_s: float
    seed: int = 0
    start_ms: float = 0.0
    request_prefix: str = "sled"
    draft_length: int = 1
    prompt_tokens: int = 0


@dataclass(frozen=True)
class VerificationArrival:
    """One verification request emitted by an edge device."""

    arrival_id: str
    request_id: str
    device_id: str
    arrival_ms: float
    draft_length: int = 1
    prompt_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SLEDQueueBatch:
    """A batch dispatched by the central SLED queue planner."""

    batch_id: str
    arrivals: list[VerificationArrival]
    dispatch_ms: float
    queue_wait_ms_by_request: dict[str, float]
    padded_draft_length: int
    token_slots: int
    padding_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticQueueBatchPlanner:
    """Online static batch planner with optional oldest-request timeout."""

    batch_size: int
    max_wait_ms: float | None = None
    pad_to_max_length: bool = True

    def plan(self, arrivals: list[VerificationArrival]) -> list[SLEDQueueBatch]:
        if self.batch_size <= 0:
            raise ValueError("SLED queue batch_size must be positive.")
        sorted_arrivals = sorted(arrivals, key=lambda item: (float(item.arrival_ms), item.arrival_id))
        queue: list[VerificationArrival] = []
        batches: list[SLEDQueueBatch] = []
        for arrival in sorted_arrivals:
            while queue and self._timeout_due(queue, arrival.arrival_ms):
                dispatch_ms = self._timeout_dispatch_ms(queue)
                batches.append(self._make_batch(queue[: self.batch_size], dispatch_ms, len(batches), reason="timeout"))
                del queue[: self.batch_size]
            queue.append(arrival)
            while len(queue) >= self.batch_size:
                batches.append(self._make_batch(queue[: self.batch_size], arrival.arrival_ms, len(batches), reason="batch_full"))
                del queue[: self.batch_size]
        while queue:
            dispatch_ms = self._flush_dispatch_ms(queue)
            batches.append(self._make_batch(queue[: self.batch_size], dispatch_ms, len(batches), reason="final_flush"))
            del queue[: self.batch_size]
        return batches

    def _timeout_due(self, queue: list[VerificationArrival], next_arrival_ms: float) -> bool:
        if self.max_wait_ms is None:
            return False
        return float(queue[0].arrival_ms) + float(self.max_wait_ms) <= float(next_arrival_ms)

    def _timeout_dispatch_ms(self, queue: list[VerificationArrival]) -> float:
        assert self.max_wait_ms is not None
        return float(queue[0].arrival_ms) + float(self.max_wait_ms)

    def _flush_dispatch_ms(self, queue: list[VerificationArrival]) -> float:
        if self.max_wait_ms is None:
            return max(float(item.arrival_ms) for item in queue)
        return max(
            max(float(item.arrival_ms) for item in queue),
            float(queue[0].arrival_ms) + float(self.max_wait_ms),
        )

    def _make_batch(
        self,
        arrivals: list[VerificationArrival],
        dispatch_ms: float,
        batch_index: int,
        *,
        reason: str,
    ) -> SLEDQueueBatch:
        padded = max((int(item.draft_length) for item in arrivals), default=0) if self.pad_to_max_length else 0
        token_slots = sum(padded if self.pad_to_max_length else int(item.draft_length) for item in arrivals)
        real_tokens = sum(int(item.draft_length) for item in arrivals)
        return SLEDQueueBatch(
            batch_id=f"sled-queue-batch-{batch_index}",
            arrivals=list(arrivals),
            dispatch_ms=float(dispatch_ms),
            queue_wait_ms_by_request={
                item.request_id: max(0.0, float(dispatch_ms) - float(item.arrival_ms))
                for item in arrivals
            },
            padded_draft_length=padded,
            token_slots=token_slots,
            padding_token_count=max(0, token_slots - real_tokens),
            metadata={
                "dispatch_reason": reason,
                "batch_size": len(arrivals),
                "target_batch_size": int(self.batch_size),
                "max_wait_ms": self.max_wait_ms,
                "static_batching": True,
            },
        )


def generate_poisson_arrivals(config: PoissonArrivalConfig) -> list[VerificationArrival]:
    """Generate independent exponential inter-arrival streams per device."""
    if config.device_count < 0:
        raise ValueError("device_count must be non-negative.")
    if config.duration_s < 0:
        raise ValueError("duration_s must be non-negative.")
    if config.arrival_rate_per_device_s < 0:
        raise ValueError("arrival_rate_per_device_s must be non-negative.")
    rng = random.Random(config.seed)
    arrivals: list[VerificationArrival] = []
    for device_index in range(config.device_count):
        if config.arrival_rate_per_device_s <= 0 or config.duration_s <= 0:
            continue
        device_id = f"device-{device_index}"
        t_s = 0.0
        request_index = 0
        while True:
            t_s += rng.expovariate(config.arrival_rate_per_device_s)
            if t_s > config.duration_s:
                break
            request_id = f"{config.request_prefix}-{device_index}-{request_index}"
            arrivals.append(
                VerificationArrival(
                    arrival_id=f"{device_id}:{request_index}",
                    request_id=request_id,
                    device_id=device_id,
                    arrival_ms=float(config.start_ms) + 1000.0 * t_s,
                    draft_length=max(0, int(config.draft_length)),
                    prompt_tokens=max(0, int(config.prompt_tokens)),
                    metadata={
                        "arrival_process": "poisson",
                        "arrival_rate_per_device_s": float(config.arrival_rate_per_device_s),
                        "duration_s": float(config.duration_s),
                        "seed": int(config.seed),
                    },
                )
            )
            request_index += 1
    return sorted(arrivals, key=lambda item: (float(item.arrival_ms), item.arrival_id))


def summarize_queue_batches(batches: list[SLEDQueueBatch]) -> dict[str, Any]:
    """Return queue and padding metrics used by SLED paper-style plots."""
    waits = [
        float(wait_ms)
        for batch in batches
        for wait_ms in batch.queue_wait_ms_by_request.values()
    ]
    batch_sizes = [len(batch.arrivals) for batch in batches]
    token_slots = sum(int(batch.token_slots) for batch in batches)
    padding_tokens = sum(int(batch.padding_token_count) for batch in batches)
    request_count = sum(len(batch.arrivals) for batch in batches)
    arrival_times = [
        float(arrival.arrival_ms)
        for batch in batches
        for arrival in batch.arrivals
    ]
    wall_ms = 0.0
    if arrival_times and batches:
        wall_ms = max(batch.dispatch_ms for batch in batches) - min(arrival_times)
    return {
        "request_count": request_count,
        "batch_count": len(batches),
        "avg_batch_size": None if not batch_sizes else mean(batch_sizes),
        "max_batch_size": max(batch_sizes, default=0),
        "avg_queue_wait_ms": None if not waits else mean(waits),
        "p95_queue_wait_ms": _percentile(waits, 0.95),
        "max_queue_wait_ms": max(waits, default=0.0),
        "token_slots": token_slots,
        "padding_token_count": padding_tokens,
        "padding_overhead_ratio": None if token_slots == 0 else padding_tokens / token_slots,
        "wall_ms": wall_ms,
        "throughput_requests_per_s": None if wall_ms <= 0 else request_count / (wall_ms / 1000.0),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = max(0.0, min(1.0, float(q)))
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
