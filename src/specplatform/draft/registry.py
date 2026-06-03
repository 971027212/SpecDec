from __future__ import annotations

"""Configurable draft worker registry.

The registry is the boundary between experiment configuration and runtime
execution.  It owns model loading and runner construction; runtimes still only
consume ``dict[worker_id, runner]``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from specplatform.draft.runner import GreedyDraftRunner, TopKTreeDraftRunner
from specplatform.model import CausalLMRunner, load_causal_lm_runner


DraftModelLoader = Callable[..., CausalLMRunner]


@dataclass(frozen=True)
class DraftSpeedProfile:
    """Optional scheduling metadata for a draft worker."""

    name: str = "observed"
    tokens_per_second: float | None = None
    latency_ms: float | None = None
    relative_speed: float = 1.0
    quality: float | None = None
    expected_acceptance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: Any) -> "DraftSpeedProfile":
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(name=raw)
        if not isinstance(raw, dict):
            raise TypeError("draft speed_profile must be a mapping or string.")
        return cls(
            name=str(raw.get("name") or "configured"),
            tokens_per_second=_optional_float(raw.get("tokens_per_second")),
            latency_ms=_optional_float(raw.get("latency_ms")),
            relative_speed=float(raw.get("relative_speed", 1.0)),
            quality=_optional_float(raw.get("quality", raw.get("acceptance_rate"))),
            expected_acceptance=_optional_float(raw.get("expected_acceptance")),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tokens_per_second": self.tokens_per_second,
            "latency_ms": self.latency_ms,
            "relative_speed": self.relative_speed,
            "quality": self.quality,
            "expected_acceptance": self.expected_acceptance,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DraftWorkerConfig:
    """One independently configurable draft worker."""

    worker_id: str
    model_path: str
    draft_type: str = "tree"
    device: str | None = "cuda"
    backend: str = "hf_eager"
    torch_dtype: str | None = "auto"
    device_map: str | None = None
    trust_remote_code: bool = True
    allow_fallback: bool = True
    max_graph_len: int | None = None
    max_graph_tokens: int | None = None
    max_graph_batch_size: int | None = None
    speed_profile: DraftSpeedProfile = field(default_factory=DraftSpeedProfile)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any],
        *,
        index: int,
        defaults: dict[str, Any],
    ) -> "DraftWorkerConfig":
        if not isinstance(raw, dict):
            raise TypeError("draft.workers entries must be mappings.")
        model_path = raw.get("model_path", defaults.get("model_path"))
        if not model_path:
            raise ValueError("Draft worker requires model_path.")
        return cls(
            worker_id=str(raw.get("worker_id") or raw.get("id") or f"draft-worker-{index}"),
            model_path=str(model_path),
            draft_type=_normalize_draft_type(raw.get("draft_type", defaults.get("draft_type", "tree"))),
            device=raw.get("device", defaults.get("device", "cuda")),
            backend=str(raw.get("backend", defaults.get("backend", "hf_eager"))),
            torch_dtype=raw.get("torch_dtype", defaults.get("torch_dtype", "auto")),
            device_map=raw.get("device_map", defaults.get("device_map")),
            trust_remote_code=bool(raw.get("trust_remote_code", defaults.get("trust_remote_code", True))),
            allow_fallback=bool(raw.get("allow_fallback", defaults.get("allow_fallback", True))),
            max_graph_len=_optional_int(raw.get("max_graph_len", defaults.get("max_graph_len"))),
            max_graph_tokens=_optional_int(raw.get("max_graph_tokens", defaults.get("max_graph_tokens"))),
            max_graph_batch_size=_optional_int(
                raw.get("max_graph_batch_size", defaults.get("max_graph_batch_size"))
            ),
            speed_profile=DraftSpeedProfile.from_config(raw.get("speed_profile")),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "model_path": self.model_path,
            "draft_type": self.draft_type,
            "device": self.device,
            "backend": self.backend,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "allow_fallback": self.allow_fallback,
            "max_graph_len": self.max_graph_len,
            "max_graph_tokens": self.max_graph_tokens,
            "max_graph_batch_size": self.max_graph_batch_size,
            "speed_profile": self.speed_profile.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class DraftWorker:
    """Loaded worker with a model and typed draft runners."""

    config: DraftWorkerConfig
    model: CausalLMRunner

    @property
    def worker_id(self) -> str:
        return self.config.worker_id

    def runner_for(self, draft_type: str) -> Any:
        normalized = _normalize_draft_type(draft_type)
        metadata = self.to_metadata()
        if normalized == "greedy":
            return GreedyDraftRunner(model=self.model, runner_id=self.worker_id, metadata=metadata)
        if normalized == "tree":
            return TopKTreeDraftRunner(model=self.model, runner_id=self.worker_id, metadata=metadata)
        raise ValueError(f"Unsupported draft runner type: {draft_type}")

    def supports(self, draft_type: str) -> bool:
        configured = _normalize_draft_type(self.config.draft_type)
        requested = _normalize_draft_type(draft_type)
        return configured in {requested, "both"}

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "backend_capabilities": self.model.backend_capabilities().to_dict(),
        }


@dataclass
class DraftWorkerRegistry:
    """Collection of loaded draft workers."""

    workers: dict[str, DraftWorker]

    @classmethod
    def from_configs(
        cls,
        configs: Iterable[DraftWorkerConfig],
        *,
        loader: DraftModelLoader = load_causal_lm_runner,
    ) -> "DraftWorkerRegistry":
        workers: dict[str, DraftWorker] = {}
        for config in configs:
            if config.worker_id in workers:
                raise ValueError(f"Duplicate draft worker_id: {config.worker_id}")
            model = loader(
                config.model_path,
                runner_id=config.worker_id,
                backend=config.backend,
                device=config.device,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
                trust_remote_code=config.trust_remote_code,
                allow_fallback=config.allow_fallback,
                max_graph_len=config.max_graph_len,
                max_graph_tokens=config.max_graph_tokens,
                max_graph_batch_size=config.max_graph_batch_size,
            )
            workers[config.worker_id] = DraftWorker(config=config, model=model)
        if not workers:
            raise ValueError("DraftWorkerRegistry requires at least one worker.")
        return cls(workers=workers)

    @classmethod
    def from_shared_model(
        cls,
        model: CausalLMRunner,
        *,
        worker_count: int,
        model_path: str,
        draft_type: str = "both",
        device: str | None = None,
        backend: str = "shared",
        torch_dtype: str | None = None,
        max_graph_len: int | None = None,
        max_graph_tokens: int | None = None,
        max_graph_batch_size: int | None = None,
    ) -> "DraftWorkerRegistry":
        """Compatibility path for memory-constrained smoke runs.

        Configured ``draft.workers`` should be preferred when independent
        workers are required; this path keeps legacy configs runnable.
        """
        workers: dict[str, DraftWorker] = {}
        for index in range(max(1, int(worker_count))):
            worker_id = f"draft-worker-{index}"
            config = DraftWorkerConfig(
                worker_id=worker_id,
                model_path=model_path,
                draft_type=_normalize_draft_type(draft_type),
                device=device,
                backend=backend,
                torch_dtype=torch_dtype,
                max_graph_len=_optional_int(max_graph_len),
                max_graph_tokens=_optional_int(max_graph_tokens),
                max_graph_batch_size=_optional_int(max_graph_batch_size),
                speed_profile=DraftSpeedProfile(name="shared_model"),
                metadata={"shared_model": True},
            )
            workers[worker_id] = DraftWorker(config=config, model=model)
        return cls(workers=workers)

    def worker_ids(self, *, draft_type: str | None = None) -> list[str]:
        if draft_type is None:
            return list(self.workers)
        return [
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.supports(draft_type)
        ]

    def runners_for(self, draft_type: str) -> dict[str, Any]:
        runners = {
            worker_id: worker.runner_for(draft_type)
            for worker_id, worker in self.workers.items()
            if worker.supports(draft_type)
        }
        if not runners:
            raise ValueError(f"No draft workers support draft_type={draft_type!r}.")
        return runners

    def first_model(self) -> CausalLMRunner:
        return next(iter(self.workers.values())).model

    def to_metadata(self) -> dict[str, Any]:
        return {
            "draft_worker_count": len(self.workers),
            "draft_workers": [
                worker.to_metadata()
                for worker in self.workers.values()
            ],
        }


def draft_worker_configs_from_settings(settings: dict[str, Any]) -> list[DraftWorkerConfig]:
    """Build worker configs from normalized smoke settings."""
    raw_workers = settings.get("draft_workers") or []
    defaults = {
        "model_path": settings.get("draft_model_path"),
        "device": settings.get("device"),
        "backend": settings.get("draft_backend"),
        "torch_dtype": settings.get("torch_dtype"),
        "device_map": settings.get("device_map"),
        "allow_fallback": settings.get("allow_backend_fallback", True),
        "max_graph_len": settings.get("draft_max_graph_len"),
        "max_graph_tokens": settings.get("draft_max_graph_tokens"),
        "max_graph_batch_size": settings.get("draft_max_graph_batch_size"),
        "draft_type": "both",
    }
    if raw_workers:
        return [
            DraftWorkerConfig.from_config(raw, index=index, defaults=defaults)
            for index, raw in enumerate(raw_workers)
        ]
    return [
        DraftWorkerConfig(
            worker_id=f"draft-worker-{index}",
            model_path=str(defaults["model_path"]),
            draft_type="both",
            device=defaults["device"],
            backend=str(defaults["backend"] or "hf_eager"),
            torch_dtype=defaults["torch_dtype"],
            device_map=defaults["device_map"],
            allow_fallback=bool(defaults["allow_fallback"]),
            max_graph_len=_optional_int(defaults.get("max_graph_len")),
            max_graph_tokens=_optional_int(defaults.get("max_graph_tokens")),
            max_graph_batch_size=_optional_int(defaults.get("max_graph_batch_size")),
            speed_profile=DraftSpeedProfile(name="legacy_worker_count"),
            metadata={"legacy_worker_count": True},
        )
        for index in range(max(1, int(settings.get("draft_worker_count", 1))))
    ]


def _normalize_draft_type(value: Any) -> str:
    text = str(value or "tree").lower().replace("-", "_")
    aliases = {
        "linear": "greedy",
        "speculative": "greedy",
        "topk_tree": "tree",
        "specedge": "tree",
        "dip_sd": "tree",
        "dipsd": "tree",
        "sled": "tree",
        "both": "both",
    }
    return aliases.get(text, text)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
