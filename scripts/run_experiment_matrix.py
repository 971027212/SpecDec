from __future__ import annotations

"""Run method/system experiment matrices over the shared smoke runner."""

import argparse
import csv
import itertools
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from specplatform.config import load_config


NETWORK_PROFILES: dict[str, dict[str, Any]] = {
    "observe": {"mode": "observe"},
    "low_uplink": {"mode": "observe", "uplink_mbps": 10.0},
    "high_rtt": {"mode": "observe", "rtt_ms": 80.0},
}

PHASE_CATEGORIES = ("scheduler", "draft", "verify", "accept", "session", "runtime", "other")
MATRIX_COMPARISON_PLOT_NAMES = (
    "matrix_runtime_by_method",
    "matrix_runtime_by_depth",
    "matrix_speedup_vs_target_only",
    "matrix_speedup_vs_tree_stop_wait",
    "matrix_verify_batch_size",
    "matrix_server_idle_gap",
    "matrix_sled_wstgr",
    "matrix_target_call_efficiency",
    "matrix_sled_queue_wait",
    "matrix_sled_timeout_fallback",
    "matrix_best_method_counts",
    "matrix_method_aggregate",
    "matrix_phase_distribution",
    "matrix_speedup_heatmap_specedge_vs_target",
    "matrix_speedup_heatmap_sled_vs_target",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SpecEdge/DiP-SD/SLED experiment matrix.")
    parser.add_argument("--base-config", default="configs/specedge_dipsd_sled_multidraft_smoke.yaml")
    parser.add_argument("--output-dir", default="experiments/method_matrix/latest")
    parser.add_argument("--request-counts", default="1,2,4,8,16")
    parser.add_argument("--draft-worker-counts", default="1,2,4")
    parser.add_argument(
        "--request-draft-pairs",
        default=None,
        help="Comma-separated paired request/draft-worker counts such as 1-1,2-2,8-1. Overrides the request/draft Cartesian product.",
    )
    parser.add_argument(
        "--draft-worker-mode",
        choices=("shared", "explicit"),
        default="explicit",
        help="explicit writes real draft.workers entries; shared keeps legacy shared-model workers for compatibility-only runs.",
    )
    parser.add_argument(
        "--worker-speed-profile",
        choices=("homogeneous", "heterogeneous", "model_size"),
        default="homogeneous",
        help="Speed metadata used when --draft-worker-mode=explicit.",
    )
    parser.add_argument(
        "--draft-worker-model-paths",
        default=None,
        help="Comma-separated model paths cycled across explicit draft workers.",
    )
    parser.add_argument(
        "--draft-worker-devices",
        default=None,
        help="Comma-separated devices cycled across explicit draft workers.",
    )
    parser.add_argument(
        "--draft-worker-backends",
        default=None,
        help="Comma-separated backends cycled across explicit draft workers.",
    )
    parser.add_argument(
        "--draft-worker-torch-dtypes",
        default=None,
        help="Comma-separated torch dtypes cycled across explicit draft workers.",
    )
    parser.add_argument(
        "--draft-worker-draft-types",
        default=None,
        help="Comma-separated draft types cycled across explicit draft workers.",
    )
    parser.add_argument(
        "--depths",
        default="1,2,4,8",
        help="Comma-separated tree depths, or 'locked'/'base' to preserve the base config depth.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--max-new-token-counts",
        default=None,
        help="Comma-separated max_new_tokens sweep such as 8,16,32,64. Overrides --max-new-tokens as a matrix dimension.",
    )
    parser.add_argument("--network-profiles", default="observe,low_uplink,high_rtt")
    parser.add_argument("--methods", default="target_only,specedge_pipeline,sled_async,dip_sd")
    parser.add_argument("--plot-formats", default="png,svg")
    parser.add_argument("--summary-plot-formats", default=None)
    parser.add_argument(
        "--local-results-dir",
        default=None,
        help="Optional server-side directory that receives a timestamped copy of matrix summaries and plots.",
    )
    parser.add_argument(
        "--result-zip-dir",
        default="transfer",
        help="Directory for a lightweight zip bundle that can be downloaded to the Windows client.",
    )
    parser.add_argument(
        "--no-result-zip",
        action="store_true",
        help="Disable writing the lightweight result zip bundle.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("single", "matrix", "both", "none"),
        default="both",
        help="single renders per-method result plots, matrix renders multi-result comparison plots.",
    )
    parser.add_argument("--no-summary-plots", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-mismatches",
        action="store_true",
        help="With --resume, rerun cells whose combined_summary has any target mismatch.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed matrix cells and continue with the remaining combinations.",
    )
    parser.add_argument(
        "--stream-cell-output",
        action="store_true",
        help="Stream each smoke subprocess instead of writing logs/<run_id>.log.",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    single_plots_enabled = args.plot_mode in {"single", "both"} and not args.no_plots
    matrix_plots_enabled = args.plot_mode in {"matrix", "both"} and not args.no_summary_plots
    base_config = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    run_root = output_dir / "runs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.request_draft_pairs:
        request_draft_pairs = _request_draft_pairs(args.request_draft_pairs)
        combinations = [
            (request_count, worker_count, depth, max_new_tokens, network_profile_name)
            for request_count, worker_count in request_draft_pairs
            for depth in _depth_values(args.depths)
            for max_new_tokens in _max_new_token_counts(args)
            for network_profile_name in _str_list(args.network_profiles)
        ]
    else:
        combinations = list(
            itertools.product(
                _int_list(args.request_counts),
                _int_list(args.draft_worker_counts),
                _depth_values(args.depths),
                _max_new_token_counts(args),
                _str_list(args.network_profiles),
            )
        )
    if args.max_cells is not None:
        combinations = combinations[: max(0, int(args.max_cells))]
    for request_count, worker_count, depth, max_new_tokens, network_profile_name in combinations:
        max_new_label = "base" if max_new_tokens is None else str(max_new_tokens)
        depth_label = "locked" if depth is None else str(depth)
        run_id = f"rc{request_count}_dw{worker_count}_d{depth_label}_mt{max_new_label}_{network_profile_name}"
        run_output = run_root / run_id
        config = _matrix_config(
            base_config,
            base_config_path=args.base_config,
            run_id=run_id,
            run_output=run_output,
            request_count=request_count,
            worker_count=worker_count,
            worker_mode=args.draft_worker_mode,
            worker_speed_profile=args.worker_speed_profile,
            worker_model_paths=_optional_str_list(args.draft_worker_model_paths),
            worker_devices=_optional_str_list(args.draft_worker_devices),
            worker_backends=_optional_str_list(args.draft_worker_backends),
            worker_torch_dtypes=_optional_str_list(args.draft_worker_torch_dtypes),
            worker_draft_types=_optional_str_list(args.draft_worker_draft_types),
            depth=depth,
            network_profile_name=network_profile_name,
            methods=_str_list(args.methods),
            plot_formats=args.plot_formats,
            disable_plots=not single_plots_enabled,
            max_new_tokens=max_new_tokens,
        )
        recorded_depth = _configured_depth(config, requested_depth=depth)
        worker_summary = _worker_config_summary_from_config(config)
        config_path = config_dir / f"{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if args.dry_run:
            rows.append(
                _plan_row(
                    run_id,
                    request_count,
                    worker_count,
                    args.draft_worker_mode,
                    args.worker_speed_profile,
                    recorded_depth,
                    network_profile_name,
                    config_path,
                    worker_summary=worker_summary,
                    max_new_tokens=max_new_tokens,
                )
            )
            continue
        if args.resume and (run_output / "combined_summary.json").exists() and not (
            args.rerun_mismatches and _summary_has_mismatch(run_output)
        ):
            print(f"[matrix] resume skip {run_id}", flush=True)
            artifact_worker_summary = _worker_config_summary_from_artifacts(run_output) or worker_summary
            rows.extend(
                _summary_rows(
                    run_id,
                    request_count,
                    worker_count,
                    args.draft_worker_mode,
                    args.worker_speed_profile,
                    recorded_depth,
                    network_profile_name,
                    run_output,
                    worker_summary=artifact_worker_summary,
                    max_new_tokens=max_new_tokens,
                )
            )
            continue
        if args.resume and (run_output / "combined_summary.json").exists() and args.rerun_mismatches:
            print(f"[matrix] rerun mismatch {run_id}", flush=True)
        command = [
            sys.executable,
            "scripts/3090_specedge_smoke.py",
            "--config",
            str(config_path),
        ]
        log_path = log_dir / f"{run_id}.log"
        print(f"[matrix] run {run_id}", flush=True)
        try:
            _run_cell_command(command, log_path=log_path, stream_output=args.stream_cell_output)
        except subprocess.CalledProcessError as exc:
            failure_row = _failure_row(
                run_id,
                request_count,
                worker_count,
                args.draft_worker_mode,
                args.worker_speed_profile,
                recorded_depth,
                max_new_tokens,
                network_profile_name,
                config_path,
                log_path=log_path,
                returncode=exc.returncode,
                worker_summary=worker_summary,
            )
            rows.append(failure_row)
            _write_matrix_outputs(rows, output_dir)
            print(f"[matrix] failed {run_id} returncode={exc.returncode} log={log_path}", flush=True)
            if not args.continue_on_error:
                raise
            continue
        rows.extend(
            _summary_rows(
                run_id,
                request_count,
                worker_count,
                args.draft_worker_mode,
                args.worker_speed_profile,
                recorded_depth,
                network_profile_name,
                run_output,
                worker_summary=_worker_config_summary_from_artifacts(run_output) or worker_summary,
                max_new_tokens=max_new_tokens,
            )
        )
        print(f"[matrix] done {run_id} log={log_path}", flush=True)

    _write_matrix_outputs(rows, output_dir)
    if matrix_plots_enabled and not args.dry_run:
        summary_formats = tuple(_str_list(args.summary_plot_formats or args.plot_formats))
        _write_matrix_plots(rows, output_dir / "plots", formats=summary_formats)
    if not args.dry_run and args.local_results_dir:
        local_export_dir = _export_result_bundle(output_dir, Path(args.local_results_dir))
        print("server_results_dir:", local_export_dir)
    if not args.dry_run and not args.no_result_zip:
        result_zip = _write_result_zip_bundle(output_dir, Path(args.result_zip_dir))
        print("result_zip:", result_zip)
    print("matrix_output_dir:", output_dir)
    print("matrix_rows:", len(rows))
    if args.dry_run:
        print("dry_run: true")


def _matrix_config(
    base_config: dict[str, Any],
    *,
    base_config_path: str | Path | None = None,
    run_id: str,
    run_output: Path,
    request_count: int,
    worker_count: int,
    worker_mode: str,
    worker_speed_profile: str,
    worker_model_paths: list[str] | None = None,
    worker_devices: list[str] | None = None,
    worker_backends: list[str] | None = None,
    worker_torch_dtypes: list[str] | None = None,
    worker_draft_types: list[str] | None = None,
    depth: int | None,
    network_profile_name: str,
    methods: list[str],
    plot_formats: str,
    disable_plots: bool,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("run", {})
    config["run"]["id"] = run_id
    config["run"]["methods"] = methods
    config["run"]["output_dir"] = str(run_output)
    config.setdefault("data", {})
    config["data"]["sample_count"] = request_count
    config["data"]["use_sample_prompts"] = True
    if max_new_tokens is not None:
        config.setdefault("generation", {})
        config["generation"]["max_new_tokens"] = int(max_new_tokens)
    config.setdefault("draft", {})
    config["draft"]["worker_count"] = worker_count
    config["draft"]["worker_mode"] = worker_mode
    if worker_model_paths is not None:
        config["draft"]["worker_model_paths"] = worker_model_paths
    if worker_devices is not None:
        config["draft"]["worker_devices"] = worker_devices
    if worker_backends is not None:
        config["draft"]["worker_backends"] = worker_backends
    if worker_torch_dtypes is not None:
        config["draft"]["worker_torch_dtypes"] = worker_torch_dtypes
    if worker_draft_types is not None:
        config["draft"]["worker_draft_types"] = worker_draft_types
    if worker_mode == "explicit":
        _apply_cell_graph_batch_size(config, request_count=request_count, worker_count=worker_count)
        config["draft"]["workers"] = _generated_worker_configs(
            config,
            worker_count=worker_count,
            worker_speed_profile=worker_speed_profile,
        )
        config["draft"]["audit"] = _draft_audit_config_for_workers(config["draft"]["workers"])
    else:
        config["draft"].pop("workers", None)
    config.setdefault("tree", {})
    config.setdefault("pipeline", {})
    if depth is not None:
        config["tree"]["max_depth"] = depth
        config["pipeline"]["max_depth"] = max(depth, int(config["pipeline"].get("max_depth", depth)))
        config["pipeline"]["proactive_depth"] = depth
    config["transport"] = dict(NETWORK_PROFILES[network_profile_name])
    config.setdefault("plots", {})
    config["plots"]["formats"] = plot_formats
    config["plots"]["disabled"] = disable_plots
    _resolve_matrix_relative_paths(config, base_config_path=base_config_path)
    return config


def _resolve_matrix_relative_paths(config: dict[str, Any], *, base_config_path: str | Path | None) -> None:
    dip_sd = config.get("dip_sd")
    if not isinstance(dip_sd, dict):
        return
    for key in ("offline_plan_table_file", "offline_plan_table_path", "calibration_profile"):
        raw = dip_sd.get(key)
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            dip_sd[key] = str(path)
            continue
        if base_config_path is not None:
            dip_sd[key] = str((Path(base_config_path).expanduser().resolve().parent / path).resolve())
        else:
            dip_sd[key] = str(path.resolve())


def _apply_cell_graph_batch_size(config: dict[str, Any], *, request_count: int, worker_count: int) -> None:
    draft = config.setdefault("draft", {})
    if draft.get("auto_graph_batch_size", True) is False:
        return
    configured = _optional_int(draft.get("max_graph_batch_size")) or 1
    active_workers = max(1, int(worker_count))
    needed = max(1, (max(1, int(request_count)) + active_workers - 1) // active_workers)
    draft["max_graph_batch_size"] = max(configured, needed)


def _generated_worker_configs(
    config: dict[str, Any],
    *,
    worker_count: int,
    worker_speed_profile: str,
) -> list[dict[str, Any]]:
    """Generate explicit draft.workers entries for matrix cells.

    This keeps the experiment dimensions honest: each worker has its own
    registry config and speed metadata.  Configs may still point at the same
    model path/device when the machine cannot afford fully separate models.
    """
    draft = dict(config.get("draft") or {})
    template_workers = [
        dict(worker)
        for worker in draft.get("workers", [])
        if isinstance(worker, dict)
    ]
    model_paths = _as_list(
        draft.get("worker_model_paths")
        or draft.get("model_paths")
        or [worker.get("model_path") for worker in template_workers]
        or _nested(config, "models", "draft")
    )
    if not model_paths or model_paths[0] is None:
        raise ValueError("Explicit draft worker mode requires models.draft or draft.worker_model_paths.")
    devices = _as_list(
        draft.get("worker_devices")
        or draft.get("devices")
        or [worker.get("device") for worker in template_workers]
        or draft.get("device", "cuda:0")
    )
    backends = _as_list(
        draft.get("worker_backends")
        or [worker.get("backend") for worker in template_workers]
        or draft.get("backend", "hf_eager")
    )
    torch_dtypes = _as_list(
        draft.get("worker_torch_dtypes")
        or [worker.get("torch_dtype") for worker in template_workers]
        or draft.get("torch_dtype", "auto")
    )
    device_maps = _as_list(
        draft.get("worker_device_maps")
        or [worker.get("device_map") for worker in template_workers]
        or draft.get("device_map")
    )
    draft_types = _as_list(
        draft.get("worker_draft_types")
        or [worker.get("draft_type") for worker in template_workers]
        or draft.get("draft_type", "both")
    )
    template_max_graph_lens = [worker.get("max_graph_len") for worker in template_workers if worker.get("max_graph_len") is not None]
    template_max_graph_tokens = [worker.get("max_graph_tokens") for worker in template_workers if worker.get("max_graph_tokens") is not None]
    template_max_graph_batch_sizes = [
        worker.get("max_graph_batch_size")
        for worker in template_workers
        if worker.get("max_graph_batch_size") is not None
    ]
    max_graph_lens = _as_list(
        draft.get("worker_max_graph_lens")
        or template_max_graph_lens
        or draft.get("max_graph_len")
    )
    max_graph_tokens = _as_list(
        draft.get("worker_max_graph_tokens")
        or template_max_graph_tokens
        or draft.get("max_graph_tokens")
    )
    max_graph_batch_sizes = _as_list(
        draft.get("worker_max_graph_batch_sizes")
        or template_max_graph_batch_sizes
        or draft.get("max_graph_batch_size")
    )
    profiles = _speed_profiles(worker_count, worker_speed_profile, model_paths=model_paths)
    workers: list[dict[str, Any]] = []
    for index in range(max(1, int(worker_count))):
        template = dict(_pick(template_workers, index) or {}) if template_workers else {}
        worker = {
            "worker_id": f"draft-worker-{index}",
            "model_path": str(_pick(model_paths, index)),
            "device": _pick(devices, index),
            "backend": str(_pick(backends, index)),
            "torch_dtype": _pick(torch_dtypes, index),
            "draft_type": str(_pick(draft_types, index)),
            "speed_profile": dict(template.get("speed_profile") or profiles[index]),
            "metadata": {
                **dict(template.get("metadata") or {}),
                "matrix_generated": True,
                "worker_speed_profile": worker_speed_profile,
                "worker_model_path_index": index % len(model_paths),
            },
        }
        if "trust_remote_code" in template:
            worker["trust_remote_code"] = bool(template["trust_remote_code"])
        if "allow_fallback" in template:
            worker["allow_fallback"] = bool(template["allow_fallback"])
        max_graph_len = _pick(max_graph_lens, index)
        if max_graph_len is not None:
            worker["max_graph_len"] = int(max_graph_len)
        max_graph_token = _pick(max_graph_tokens, index)
        if max_graph_token is not None:
            worker["max_graph_tokens"] = int(max_graph_token)
        max_graph_batch_size = _pick(max_graph_batch_sizes, index)
        if max_graph_batch_size is not None:
            worker["max_graph_batch_size"] = int(max_graph_batch_size)
        device_map = _pick(device_maps, index)
        if device_map is not None:
            worker["device_map"] = device_map
        workers.append(worker)
    return workers


def _draft_audit_config_for_workers(workers: list[dict[str, Any]]) -> dict[str, Any]:
    devices = sorted({
        str(worker.get("device"))
        for worker in workers
        if isinstance(worker, dict) and worker.get("device") is not None
    })
    return {
        "enabled": True,
        "enforce": True,
        "require_explicit_workers": True,
        "forbid_shared_model": True,
        "forbid_backend_fallback": True,
        "min_device_count": len(devices),
        "required_devices": devices,
    }


def _worker_config_summary_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize configured draft worker identities for matrix outputs."""
    draft = dict(config.get("draft") or {})
    workers = draft.get("workers") if isinstance(draft.get("workers"), list) else []
    if workers:
        model_paths = [str(worker.get("model_path") or "") for worker in workers if isinstance(worker, dict)]
        devices = [str(worker.get("device") or "") for worker in workers if isinstance(worker, dict)]
        backends = [str(worker.get("backend") or "") for worker in workers if isinstance(worker, dict)]
        draft_types = [str(worker.get("draft_type") or "") for worker in workers if isinstance(worker, dict)]
    else:
        model_path = _nested(config, "models", "draft")
        model_paths = [str(model_path)] if model_path else []
        devices = [str(draft.get("device") or "")]
        backends = [str(draft.get("backend") or "")]
        draft_types = [str(draft.get("draft_type") or "both")]
    return _worker_identity_summary(model_paths, devices=devices, backends=backends, draft_types=draft_types)


def _worker_config_summary_from_artifacts(run_output: Path) -> dict[str, Any]:
    """Read worker identities from a completed smoke run when available."""
    for smoke_path in sorted(run_output.glob("*/smoke_output.json")):
        try:
            payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        registry = dict(payload.get("draft_registry") or {})
        workers = registry.get("draft_workers")
        if not isinstance(workers, list):
            continue
        model_paths = [str(worker.get("model_path") or "") for worker in workers if isinstance(worker, dict)]
        devices = [str(worker.get("device") or "") for worker in workers if isinstance(worker, dict)]
        backends = [str(worker.get("backend") or "") for worker in workers if isinstance(worker, dict)]
        draft_types = [str(worker.get("draft_type") or "") for worker in workers if isinstance(worker, dict)]
        summary = _worker_identity_summary(model_paths, devices=devices, backends=backends, draft_types=draft_types)
        shared_count = 0
        fallback_count = 0
        fallback_backends: set[str] = set()
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            metadata = dict(worker.get("metadata") or {})
            capabilities = dict(worker.get("backend_capabilities") or {})
            if metadata.get("shared_model"):
                shared_count += 1
            if capabilities.get("backend_fallback"):
                fallback_count += 1
                if capabilities.get("backend_name"):
                    fallback_backends.add(str(capabilities["backend_name"]))
        audit = dict(payload.get("draft_registry_audit") or {})
        summary.update(
            {
                "draft_worker_shared_model_count": shared_count,
                "draft_worker_backend_fallback_count": fallback_count,
                "draft_worker_backend_fallback_set": ";".join(sorted(fallback_backends)),
                "draft_registry_audit_ok": audit.get("ok"),
                "draft_registry_audit_violations": ";".join(str(item) for item in audit.get("violations", [])),
            }
        )
        return summary
    return {}


def _worker_identity_summary(
    model_paths: list[str],
    *,
    devices: list[str],
    backends: list[str],
    draft_types: list[str],
) -> dict[str, Any]:
    normalized_paths = [path for path in model_paths if path]
    normalized_devices = [device for device in devices if device]
    unique_paths = sorted(set(normalized_paths))
    unique_devices = sorted(set(normalized_devices))
    unique_backends = sorted({backend for backend in backends if backend})
    unique_draft_types = sorted({draft_type for draft_type in draft_types if draft_type})
    if len(unique_paths) > 1:
        mode = "multi_model"
    elif len(unique_paths) == 1:
        mode = "single_model"
    else:
        mode = "unknown"
    return {
        "draft_worker_model_mode": mode,
        "draft_worker_model_id_count": len(unique_paths),
        "draft_worker_model_paths": ";".join(normalized_paths),
        "draft_worker_model_names": ";".join(Path(path).name for path in normalized_paths),
        "draft_worker_device_count": len(unique_devices),
        "draft_worker_devices": ";".join(normalized_devices),
        "draft_worker_device_set": ";".join(unique_devices),
        "draft_worker_backend_set": ";".join(unique_backends),
        "draft_worker_draft_type_set": ";".join(unique_draft_types),
    }


def _run_cell_command(command: list[str], *, log_path: Path, stream_output: bool) -> None:
    if stream_output:
        subprocess.run(command, check=True)
        return
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)


def _speed_profiles(
    worker_count: int,
    worker_speed_profile: str,
    *,
    model_paths: list[Any] | None = None,
) -> list[dict[str, Any]]:
    if worker_speed_profile == "model_size":
        paths = [str(path or "") for path in (model_paths or [])]
        sizes = [_model_size_billions(_pick(paths, index)) for index in range(max(1, int(worker_count)))]
        known_sizes = [size for size in sizes if size is not None and size > 0]
        if not known_sizes:
            return _speed_profiles(worker_count, "homogeneous")
        max_size = max(known_sizes)
        profiles: list[dict[str, Any]] = []
        for index, size in enumerate(sizes):
            size = float(size or max_size)
            relative_speed = max(0.25, max_size / max(size, 1e-6))
            quality = min(1.0, 0.72 + 0.25 * min(1.0, size / max(max_size, 1e-6)))
            profiles.append(
                {
                    "name": f"model-size-{Path(_pick(paths, index)).name or index}",
                    "relative_speed": round(relative_speed, 6),
                    "latency_ms": round(max(0.5, 4.0 / relative_speed), 6),
                    "quality": round(quality, 6),
                    "metadata": {"model_size_billions": round(size, 6)},
                }
            )
        return profiles
    if worker_speed_profile == "heterogeneous":
        speeds = [0.75, 1.0, 1.5, 2.0]
        qualities = [0.92, 1.0, 0.97, 0.9]
        latencies = [4.0, 2.0, 1.0, 0.5]
        return [
            {
                "name": f"heterogeneous-{index}",
                "relative_speed": speeds[index % len(speeds)],
                "latency_ms": latencies[index % len(latencies)],
                "quality": qualities[index % len(qualities)],
            }
            for index in range(max(1, int(worker_count)))
        ]
    return [
        {
            "name": "homogeneous",
            "relative_speed": 1.0,
            "latency_ms": 0.0,
            "quality": 1.0,
        }
        for _ in range(max(1, int(worker_count)))
    ]


def _model_size_billions(model_path: str) -> float | None:
    name = Path(str(model_path)).name
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])\b", name)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "m":
        return value / 1000.0
    return value


def _summary_rows(
    run_id: str,
    request_count: int,
    worker_count: int,
    worker_mode: str,
    worker_speed_profile: str,
    depth: int,
    network_profile_name: str,
    run_output: Path,
    *,
    worker_summary: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
) -> list[dict[str, Any]]:
    summary_path = run_output / "combined_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    worker_summary = dict(worker_summary or _worker_config_summary_from_artifacts(run_output))
    rows: list[dict[str, Any]] = []
    efficiency = dict(summary.get("method_efficiency") or {})
    matches = dict(summary.get("matches_target_only") or {})
    reproduction = dict(summary.get("method_reproduction") or {})
    for method, metrics in efficiency.items():
        reproduction_row = _flatten_reproduction_status(reproduction.get(method))
        row = {
            "run_id": run_id,
            "method": method,
            "request_count": request_count,
            "draft_worker_count": worker_count,
            "draft_worker_mode": worker_mode,
            "worker_speed_profile": worker_speed_profile,
            "depth": depth,
            "max_new_tokens": max_new_tokens,
            "network_profile": network_profile_name,
            "matches_target_only": matches.get(method),
            **reproduction_row,
        }
        row.update(worker_summary)
        for key in (
            "output_token_count",
            "proposal_count",
            "runtime_round_total_ms",
            "http_total_ms",
            "wstgr_tokens_per_s",
            "output_tokens_per_target_call",
            "target_calls_per_output_token",
            "main_target_calls_per_output_token",
            "main_target_forward_call_count",
            "raw_target_forward_call_count",
            "avg_verify_batch_size",
            "server_idle_gap_ms",
            "queue_wait_total_ms",
            "overlap_ratio",
            "proactive_alignment_rate",
            "proactive_reused_token_count",
            "proactive_discarded_token_count",
            "verify_timeout_count",
            "verify_retry_enqueue_count",
            "verify_retry_exhausted_count",
            "fallback_release_count",
            "fallback_released_token_count",
            "candidate_accept_event_count",
            "candidate_winner_count",
            "candidate_loser_count",
            "avg_candidate_count",
            "tree_forward_batch_kinds",
            "linear_forward_batch_kinds",
            "setup_load_draft_model_ms",
            "setup_warm_draft_workers_ms",
            "setup_warm_draft_worker_event_count",
            "setup_total_ms",
            "planner_wait_total_ms",
            "draft_ready_wait_total_ms",
            "dip_sd_solver_total_ms",
            "dip_sd_solver_event_count",
            "dip_sd_solver_cache_hit_count",
            "dip_sd_solver_shape_cache_hit_count",
            "dip_sd_offline_plan_table_hit_count",
            "dip_sd_requested_solver_mode",
            "dip_sd_solver_mode",
            "dip_sd_solver_backend_name",
            "dip_sd_online_solver_enabled",
            "dip_sd_paper_solver_complete",
            "dip_sd_solver_backend_fallback_used",
            "dip_sd_solver_backend_fallback_reason",
            "dip_sd_latency_calibration_profile",
            "dip_sd_latency_calibration_applied",
            "dip_sd_acceptance_feedback_enabled",
            "dip_sd_acceptance_feedback_applied_count",
            "dip_sd_acceptance_feedback_request_count",
            "steady_state_prefetch_submit_count",
            "steady_state_prefetch_reuse_event_count",
            "steady_state_prefetch_reused_draft_count",
            "steady_state_prefetch_truncated_reuse_count",
            "steady_state_prefetch_original_draft_token_count",
            "steady_state_prefetch_reused_draft_token_count",
            "steady_state_prefetch_discard_count",
        ):
            row[key] = metrics.get(key)
        row.update(_phase_summary_fields(run_output, method))
        row.update(_dip_sd_calibration_fields(run_output, method))
        row["effective_total_ms"] = _effective_total_ms(row)
        if row.get("phase_leaf_total_ms") is not None:
            row["phase_busy_over_wall_ratio"] = _safe_div(
                row.get("phase_leaf_total_ms"),
                row.get("effective_total_ms"),
            )
        rows.append(row)
    return rows


def _dip_sd_calibration_fields(run_output: Path, method: str) -> dict[str, Any]:
    if method != "dip_sd":
        return {}
    path = run_output / method / "dip_sd_latency_calibration.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    recommended = dict(data.get("recommended_method_config") or {})
    fits = list(data.get("fits") or [])
    return {
        "dip_sd_latency_observation_count": data.get("observation_count"),
        "dip_sd_calibrated_draft_c": recommended.get("dip_sd_draft_c"),
        "dip_sd_calibrated_draft_beta": recommended.get("dip_sd_draft_beta"),
        "dip_sd_calibrated_verify_c": recommended.get("dip_sd_verify_c"),
        "dip_sd_calibrated_verify_beta": recommended.get("dip_sd_verify_beta"),
        "dip_sd_latency_fit_count": len(fits),
        "dip_sd_latency_fit_status_set": ";".join(
            sorted({str(row.get("fit_status")) for row in fits if isinstance(row, dict)})
        ),
    }


def _phase_summary_fields(run_output: Path, method: str) -> dict[str, Any]:
    """Read per-method system leaf timing so matrix reports can compare phase mix."""
    path = run_output / method / "phase_summary.csv"
    if not path.exists():
        return {}
    category_totals: dict[str, float] = defaultdict(float)
    phase_totals: dict[str, float] = defaultdict(float)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("summary_view") != "system_leaf_summary":
                continue
            if row.get("event_scope") != "system" or row.get("span_kind") != "leaf":
                continue
            duration_ms = _float_or_none(row.get("total_measured_duration_ms"))
            if duration_ms is None:
                continue
            phase = str(row.get("phase") or "unknown")
            category = str(row.get("phase_category") or "").strip()
            if not category:
                category = _phase_category_from_name(phase)
            if category == "setup":
                continue
            if category not in PHASE_CATEGORIES:
                category = "other"
            category_totals[category] += duration_ms
            phase_totals[phase] += duration_ms
    if not category_totals:
        return {}
    leaf_total = float(sum(category_totals.values()))
    fields: dict[str, Any] = {"phase_leaf_total_ms": leaf_total}
    for category in PHASE_CATEGORIES:
        value = float(category_totals.get(category, 0.0))
        fields[f"phase_{category}_ms"] = value
        fields[f"phase_{category}_pct_of_leaf"] = _safe_div(value, leaf_total)
    for phase, value in sorted(phase_totals.items()):
        fields[f"phase_name_{_slug_key(phase)}_ms"] = float(value)
    return fields


def _phase_category_from_name(phase: str) -> str:
    if "." in phase:
        prefix = phase.split(".", 1)[0]
        if prefix in PHASE_CATEGORIES:
            return prefix
        if prefix in {"target", "pipeline"}:
            return "runtime"
    return "other"


def _slug_key(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "unknown"


def _summary_has_mismatch(run_output: Path) -> bool:
    summary_path = run_output / "combined_summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = dict(summary.get("matches_target_only") or {})
    return any(value is False for value in matches.values())


def _plan_row(
    run_id: str,
    request_count: int,
    worker_count: int,
    worker_mode: str,
    worker_speed_profile: str,
    depth: int,
    network_profile_name: str,
    config_path: Path,
    *,
    max_new_tokens: int | None = None,
    worker_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "request_count": request_count,
        "draft_worker_count": worker_count,
        "draft_worker_mode": worker_mode,
        "worker_speed_profile": worker_speed_profile,
        "depth": depth,
        "max_new_tokens": max_new_tokens,
        "network_profile": network_profile_name,
        "config_path": str(config_path),
        "planned": True,
    }
    row.update(worker_summary or {})
    return row


def _failure_row(
    run_id: str,
    request_count: int,
    worker_count: int,
    worker_mode: str,
    worker_speed_profile: str,
    depth: int,
    max_new_tokens: int | None,
    network_profile_name: str,
    config_path: Path,
    *,
    log_path: Path | None = None,
    returncode: int,
    worker_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "request_count": request_count,
        "draft_worker_count": worker_count,
        "draft_worker_mode": worker_mode,
        "worker_speed_profile": worker_speed_profile,
        "depth": depth,
        "max_new_tokens": max_new_tokens,
        "network_profile": network_profile_name,
        "config_path": str(config_path),
        "status": "failed",
        "returncode": int(returncode),
    }
    row.update(worker_summary or {})
    if log_path is not None:
        row["log_path"] = str(log_path)
    return row


def _write_matrix_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "matrix_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best_rows = _best_method_rows(rows)
    (output_dir / "matrix_best_methods.json").write_text(
        json.dumps(best_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_fieldnames = sorted({key for row in best_rows for key in row})
    with (output_dir / "matrix_best_methods.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=best_fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)
    comparison_rows = _comparison_rows(rows)
    (output_dir / "matrix_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_fieldnames = sorted({key for row in comparison_rows for key in row})
    with (output_dir / "matrix_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    aggregate_rows = _aggregate_method_rows(rows)
    (output_dir / "matrix_method_aggregate.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_fieldnames = sorted({key for row in aggregate_rows for key in row})
    with (output_dir / "matrix_method_aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    phase_rows = _phase_distribution_rows(rows)
    (output_dir / "matrix_phase_distribution.json").write_text(
        json.dumps(phase_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase_fieldnames = (
        sorted({key for row in phase_rows for key in row})
        or [
            "method",
            "matrix_row_count",
            "mean_effective_total_ms",
            "mean_phase_leaf_total_ms",
            "mean_phase_busy_over_wall_ratio",
        ]
    )
    with (output_dir / "matrix_phase_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=phase_fieldnames)
        writer.writeheader()
        writer.writerows(phase_rows)
    status = _matrix_status(rows)
    (output_dir / "matrix_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "matrix_report.md").write_text(
        _matrix_report_text(
            rows,
            best_rows=best_rows,
            comparison_rows=comparison_rows,
            aggregate_rows=aggregate_rows,
            phase_rows=phase_rows,
            status=status,
        ),
        encoding="utf-8",
    )


def _write_matrix_plots(rows: list[dict[str, Any]], output_dir: Path, *, formats: tuple[str, ...]) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment
        (output_dir / "matrix_plots_error.txt").write_text(
            f"matplotlib unavailable: {exc}\n",
            encoding="utf-8",
        )
        return {}
    metric_rows = []
    for row in rows:
        if not row.get("method"):
            continue
        total_ms = _effective_total_ms(row)
        if total_ms is None:
            continue
        metric_rows.append({**row, "effective_total_ms": total_ms})
    written: dict[str, list[str]] = {}
    builders = {
        "matrix_runtime_by_method": _build_runtime_by_method_plot,
        "matrix_runtime_by_depth": _build_runtime_by_depth_plot,
        "matrix_speedup_vs_target_only": _build_speedup_vs_target_only_plot,
        "matrix_speedup_vs_tree_stop_wait": _build_speedup_plot,
        "matrix_verify_batch_size": _build_verify_batch_size_plot,
        "matrix_server_idle_gap": _build_idle_gap_plot,
        "matrix_sled_wstgr": _build_wstgr_plot,
        "matrix_target_call_efficiency": _build_target_call_efficiency_plot,
        "matrix_sled_queue_wait": _build_queue_wait_plot,
        "matrix_sled_timeout_fallback": _build_timeout_fallback_plot,
        "matrix_best_method_counts": _build_best_method_counts_plot,
        "matrix_method_aggregate": _build_method_aggregate_plot,
        "matrix_phase_distribution": _build_phase_distribution_plot,
        "matrix_speedup_heatmap_specedge_vs_target": lambda plot, data: _build_speedup_heatmap_plot(
            plot,
            data,
            method="specedge_pipeline",
            baseline_method="target_only",
        ),
        "matrix_speedup_heatmap_sled_vs_target": lambda plot, data: _build_speedup_heatmap_plot(
            plot,
            data,
            method="sled_async",
            baseline_method="target_only",
        ),
    }
    for name in MATRIX_COMPARISON_PLOT_NAMES:
        builder = builders[name]
        fig = builder(plt, metric_rows)
        paths = _save_figure(fig, output_dir / name, formats)
        plt.close(fig)
        written[name] = paths
    return written


def _export_result_bundle(output_dir: Path, local_results_root: Path) -> Path:
    """Copy lightweight summaries and plots to a local timestamped archive."""
    output_dir = Path(output_dir)
    local_results_root = Path(local_results_root).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = local_results_root / output_dir.name / timestamp
    export_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_output_dir": str(output_dir.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "copied": [],
    }

    for path in sorted(output_dir.iterdir()) if output_dir.exists() else []:
        if path.is_file() and (
            path.name.startswith("matrix_")
            or path.name in {
                "combined_summary.json",
                "phase_summary.csv",
                "request_results.json",
                "smoke_output.json",
                "plot_render_ms.txt",
            }
        ):
            _copy_file(path, export_dir / path.name, manifest)
    _copy_tree(output_dir / "plots", export_dir / "plots", manifest)

    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            run_export = export_dir / "runs" / run_dir.name
            for name in ("combined_summary.json",):
                _copy_file(run_dir / name, run_export / name, manifest)
            for method_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
                method_export = run_export / method_dir.name
                for name in (
                    "dip_sd_solver_trace.json",
                    "dip_sd_offline_plan_table.json",
                    "dip_sd_stage_plan.csv",
                    "dip_sd_latency_calibration.json",
                    "dip_sd_latency_calibration.csv",
                    "dip_sd_latency_observations.csv",
                    "solver_time.csv",
                    "estimated_vs_actual_pipeline_span.csv",
                    "phase_summary.csv",
                    "pipeline_stage_timeline.csv",
                    "request_results.json",
                    "smoke_output.json",
                    "plot_render_ms.txt",
                ):
                    _copy_file(method_dir / name, method_export / name, manifest)
                _copy_tree(method_dir / "plots", method_export / "plots", manifest)

    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_dir


def _write_result_zip_bundle(output_dir: Path, zip_dir: Path) -> Path:
    """Write a lightweight zip bundle with summaries and plots for download."""
    output_dir = Path(output_dir)
    zip_dir = Path(zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{output_dir.name}_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _result_bundle_paths(output_dir):
            archive.write(path, path.relative_to(output_dir.parent))
    return zip_path


def _result_bundle_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if not output_dir.exists():
        return paths
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name.startswith("matrix_"):
            paths.append(path)
    paths.extend(sorted(path for path in (output_dir / "plots").rglob("*") if path.is_file()))
    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            combined = run_dir / "combined_summary.json"
            if combined.exists():
                paths.append(combined)
            for method_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
                for name in (
                    "dip_sd_solver_trace.json",
                    "dip_sd_offline_plan_table.json",
                    "dip_sd_stage_plan.csv",
                    "dip_sd_latency_calibration.json",
                    "dip_sd_latency_calibration.csv",
                    "dip_sd_latency_observations.csv",
                    "solver_time.csv",
                    "estimated_vs_actual_pipeline_span.csv",
                    "phase_summary.csv",
                    "pipeline_stage_timeline.csv",
                    "request_results.json",
                    "smoke_output.json",
                    "plot_render_ms.txt",
                ):
                    path = method_dir / name
                    if path.exists():
                        paths.append(path)
                paths.extend(sorted(path for path in (method_dir / "plots").rglob("*") if path.is_file()))
    return paths


def _copy_file(source: Path, dest: Path, manifest: dict[str, Any]) -> None:
    if not source.exists() or not source.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    manifest["copied"].append(str(dest))


def _copy_tree(source: Path, dest: Path, manifest: dict[str, Any]) -> None:
    if not source.exists() or not source.is_dir():
        return
    shutil.copytree(source, dest)
    manifest["copied"].append(str(dest))


def _build_runtime_by_method_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _aggregate_mean(rows, x_key="request_count", y_key="effective_total_ms", series_key="method")
    if not grouped:
        ax.text(0.5, 0.5, "No runtime rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for method, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("request_count")
    ax.set_ylabel("mean effective_total_ms")
    ax.set_title("Matrix Runtime By Method")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_runtime_by_depth_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _aggregate_mean(rows, x_key="depth", y_key="effective_total_ms", series_key="method")
    if not grouped:
        ax.text(0.5, 0.5, "No runtime rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    all_depths = sorted({depth for points in grouped.values() for depth in points})
    for method, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xticks(all_depths, [str(value) for value in all_depths])
    ax.set_xlabel("depth")
    ax.set_ylabel("mean effective_total_ms")
    ax.set_title("Matrix Runtime By Depth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_speedup_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_speedup_plot_for_baseline(
        plt,
        rows,
        baseline_method="tree_stop_wait",
        ylabel="mean speedup vs tree_stop_wait",
        title="Matrix Speedup vs tree_stop_wait",
    )


def _build_speedup_vs_target_only_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_speedup_plot_for_baseline(
        plt,
        rows,
        baseline_method="target_only",
        ylabel="mean speedup vs target_only",
        title="Matrix Speedup vs target_only",
    )


def _build_speedup_plot_for_baseline(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    ylabel: str,
    title: str,
) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    speedup_rows = _speedup_rows(rows, baseline_method=baseline_method)
    grouped = _aggregate_mean(speedup_rows, x_key="request_count", y_key="speedup", series_key="method")
    if not grouped:
        ax.text(0.5, 0.5, "No speedup rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for method, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("request_count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_idle_gap_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _aggregate_mean(rows, x_key="request_count", y_key="server_idle_gap_ms", series_key="method")
    if not grouped:
        ax.text(0.5, 0.5, "No idle-gap rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for method, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("request_count")
    ax.set_ylabel("mean server_idle_gap_ms")
    ax.set_title("Matrix Server Idle Gap")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_wstgr_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_metric_line_plot(
        plt,
        rows,
        y_key="wstgr_tokens_per_s",
        ylabel="mean WSTGR (tokens/s)",
        title="SLED Paper Metric: WSTGR",
    )


def _build_verify_batch_size_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_metric_line_plot(
        plt,
        rows,
        y_key="avg_verify_batch_size",
        ylabel="mean avg_verify_batch_size",
        title="Matrix Verify Batch Size",
    )


def _build_target_call_efficiency_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_metric_line_plot(
        plt,
        rows,
        y_key="output_tokens_per_target_call",
        ylabel="output tokens / target call",
        title="SLED Paper Metric: Target Call Efficiency",
    )


def _build_queue_wait_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    return _build_metric_line_plot(
        plt,
        rows,
        y_key="queue_wait_total_ms",
        ylabel="mean queue_wait_total_ms",
        title="SLED Paper Metric: Queue Wait",
    )


def _build_timeout_fallback_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    selected = [
        row
        for row in rows
        if str(row.get("method", "")).startswith("sled")
        and (
            _float_or_none(row.get("verify_timeout_count")) is not None
            or _float_or_none(row.get("fallback_release_count")) is not None
        )
    ]
    timeout_grouped = _aggregate_mean(selected, x_key="request_count", y_key="verify_timeout_count", series_key="method")
    fallback_grouped = _aggregate_mean(selected, x_key="request_count", y_key="fallback_release_count", series_key="method")
    if not timeout_grouped and not fallback_grouped:
        ax.text(0.5, 0.5, "No SLED timeout/fallback rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for method, points in sorted(timeout_grouped.items()):
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], marker="o", label=f"{method} timeout")
    for method, points in sorted(fallback_grouped.items()):
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], marker="s", linestyle="--", label=f"{method} fallback")
    ax.set_xlabel("request_count")
    ax.set_ylabel("mean event count")
    ax.set_title("SLED Paper Metric: Timeout / Fallback Events")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_metric_line_plot(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _aggregate_mean(rows, x_key="request_count", y_key=y_key, series_key="method")
    if not grouped:
        ax.text(0.5, 0.5, f"No rows for {y_key}", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for method, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("request_count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _build_best_method_counts_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts: dict[str, int] = defaultdict(int)
    for row in _best_method_rows(rows):
        method = row.get("best_method")
        if method:
            counts[str(method)] += 1
    if not counts:
        ax.text(0.5, 0.5, "No best-method rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    methods = sorted(counts)
    ax.bar(methods, [counts[method] for method in methods], color="#4C78A8")
    ax.set_ylabel("winning matrix cells")
    ax.set_title("Best Method Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_method_aggregate_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    aggregates = [
        row
        for row in _aggregate_method_rows(rows)
        if _float_or_none(row.get("mean_speedup_vs_tree_stop_wait")) is not None
    ]
    if not aggregates:
        ax.text(0.5, 0.5, "No aggregate rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    methods = [str(row["method"]) for row in aggregates]
    speedups = [_float_or_none(row.get("mean_speedup_vs_tree_stop_wait")) or 0.0 for row in aggregates]
    colors = ["#72B7B2" if value >= 1.0 else "#E45756" for value in speedups]
    bars = ax.bar(methods, speedups, color=colors)
    for bar, value in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_ylabel("mean speedup vs tree_stop_wait")
    ax.set_title("Method Aggregate Speedup")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def _build_phase_distribution_plot(plt: Any, rows: list[dict[str, Any]]) -> Any:
    fig, ax = plt.subplots(figsize=(10, 5))
    phase_rows = _phase_distribution_rows(rows)
    if not phase_rows:
        ax.text(0.5, 0.5, "No system leaf phase rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    phase_rows.sort(key=lambda row: float(_float_or_none(row.get("mean_effective_total_ms")) or 0.0))
    methods = [str(row.get("method")) for row in phase_rows]
    y_positions = list(range(len(methods)))
    colors = {
        "scheduler": "#7F7F7F",
        "draft": "#2A9D8F",
        "verify": "#E76F51",
        "accept": "#59A14F",
        "session": "#4E79A7",
        "runtime": "#B07AA1",
        "other": "#BAB0AC",
    }
    left = [0.0 for _method in methods]
    for category in PHASE_CATEGORIES:
        values = [
            float(_float_or_none(row.get(f"mean_phase_{category}_ms")) or 0.0)
            for row in phase_rows
        ]
        if not any(value > 0 for value in values):
            continue
        ax.barh(
            y_positions,
            values,
            left=left,
            label=category,
            color=colors.get(category, "#BAB0AC"),
            height=0.62,
        )
        left = [current + value for current, value in zip(left, values)]
    for index, row in enumerate(phase_rows):
        wall_ms = _float_or_none(row.get("mean_effective_total_ms"))
        if wall_ms is None:
            continue
        ax.plot([wall_ms, wall_ms], [index - 0.38, index + 0.38], color="#111111", linewidth=1.4)
        ratio = _float_or_none(row.get("mean_phase_busy_over_wall_ratio"))
        label = f"wall {wall_ms:.0f} ms"
        if ratio is not None:
            label += f", busy/wall {ratio:.2f}x"
        ax.text(wall_ms, index + 0.34, label, fontsize=8, va="bottom", ha="left")
    ax.set_yticks(y_positions, methods)
    ax.set_xlabel("mean system leaf measured ms")
    ax.set_title("Matrix Phase Distribution")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, ncols=4, loc="lower right")
    fig.tight_layout()
    return fig


def _build_speedup_heatmap_plot(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    method: str,
    baseline_method: str = "tree_stop_wait",
) -> Any:
    fig, ax = plt.subplots(figsize=(8, 5))
    speedup_rows = [
        row
        for row in _speedup_rows(rows, baseline_method=baseline_method)
        if row.get("method") == method
    ]
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in speedup_rows:
        speedup = _float_or_none(row.get("speedup"))
        if speedup is None:
            continue
        buckets[(int(row["request_count"]), int(row["depth"]))].append(speedup)
    if not buckets:
        ax.text(0.5, 0.5, f"No speedup rows for {method} vs {baseline_method}", ha="center", va="center")
        ax.set_axis_off()
        return fig
    request_counts = sorted({request_count for request_count, _depth in buckets})
    depths = sorted({depth for _request_count, depth in buckets})
    matrix = []
    for depth in depths:
        matrix.append(
            [
                mean(buckets[(request_count, depth)]) if buckets.get((request_count, depth)) else float("nan")
                for request_count in request_counts
            ]
        )
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(request_counts)), [str(value) for value in request_counts])
    ax.set_yticks(range(len(depths)), [str(value) for value in depths])
    ax.set_xlabel("request_count")
    ax.set_ylabel("depth")
    ax.set_title(f"{method} Speedup vs {baseline_method}")
    fig.colorbar(image, ax=ax, label="mean speedup")
    fig.tight_layout()
    return fig


def _aggregate_mean(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
) -> dict[str, dict[int, float]]:
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        y = _float_or_none(row.get(y_key))
        if y is None:
            continue
        series = str(row.get(series_key))
        x = int(row.get(x_key))
        buckets[(series, x)].append(y)
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for (series, x), values in buckets.items():
        grouped[series][x] = float(mean(values))
    return dict(grouped)


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        by_cell[_matrix_cell_key(row)][_comparison_method_key(str(method))] = row
    comparisons: list[dict[str, Any]] = []
    for cell, methods in sorted(by_cell.items()):
        base = _cell_base_row(cell)
        target_ms = _method_effective_total(methods, "target_only")
        tree_ms = _method_effective_total(methods, "tree_stop_wait")
        candidates = {
            method: _method_effective_total(methods, method)
            for method in ("tree_stop_wait", "specedge_pipeline", "dip_sd", "sled")
        }
        speculative_ranked = [
            (method, total_ms)
            for method, total_ms in candidates.items()
            if total_ms is not None and total_ms > 0
        ]
        speculative_ranked.sort(key=lambda item: item[1])
        best_speculative_method, best_speculative_ms = (
            speculative_ranked[0] if speculative_ranked else (None, None)
        )
        overall_candidates = [
            (method, total_ms)
            for method, total_ms in (("target_only", target_ms), *candidates.items())
            if total_ms is not None and total_ms > 0
        ]
        overall_candidates.sort(key=lambda item: item[1])
        best_method, best_ms = overall_candidates[0] if overall_candidates else (None, None)
        comparison = {
            **base,
            "run_id": next((row.get("run_id") for row in methods.values() if row.get("run_id")), None),
            **_cell_setup_metrics(methods),
            "target_only_effective_total_ms": target_ms,
            "tree_stop_wait_effective_total_ms": tree_ms,
            "specedge_pipeline_effective_total_ms": candidates.get("specedge_pipeline"),
            "dip_sd_effective_total_ms": candidates.get("dip_sd"),
            "sled_effective_total_ms": candidates.get("sled"),
            "best_method": best_method,
            "best_effective_total_ms": best_ms,
            "best_speculative_method": best_speculative_method,
            "best_speculative_effective_total_ms": best_speculative_ms,
        }
        for method in ("tree_stop_wait", "specedge_pipeline", "dip_sd", "sled"):
            row = methods.get(method, {})
            method_ms = candidates.get(method)
            comparison[f"{method}_matches_target_only"] = row.get("matches_target_only")
            comparison[f"{method}_speedup_vs_tree_stop_wait"] = _safe_div(tree_ms, method_ms)
            comparison[f"{method}_speedup_vs_target_only"] = _safe_div(target_ms, method_ms)
            comparison[f"{method}_server_idle_gap_ms"] = row.get("server_idle_gap_ms")
            comparison[f"{method}_avg_verify_batch_size"] = row.get("avg_verify_batch_size")
            comparison[f"{method}_main_target_forward_call_count"] = row.get("main_target_forward_call_count")
            comparison[f"{method}_raw_target_forward_call_count"] = row.get("raw_target_forward_call_count")
            comparison[f"{method}_overlap_ratio"] = row.get("overlap_ratio")
            comparison[f"{method}_proactive_reused_token_count"] = row.get("proactive_reused_token_count")
            comparison[f"{method}_candidate_winner_count"] = row.get("candidate_winner_count")
            comparison[f"{method}_avg_candidate_count"] = row.get("avg_candidate_count")
            comparison[f"{method}_setup_total_ms"] = row.get("setup_total_ms")
            comparison[f"{method}_reproduction_execution_mode"] = row.get("reproduction_execution_mode")
            comparison[f"{method}_reproduction_partial_or_missing_count"] = row.get(
                "reproduction_partial_or_missing_count"
            )
            comparison[f"{method}_reproduction_not_original_count"] = row.get(
                "reproduction_not_original_count"
            )
        comparison["best_speedup_vs_tree_stop_wait"] = _safe_div(tree_ms, best_ms)
        comparison["best_speedup_vs_target_only"] = _safe_div(target_ms, best_ms)
        comparison["best_speculative_speedup_vs_target_only"] = _safe_div(target_ms, best_speculative_ms)
        comparisons.append(comparison)
    return comparisons


def _comparison_method_key(method: str) -> str:
    if method == "sled_async":
        return "sled"
    return str(method)


def _aggregate_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    speedups = _speedup_rows(rows, baseline_method="tree_stop_wait")
    speedups_by_method: dict[str, list[float]] = defaultdict(list)
    for row in speedups:
        speedup = _float_or_none(row.get("speedup"))
        if speedup is not None:
            speedups_by_method[str(row.get("method"))].append(speedup)
    win_counts: dict[str, int] = defaultdict(int)
    for row in _best_method_rows(rows):
        method = row.get("best_method")
        if method:
            win_counts[str(method)] += 1
    for row in rows:
        method = row.get("method")
        if method:
            by_method[str(method)].append(row)
    aggregates: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        total_values = [_effective_total_ms(row) for row in method_rows]
        total_values = [value for value in total_values if value is not None]
        match_values = [row.get("matches_target_only") for row in method_rows if row.get("method") != "target_only"]
        aggregates.append(
            {
                "method": method,
                "matrix_row_count": len(method_rows),
                "mean_effective_total_ms": mean(total_values) if total_values else None,
                "mean_speedup_vs_tree_stop_wait": (
                    mean(speedups_by_method[method]) if speedups_by_method.get(method) else None
                ),
                "winning_cell_count": win_counts.get(method, 0),
                "match_count": sum(1 for value in match_values if value is True),
                "mismatch_count": sum(1 for value in match_values if value is False),
                "mean_server_idle_gap_ms": _mean_metric(method_rows, "server_idle_gap_ms"),
                "mean_avg_verify_batch_size": _mean_metric(method_rows, "avg_verify_batch_size"),
                "mean_overlap_ratio": _mean_metric(method_rows, "overlap_ratio"),
                "reproduction_execution_modes": _mode_counts_text(method_rows, "reproduction_execution_mode"),
                "reproduction_complete_count": sum(
                    1
                    for row in method_rows
                    if int(row.get("reproduction_partial_or_missing_count") or 0) == 0
                ),
                "mean_reproduction_partial_or_missing_count": _mean_metric(
                    method_rows,
                    "reproduction_partial_or_missing_count",
                ),
                "mean_reproduction_not_original_count": _mean_metric(
                    method_rows,
                    "reproduction_not_original_count",
                ),
                "mean_setup_load_draft_model_ms": _mean_metric(method_rows, "setup_load_draft_model_ms"),
                "mean_setup_warm_draft_workers_ms": _mean_metric(method_rows, "setup_warm_draft_workers_ms"),
                "mean_setup_total_ms": _mean_metric(method_rows, "setup_total_ms"),
                "mean_main_target_forward_call_count": _mean_metric(
                    method_rows,
                    "main_target_forward_call_count",
                ),
                "mean_raw_target_forward_call_count": _mean_metric(
                    method_rows,
                    "raw_target_forward_call_count",
                ),
            }
        )
    return aggregates


def _phase_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = row.get("method")
        if method and _float_or_none(row.get("phase_leaf_total_ms")) is not None:
            by_method[str(method)].append(row)
    phase_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        phase_row: dict[str, Any] = {
            "method": method,
            "matrix_row_count": len(method_rows),
            "mean_effective_total_ms": _mean_metric(method_rows, "effective_total_ms"),
            "mean_phase_leaf_total_ms": _mean_metric(method_rows, "phase_leaf_total_ms"),
            "mean_phase_busy_over_wall_ratio": _mean_metric(method_rows, "phase_busy_over_wall_ratio"),
        }
        for category in PHASE_CATEGORIES:
            phase_row[f"mean_phase_{category}_ms"] = _mean_metric(method_rows, f"phase_{category}_ms")
            phase_row[f"mean_phase_{category}_pct_of_leaf"] = _mean_metric(
                method_rows,
                f"phase_{category}_pct_of_leaf",
            )
            phase_row[f"mean_phase_{category}_pct_of_wall"] = _mean_phase_pct_of_wall(
                method_rows,
                f"phase_{category}_ms",
            )
        phase_rows.append(phase_row)
    return phase_rows


def _mean_phase_pct_of_wall(rows: list[dict[str, Any]], phase_key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        ratio = _safe_div(row.get(phase_key), row.get("effective_total_ms"))
        if ratio is not None:
            values.append(ratio)
    return mean(values) if values else None


def _flatten_reproduction_status(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {
            "reproduction_reference_scope": None,
            "reproduction_execution_mode": None,
            "reproduction_implemented": "",
            "reproduction_implemented_count": 0,
            "reproduction_partial_or_missing": "",
            "reproduction_partial_or_missing_count": 0,
            "reproduction_not_counted_as_original": "",
            "reproduction_not_original_count": 0,
            "reproduction_scheduler_method_family": None,
        }
    implemented = _string_list(status.get("implemented"))
    partial_or_missing = _string_list(status.get("partial_or_missing"))
    not_original = _string_list(status.get("not_counted_as_original"))
    signals = dict(status.get("signals") or {})
    return {
        "reproduction_reference_scope": status.get("reference_scope"),
        "reproduction_execution_mode": status.get("execution_mode"),
        "reproduction_implemented": _join_text_items(implemented),
        "reproduction_implemented_count": len(implemented),
        "reproduction_partial_or_missing": _join_text_items(partial_or_missing),
        "reproduction_partial_or_missing_count": len(partial_or_missing),
        "reproduction_not_counted_as_original": _join_text_items(not_original),
        "reproduction_not_original_count": len(not_original),
        "reproduction_scheduler_method_family": signals.get("scheduler_method_family"),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _join_text_items(items: list[str]) -> str:
    return "; ".join(item.replace("\n", " ").strip() for item in items if item.strip())


def _mode_counts_text(rows: list[dict[str, Any]], key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key)
        if value:
            counts[str(value)] += 1
    return "; ".join(f"{mode}:{counts[mode]}" for mode in sorted(counts))


def _cell_setup_metrics(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick setup metrics once per matrix cell without affecting speedup math."""
    source = next(
        (
            row
            for row in methods.values()
            if row.get("setup_total_ms") is not None
            or row.get("setup_load_draft_model_ms") is not None
            or row.get("setup_warm_draft_workers_ms") is not None
        ),
        {},
    )
    return {
        "setup_load_draft_model_ms": source.get("setup_load_draft_model_ms"),
        "setup_warm_draft_workers_ms": source.get("setup_warm_draft_workers_ms"),
        "setup_warm_draft_worker_event_count": source.get("setup_warm_draft_worker_event_count"),
        "setup_total_ms": source.get("setup_total_ms"),
    }


def _matrix_report_text(
    rows: list[dict[str, Any]],
    *,
    best_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    status: dict[str, Any],
) -> str:
    lines = [
        "# Matrix Report",
        "",
        "## Status",
        "",
        f"- completed_cell_count: {status.get('completed_cell_count')}",
        f"- failed_cell_count: {status.get('failed_cell_count')}",
        f"- incomplete_cell_count: {status.get('incomplete_cell_count')}",
        f"- method_row_count: {status.get('method_row_count')}",
        "",
        "## Method Aggregate",
        "",
        "| method | wins | match | mismatch | mean_total_ms | setup_ms | speedup_vs_tree | avg_batch | idle_ms | overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(aggregate_rows, key=lambda item: int(item.get("winning_cell_count") or 0), reverse=True):
        lines.append(
            "| {method} | {wins} | {match} | {mismatch} | {total} | {setup} | {speedup} | {batch} | {idle} | {overlap} |".format(
                method=row.get("method"),
                wins=row.get("winning_cell_count"),
                match=row.get("match_count"),
                mismatch=row.get("mismatch_count"),
                total=_fmt(row.get("mean_effective_total_ms")),
                setup=_fmt(row.get("mean_setup_total_ms")),
                speedup=_fmt(row.get("mean_speedup_vs_tree_stop_wait")),
                batch=_fmt(row.get("mean_avg_verify_batch_size")),
                idle=_fmt(row.get("mean_server_idle_gap_ms")),
                overlap=_fmt(row.get("mean_overlap_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Method Reproduction",
            "",
            "| method | modes | complete_rows | mean_missing | mean_future_opt |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(aggregate_rows, key=lambda item: str(item.get("method"))):
        lines.append(
            "| {method} | {modes} | {complete} | {missing} | {future} |".format(
                method=row.get("method"),
                modes=row.get("reproduction_execution_modes") or "",
                complete=row.get("reproduction_complete_count"),
                missing=_fmt(row.get("mean_reproduction_partial_or_missing_count")),
                future=_fmt(row.get("mean_reproduction_not_original_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Phase Distribution",
            "",
            "Uses `system_leaf_summary` only. `busy/wall` can exceed 1.0 when workers run in parallel or draft and verify overlap.",
            "",
        ]
    )
    if not phase_rows:
        lines.extend(["No phase distribution rows.", ""])
    else:
        lines.append(
            "| method | wall_ms | leaf_busy_ms | busy/wall | scheduler% | draft% | verify% | accept% | session% | runtime% | other% |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(phase_rows, key=lambda item: str(item.get("method"))):
            lines.append(
                "| {method} | {wall} | {leaf} | {ratio} | {scheduler} | {draft} | {verify} | {accept} | {session} | {runtime} | {other} |".format(
                    method=row.get("method"),
                    wall=_fmt(row.get("mean_effective_total_ms")),
                    leaf=_fmt(row.get("mean_phase_leaf_total_ms")),
                    ratio=_fmt(row.get("mean_phase_busy_over_wall_ratio")),
                    scheduler=_fmt_percent(row.get("mean_phase_scheduler_pct_of_leaf")),
                    draft=_fmt_percent(row.get("mean_phase_draft_pct_of_leaf")),
                    verify=_fmt_percent(row.get("mean_phase_verify_pct_of_leaf")),
                    accept=_fmt_percent(row.get("mean_phase_accept_pct_of_leaf")),
                    session=_fmt_percent(row.get("mean_phase_session_pct_of_leaf")),
                    runtime=_fmt_percent(row.get("mean_phase_runtime_pct_of_leaf")),
                    other=_fmt_percent(row.get("mean_phase_other_pct_of_leaf")),
                )
            )
    depth_effect_rows = _depth_effect_rows(comparison_rows)
    if depth_effect_rows:
        lines.extend(
            [
                "",
                "## Depth Effect",
                "",
                "| run_id | requests | workers | depth | target_ms | tree_ms | specedge_ms | dip_sd_ms | sled_ms | specedge_speedup | specedge_idle_ms | tree_idle_ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in depth_effect_rows[:30]:
            lines.append(
                "| {run_id} | {requests} | {workers} | {depth} | {target} | {tree} | {specedge} | {dip} | {sled} | {speedup} | {specedge_idle} | {tree_idle} |".format(
                    run_id=row.get("run_id"),
                    requests=row.get("request_count"),
                    workers=row.get("draft_worker_count"),
                    depth=row.get("depth"),
                    target=_fmt(row.get("target_only_effective_total_ms")),
                    tree=_fmt(row.get("tree_stop_wait_effective_total_ms")),
                    specedge=_fmt(row.get("specedge_pipeline_effective_total_ms")),
                    dip=_fmt(row.get("dip_sd_effective_total_ms")),
                    sled=_fmt(row.get("sled_effective_total_ms")),
                    speedup=_fmt(row.get("specedge_pipeline_speedup_vs_tree_stop_wait")),
                    specedge_idle=_fmt(row.get("specedge_pipeline_server_idle_gap_ms")),
                    tree_idle=_fmt(row.get("tree_stop_wait_server_idle_gap_ms")),
                )
            )
        if len(depth_effect_rows) > 30:
            lines.append(f"| ... |  |  | {len(depth_effect_rows) - 30} more rows |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Speculative Winner Counts", ""])
    for title, key in (
        ("overall", None),
        ("by request_count", "request_count"),
        ("by draft_worker_count", "draft_worker_count"),
        ("by draft_worker_device_count", "draft_worker_device_count"),
        ("by worker_speed_profile", "worker_speed_profile"),
        ("by draft_worker_model_mode", "draft_worker_model_mode"),
        ("by depth", "depth"),
        ("by network_profile", "network_profile"),
    ):
        lines.extend(_winner_count_section(best_rows, title=title, key=key))

    lines.extend(["", "## Top Speedup Cells", ""])
    speedup_section_found = False
    for baseline_method in ("target_only", "tree_stop_wait"):
        baseline_lines = [f"### vs {baseline_method}", ""]
        baseline_found = False
        for method in ("specedge_pipeline", "dip_sd", "sled"):
            top_rows = _top_speedup_cells(
                comparison_rows,
                method=method,
                baseline_method=baseline_method,
                limit=5,
            )
            if not top_rows:
                continue
            baseline_found = True
            speedup_section_found = True
            baseline_lines.extend([f"#### {method}", ""])
            baseline_lines.append(
                "| run_id | requests | workers | devices | model_mode | profile | depth | network | speedup | total_ms |"
            )
            baseline_lines.append("| --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: |")
            for row in top_rows:
                baseline_lines.append(
                    "| {run_id} | {request_count} | {draft_worker_count} | {devices} | {model_mode} | {profile} | {depth} | {network_profile} | {speedup} | {total} |".format(
                        run_id=row.get("run_id"),
                        request_count=row.get("request_count"),
                        draft_worker_count=row.get("draft_worker_count"),
                        devices=row.get("draft_worker_device_set"),
                        model_mode=row.get("draft_worker_model_mode"),
                        profile=row.get("worker_speed_profile"),
                        depth=row.get("depth"),
                        network_profile=row.get("network_profile"),
                        speedup=_fmt(row.get(f"{method}_speedup_vs_{baseline_method}")),
                        total=_fmt(row.get(f"{method}_effective_total_ms")),
                    )
                )
            baseline_lines.append("")
        if baseline_found:
            lines.extend(baseline_lines)
    if not speedup_section_found:
        lines.extend(["No speedup rows.", ""])

    bad_rows = [
        row
        for row in rows
        if row.get("method")
        and row.get("method") != "target_only"
        and row.get("matches_target_only") is False
    ]
    lines.extend(
        [
            "## Correctness",
            "",
            f"- target mismatches: {len(bad_rows)}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _depth_effect_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    depths = {row.get("depth") for row in comparison_rows if row.get("depth") is not None}
    if len(depths) <= 1:
        return []
    return sorted(
        comparison_rows,
        key=lambda row: (
            int(row.get("request_count") or 0),
            int(row.get("draft_worker_count") or 0),
            int(row.get("depth") or 0),
            str(row.get("network_profile") or ""),
        ),
    )


def _winner_count_section(
    best_rows: list[dict[str, Any]],
    *,
    title: str,
    key: str | None,
) -> list[str]:
    lines = [f"### {title}", ""]
    counts: dict[Any, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if key is None:
        for row in best_rows:
            counts["all"][str(row.get("best_method"))] += 1
    else:
        for row in best_rows:
            counts[row.get(key)][str(row.get("best_method"))] += 1
    methods = sorted({method for row in counts.values() for method in row})
    if not methods:
        return [*lines, "No completed cells.", ""]
    lines.append("| bucket | " + " | ".join(methods) + " |")
    lines.append("| --- | " + " | ".join("---:" for _method in methods) + " |")
    for bucket, method_counts in sorted(counts.items(), key=lambda item: str(item[0])):
        values = " | ".join(str(method_counts.get(method, 0)) for method in methods)
        lines.append(f"| {bucket} | {values} |")
    lines.append("")
    return lines


def _top_speedup_cells(
    comparison_rows: list[dict[str, Any]],
    *,
    method: str,
    baseline_method: str,
    limit: int,
) -> list[dict[str, Any]]:
    speedup_key = f"{method}_speedup_vs_{baseline_method}"
    rows = [
        row
        for row in comparison_rows
        if _float_or_none(row.get(speedup_key)) is not None
        and _float_or_none(row.get(f"{method}_effective_total_ms")) is not None
    ]
    rows.sort(key=lambda row: float(_float_or_none(row.get(speedup_key)) or 0.0), reverse=True)
    return rows[:limit]


def _matrix_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_rows = [row for row in rows if row.get("method")]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    planned_rows = [row for row in rows if row.get("planned")]
    methods = sorted({str(row.get("method")) for row in method_rows})
    by_cell: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in method_rows:
        by_cell[_matrix_cell_key(row)].add(str(row.get("method")))
    incomplete_cells = []
    expected_methods = set(methods)
    for cell, present_methods in sorted(by_cell.items()):
        missing = sorted(expected_methods - present_methods)
        if missing:
            incomplete_cells.append({**_cell_base_row(cell), "missing_methods": missing})
    return {
        "matrix_row_count": len(rows),
        "method_row_count": len(method_rows),
        "planned_cell_count": len(planned_rows),
        "completed_cell_count": len(by_cell),
        "failed_cell_count": len(failed_rows),
        "incomplete_cell_count": len(incomplete_cells),
        "methods": methods,
        "request_counts": sorted({row.get("request_count") for row in rows if row.get("request_count") is not None}),
        "draft_worker_counts": sorted(
            {row.get("draft_worker_count") for row in rows if row.get("draft_worker_count") is not None}
        ),
        "depths": sorted({row.get("depth") for row in rows if row.get("depth") is not None}),
        "max_new_tokens": sorted(
            {row.get("max_new_tokens") for row in rows if row.get("max_new_tokens") is not None}
        ),
        "network_profiles": sorted(
            {str(row.get("network_profile")) for row in rows if row.get("network_profile") is not None}
        ),
        "incomplete_cells": incomplete_cells[:50],
        "failed_cells": [
            {
                "run_id": row.get("run_id"),
                "request_count": row.get("request_count"),
                "draft_worker_count": row.get("draft_worker_count"),
                "draft_worker_model_mode": row.get("draft_worker_model_mode"),
                "draft_worker_device_set": row.get("draft_worker_device_set"),
                "worker_speed_profile": row.get("worker_speed_profile"),
                "depth": row.get("depth"),
                "max_new_tokens": row.get("max_new_tokens"),
                "network_profile": row.get("network_profile"),
                "returncode": row.get("returncode"),
            }
            for row in failed_rows[:50]
        ],
    }


def _speedup_rows(rows: list[dict[str, Any]], *, baseline_method: str) -> list[dict[str, Any]]:
    by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = _matrix_cell_key(row)
        by_cell[key][str(row.get("method"))] = row
    speedups: list[dict[str, Any]] = []
    for cell, methods in by_cell.items():
        baseline = methods.get(baseline_method)
        baseline_ms = None if baseline is None else _effective_total_ms(baseline)
        if not baseline_ms or baseline_ms <= 0:
            continue
        for method, row in methods.items():
            if method in {"target_only", baseline_method}:
                continue
            runtime_ms = _effective_total_ms(row)
            if not runtime_ms or runtime_ms <= 0:
                continue
            speedups.append({**row, "speedup": baseline_ms / runtime_ms, "matrix_cell": cell})
    return speedups


def _best_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "target_only":
            continue
        if _effective_total_ms(row) is None:
            continue
        by_cell[_matrix_cell_key(row)].append(row)
    best_rows: list[dict[str, Any]] = []
    for cell, candidates in sorted(by_cell.items()):
        best = min(candidates, key=lambda row: float(_effective_total_ms(row) or 0.0))
        best_rows.append(
            {
                "run_id": best.get("run_id"),
                "request_count": best.get("request_count"),
                "draft_worker_count": best.get("draft_worker_count"),
                "draft_worker_mode": best.get("draft_worker_mode"),
                "worker_speed_profile": best.get("worker_speed_profile"),
                "draft_worker_model_mode": best.get("draft_worker_model_mode"),
                "draft_worker_model_id_count": best.get("draft_worker_model_id_count"),
                "draft_worker_device_count": best.get("draft_worker_device_count"),
                "draft_worker_device_set": best.get("draft_worker_device_set"),
                "draft_worker_backend_set": best.get("draft_worker_backend_set"),
                "draft_worker_draft_type_set": best.get("draft_worker_draft_type_set"),
                "depth": best.get("depth"),
                "max_new_tokens": best.get("max_new_tokens"),
                "network_profile": best.get("network_profile"),
                "best_method": best.get("method"),
                "best_runtime_round_total_ms": best.get("runtime_round_total_ms"),
                "best_effective_total_ms": _effective_total_ms(best),
                "cell": "|".join(str(part) for part in cell),
            }
        )
    return best_rows


def _matrix_cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("request_count"),
        row.get("draft_worker_count"),
        row.get("draft_worker_mode", "shared"),
        row.get("worker_speed_profile", "homogeneous"),
        row.get("draft_worker_model_mode", "unknown"),
        row.get("draft_worker_model_id_count", 0),
        row.get("draft_worker_device_count", 0),
        row.get("draft_worker_device_set", ""),
        row.get("draft_worker_backend_set", ""),
        row.get("draft_worker_draft_type_set", ""),
        row.get("depth"),
        row.get("max_new_tokens"),
        row.get("network_profile"),
    )


def _cell_base_row(cell: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "request_count": cell[0],
        "draft_worker_count": cell[1],
        "draft_worker_mode": cell[2],
        "worker_speed_profile": cell[3],
        "draft_worker_model_mode": cell[4],
        "draft_worker_model_id_count": cell[5],
        "draft_worker_device_count": cell[6],
        "draft_worker_device_set": cell[7],
        "draft_worker_backend_set": cell[8],
        "draft_worker_draft_type_set": cell[9],
        "depth": cell[10],
        "max_new_tokens": cell[11],
        "network_profile": cell[12],
    }


def _method_effective_total(methods: dict[str, dict[str, Any]], method: str) -> float | None:
    row = methods.get(method)
    if row is None:
        return None
    return _effective_total_ms(row)


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    left = _float_or_none(numerator)
    right = _float_or_none(denominator)
    if left is None or right is None or right <= 0:
        return None
    return left / right


def _fmt(value: Any, digits: int = 3) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_percent(value: Any, digits: int = 1) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number * 100:.{digits}f}%"


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def _save_figure(fig: Any, stem: Path, formats: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        path = stem.with_suffix("." + fmt)
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    return paths


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _effective_total_ms(row: dict[str, Any]) -> float | None:
    runtime_ms = _float_or_none(row.get("runtime_round_total_ms"))
    if runtime_ms is not None and runtime_ms > 0:
        return runtime_ms
    return _float_or_none(row.get("http_total_ms"))


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        return list(value) or [None]
    return [value]


def _pick(values: list[Any], index: int) -> Any:
    return values[index % len(values)]


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _depth_values(value: str) -> list[int | None]:
    depths: list[int | None] = []
    for part in str(value).split(","):
        raw = part.strip()
        if not raw:
            continue
        if raw.lower() in {"locked", "base", "config"}:
            depths.append(None)
        else:
            depths.append(int(raw))
    return depths or [None]


def _max_new_token_counts(args: argparse.Namespace) -> list[int | None]:
    raw_counts = getattr(args, "max_new_token_counts", None)
    if raw_counts:
        return _int_list(str(raw_counts))
    single_value = getattr(args, "max_new_tokens", None)
    return [None if single_value is None else int(single_value)]


def _configured_depth(config: dict[str, Any], *, requested_depth: int | None) -> int | None:
    if requested_depth is not None:
        return int(requested_depth)
    tree_depth = _nested(config, "tree", "max_depth")
    if tree_depth is not None:
        return int(tree_depth)
    pipeline_depth = _nested(config, "pipeline", "proactive_depth")
    if pipeline_depth is not None:
        return int(pipeline_depth)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _request_draft_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        match = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", raw)
        if not match:
            raise ValueError(f"Invalid request/draft pair: {raw!r}. Expected forms like 4-4 or 8:1.")
        pairs.append((int(match.group(1)), int(match.group(2))))
    if not pairs:
        raise ValueError("--request-draft-pairs did not contain any pairs.")
    return pairs


def _str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _optional_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return _str_list(value)


if __name__ == "__main__":
    main()
