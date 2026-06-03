from __future__ import annotations

"""Tune SpecEdge, SLED, and DiP-SD before a fair comparison run.

This runner expands method-specific tuning grids into ordinary
``3090_specedge_smoke.py`` configs.  It keeps the fairness envelope fixed
(prompts, target verifier, draft workers, max_new_tokens, network profile)
while allowing each method to tune only its own core hyperparameters.
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from specplatform.config import load_config


NETWORK_PROFILES: dict[str, dict[str, Any]] = {
    "observe": {"mode": "observe"},
    "low_uplink": {"mode": "observe", "uplink_mbps": 10.0},
    "high_rtt": {"mode": "observe", "rtt_ms": 80.0},
}

TUNED_METHODS = ("specedge_pipeline", "sled_async", "dip_sd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run per-method tuning before SpecEdge/SLED/DiP-SD comparison."
    )
    parser.add_argument("--base-config", default="configs/four_method_compare_1xa100.yaml")
    parser.add_argument("--output-dir", default="experiments/three_method_tuning/latest")
    parser.add_argument("--methods", default="specedge_pipeline,sled_async,dip_sd")
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--tuning-prompts-file", default=None)
    parser.add_argument("--heldout-prompts-file", default=None)
    parser.add_argument("--heldout-request-count", type=int, default=None)
    parser.add_argument("--draft-worker-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--network-profile", default="observe", choices=tuple(NETWORK_PROFILES))
    parser.add_argument("--plot-formats", default="png,svg")
    parser.add_argument("--plot-mode", choices=("none", "single"), default="none")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--stream-candidate-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidates-per-method", type=int, default=None)

    parser.add_argument("--specedge-depths", default="4,8")
    parser.add_argument("--specedge-branch-widths", default="4,8")
    parser.add_argument("--specedge-max-budgets", default="20")
    parser.add_argument("--specedge-proactive-depths", default="4,8")
    parser.add_argument("--specedge-official", default="true")

    parser.add_argument("--sled-max-speculation-tokens", default="4,8")
    parser.add_argument("--sled-confidence-thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--sled-batch-sizes", default="4,8")
    parser.add_argument("--sled-proactive-tokens", default="4,8")
    parser.add_argument("--sled-queue-max-wait-ms", default="none")

    parser.add_argument("--dip-sd-max-draft-lengths", default="4,8")
    parser.add_argument("--dip-sd-solvers", default="paper_milp_or_dinkelbach")
    parser.add_argument("--dip-sd-batch-counts", default="auto")
    parser.add_argument(
        "--dip-sd-calibration-profiles",
        default="none",
        help="Comma-separated profiles. Use 'none' for calibration off.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    config_root = output_dir / "tuning_configs"
    run_root = output_dir / "tuning_runs"
    log_root = output_dir / "logs"
    config_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    prompt_split = _prompt_split(args, base_config)
    candidates = _candidate_specs(
        args,
        base_config,
        config_root=config_root,
        run_root=run_root,
        tuning_prompts_file=prompt_split["tuning_prompts_file"],
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        config_path = Path(candidate["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(candidate["config"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        row = _planned_row(candidate)
        if args.dry_run:
            rows.append(row)
            continue
        run_output = Path(candidate["run_output"])
        summary_path = run_output / "combined_summary.json"
        if args.resume and summary_path.exists():
            rows.append(_summary_row(candidate, summary_path=summary_path, status="completed"))
            continue
        command = [sys.executable, "scripts/3090_specedge_smoke.py", "--config", str(config_path)]
        log_path = log_root / f"{candidate['candidate_id']}.log"
        try:
            _run_candidate(command, log_path=log_path, stream_output=args.stream_candidate_output)
        except subprocess.CalledProcessError as exc:
            failed = dict(row)
            failed.update(
                {
                    "status": "failed",
                    "returncode": exc.returncode,
                    "log_path": str(log_path),
                }
            )
            rows.append(failed)
            _write_outputs(rows, output_dir, base_config=base_config, prompt_split=prompt_split)
            if not args.continue_on_error:
                raise
            continue
        rows.append(_summary_row(candidate, summary_path=summary_path, status="completed"))

    _write_outputs(rows, output_dir, base_config=base_config, prompt_split=prompt_split)
    print("tuning_output_dir:", output_dir)
    print("candidate_count:", len(rows))
    if args.dry_run:
        print("dry_run: true")


def _candidate_specs(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    *,
    config_root: Path,
    run_root: Path,
    tuning_prompts_file: Path | None = None,
) -> list[dict[str, Any]]:
    selected_methods = [method for method in _str_list(args.methods) if method in TUNED_METHODS]
    candidates: list[dict[str, Any]] = []
    for method in selected_methods:
        method_candidates = _method_candidate_specs(
            args,
            base_config,
            method=method,
            config_root=config_root / method,
            run_root=run_root / method,
            tuning_prompts_file=tuning_prompts_file,
        )
        if args.max_candidates_per_method is not None:
            method_candidates = method_candidates[: max(0, int(args.max_candidates_per_method))]
        candidates.extend(method_candidates)
    return candidates


def _method_candidate_specs(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    *,
    method: str,
    config_root: Path,
    run_root: Path,
    tuning_prompts_file: Path | None = None,
) -> list[dict[str, Any]]:
    if method == "specedge_pipeline":
        grid = _specedge_grid(args)
    elif method == "sled_async":
        grid = _sled_grid(args)
    elif method == "dip_sd":
        grid = _dip_sd_grid(args)
    else:
        raise ValueError(f"Unsupported tuning method: {method}")

    candidates: list[dict[str, Any]] = []
    for index, params in enumerate(grid):
        candidate_id = f"{method}_{index:03d}_{_param_slug(params)}"
        run_output = run_root / candidate_id
        config = _candidate_config(
            base_config,
            method=method,
            candidate_id=candidate_id,
            run_output=run_output,
            request_count=int(args.request_count),
            draft_worker_count=int(args.draft_worker_count),
            max_new_tokens=int(args.max_new_tokens),
            network_profile=str(args.network_profile),
            plot_formats=str(args.plot_formats),
            plots_disabled=str(args.plot_mode) == "none",
            tuning_prompts_file=tuning_prompts_file,
            params=params,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "method": method,
                "params": params,
                "config": config,
                "config_path": str(config_root / f"{candidate_id}.yaml"),
                "run_output": str(run_output),
            }
        )
    return candidates


def _candidate_config(
    base_config: dict[str, Any],
    *,
    method: str,
    candidate_id: str,
    run_output: Path,
    request_count: int,
    draft_worker_count: int,
    max_new_tokens: int,
    network_profile: str,
    plot_formats: str,
    plots_disabled: bool,
    params: dict[str, Any],
    tuning_prompts_file: Path | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("run", {})
    config["run"]["id"] = candidate_id
    config["run"]["methods"] = ["target_only", method]
    config["run"]["output_dir"] = str(run_output)
    config.setdefault("data", {})
    config["data"]["sample_count"] = request_count
    config["data"]["use_sample_prompts"] = True
    if tuning_prompts_file is not None:
        config["data"]["sample_prompts"] = str(tuning_prompts_file)
    config.setdefault("generation", {})
    config["generation"]["max_new_tokens"] = max_new_tokens
    config.setdefault("draft", {})
    config["draft"]["worker_count"] = draft_worker_count
    config["transport"] = dict(NETWORK_PROFILES[network_profile])
    config.setdefault("plots", {})
    config["plots"]["formats"] = plot_formats
    config["plots"]["disabled"] = plots_disabled
    _apply_method_params(config, method=method, params=params)
    return config


def _apply_method_params(config: dict[str, Any], *, method: str, params: dict[str, Any]) -> None:
    if method == "specedge_pipeline":
        config.setdefault("tree", {})
        config.setdefault("pipeline", {})
        config.setdefault("specedge", {})
        config["tree"]["max_depth"] = int(params["depth"])
        config["tree"]["branch_width"] = int(params["branch_width"])
        config["tree"]["max_budget"] = int(params["max_budget"])
        config["pipeline"]["max_depth"] = max(
            int(config["pipeline"].get("max_depth", params["depth"])),
            int(params["depth"]),
        )
        config["pipeline"]["proactive_depth"] = int(params["proactive_depth"])
        config["specedge"]["official"] = bool(params["official"])
        return
    if method == "sled_async":
        config.setdefault("sled", {})
        config["sled"]["strict"] = True
        config["sled"]["batch_size"] = int(params["batch_size"])
        config["sled"]["max_speculation_tokens"] = int(params["max_speculation_tokens"])
        config["sled"]["confidence_threshold"] = float(params["confidence_threshold"])
        config["sled"].setdefault("async", {})
        config["sled"]["async"]["proactive_tokens"] = int(params["proactive_tokens"])
        config["sled"].setdefault("static_queue", {})
        config["sled"]["static_queue"]["enabled"] = True
        config["sled"]["static_queue"]["pad_to_max_length"] = True
        if params.get("queue_max_wait_ms") is None:
            config["sled"]["static_queue"].pop("max_wait_ms", None)
        else:
            config["sled"]["static_queue"]["max_wait_ms"] = float(params["queue_max_wait_ms"])
        return
    if method == "dip_sd":
        config.setdefault("dip_sd", {})
        config["dip_sd"]["solver"] = str(params["solver"])
        config["dip_sd"]["max_draft_length"] = int(params["max_draft_length"])
        config["dip_sd"]["initial_draft_length"] = min(7, int(params["max_draft_length"]))
        config["dip_sd"]["plan_cache_enabled"] = True
        config["dip_sd"]["steady_state_enabled"] = True
        min_batch, max_batch = params["batch_count"]
        config["dip_sd"]["min_batch_count"] = int(min_batch)
        config["dip_sd"]["max_batch_count"] = 0 if max_batch is None else int(max_batch)
        profile = params.get("calibration_profile")
        if profile:
            profile_path = Path(str(profile)).expanduser()
            if not profile_path.is_absolute():
                profile_path = profile_path.resolve()
            config["dip_sd"]["calibration_profile"] = str(profile_path)
            config["dip_sd"]["calibration_enabled"] = True
        else:
            config["dip_sd"].pop("calibration_profile", None)
            config["dip_sd"]["calibration_enabled"] = False
        return
    raise ValueError(f"Unsupported tuning method: {method}")


def _specedge_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    official_values = _bool_list(args.specedge_official)
    return [
        {
            "depth": depth,
            "branch_width": branch_width,
            "max_budget": max_budget,
            "proactive_depth": proactive_depth,
            "official": official,
        }
        for depth, branch_width, max_budget, proactive_depth, official in itertools.product(
            _int_list(args.specedge_depths),
            _int_list(args.specedge_branch_widths),
            _int_list(args.specedge_max_budgets),
            _int_list(args.specedge_proactive_depths),
            official_values,
        )
    ]


def _sled_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "max_speculation_tokens": max_tokens,
            "confidence_threshold": confidence,
            "batch_size": batch_size,
            "proactive_tokens": proactive_tokens,
            "queue_max_wait_ms": queue_wait,
        }
        for max_tokens, confidence, batch_size, proactive_tokens, queue_wait in itertools.product(
            _int_list(args.sled_max_speculation_tokens),
            _float_list(args.sled_confidence_thresholds),
            _int_list(args.sled_batch_sizes),
            _int_list(args.sled_proactive_tokens),
            _optional_float_list(args.sled_queue_max_wait_ms),
        )
    ]


def _dip_sd_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "max_draft_length": max_draft_length,
            "solver": solver,
            "batch_count": batch_count,
            "calibration_profile": calibration_profile,
        }
        for max_draft_length, solver, batch_count, calibration_profile in itertools.product(
            _int_list(args.dip_sd_max_draft_lengths),
            _str_list(args.dip_sd_solvers),
            _batch_count_list(args.dip_sd_batch_counts),
            _calibration_profile_list(args.dip_sd_calibration_profiles),
        )
    ]


def _planned_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "status": "planned",
        "matches_target_only": "",
        "effective_total_ms": "",
        "wstgr_tokens_per_s": "",
        "speedup_vs_target_only": "",
        "config_path": candidate["config_path"],
        "run_output": candidate["run_output"],
        "params_json": json.dumps(candidate["params"], sort_keys=True),
    }


def _summary_row(candidate: dict[str, Any], *, summary_path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    method = candidate["method"]
    efficiency = dict((payload.get("method_efficiency") or {}).get(method) or {})
    target_efficiency = dict((payload.get("method_efficiency") or {}).get("target_only") or {})
    effective_total = _effective_total_ms(efficiency)
    target_total = _effective_total_ms(target_efficiency)
    matches_target = (payload.get("matches_target_only") or {}).get(method)
    speedup = target_total / effective_total if target_total and effective_total else None
    row = _planned_row(candidate)
    row.update(
        {
            "status": status,
            "matches_target_only": matches_target,
            "effective_total_ms": _round_or_blank(effective_total),
            "wstgr_tokens_per_s": _round_or_blank(efficiency.get("wstgr_tokens_per_s")),
            "speedup_vs_target_only": _round_or_blank(speedup),
            "target_only_effective_total_ms": _round_or_blank(target_total),
        }
    )
    return row


def _write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    base_config: dict[str, Any],
    prompt_split: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tuning_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "tuning_summary.csv", rows)
    best_rows = _best_rows(rows)
    (output_dir / "best_tuning.json").write_text(
        json.dumps(best_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    locked_config = _locked_config(base_config, best_rows, prompt_split=prompt_split)
    if locked_config is not None:
        locked_path = output_dir / "locked_three_method_compare.yaml"
        locked_path.write_text(
            yaml.safe_dump(locked_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    (output_dir / "next_commands.md").write_text(
        _next_commands_text(
            output_dir,
            locked_config_written=locked_config is not None,
            prompt_split=prompt_split,
        ),
        encoding="utf-8",
    )


def _best_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        if row.get("matches_target_only") is not True:
            continue
        effective = _optional_float(row.get("effective_total_ms"))
        if effective is None or effective <= 0:
            continue
        method = str(row.get("method"))
        incumbent = best.get(method)
        incumbent_effective = None if incumbent is None else _optional_float(incumbent.get("effective_total_ms"))
        if incumbent is None or incumbent_effective is None or effective < incumbent_effective:
            best[method] = dict(row)
    return best


def _locked_config(
    base_config: dict[str, Any],
    best_rows: dict[str, dict[str, Any]],
    *,
    prompt_split: dict[str, Any],
) -> dict[str, Any] | None:
    if not all(method in best_rows for method in TUNED_METHODS):
        return None
    heldout_prompts_file = prompt_split.get("heldout_prompts_file")
    if heldout_prompts_file is None:
        return None
    config = deepcopy(base_config)
    config.setdefault("run", {})
    config["run"]["id"] = "three_method_locked_compare"
    config["run"]["methods"] = ["target_only", "specedge_pipeline", "sled_async", "dip_sd"]
    config["run"]["output_dir"] = "experiments/three_method_locked_compare/latest"
    config.setdefault("data", {})
    config["data"]["sample_prompts"] = str(heldout_prompts_file)
    config["data"]["sample_count"] = int(prompt_split["heldout_request_count"])
    config["data"]["use_sample_prompts"] = True
    config["data"]["split"] = {
        "tuning_sample_prompts": None
        if prompt_split.get("tuning_prompts_file") is None
        else str(prompt_split["tuning_prompts_file"]),
        "heldout_sample_prompts": str(heldout_prompts_file),
        "heldout_request_count": int(prompt_split["heldout_request_count"]),
    }
    for method, row in best_rows.items():
        params = json.loads(str(row["params_json"]))
        _apply_method_params(config, method=method, params=_restore_params(params))
    return config


def _restore_params(params: dict[str, Any]) -> dict[str, Any]:
    restored = dict(params)
    if "batch_count" in restored:
        restored["batch_count"] = tuple(restored["batch_count"])
    return restored


def _next_commands_text(output_dir: Path, *, locked_config_written: bool, prompt_split: dict[str, Any]) -> str:
    locked_path = output_dir / "locked_three_method_compare.yaml"
    text = [
        "# Three-Method Tuning Next Commands",
        "",
        "## Prompt Split",
        "",
        f"- tuning prompts: {prompt_split.get('tuning_prompts_file') or 'base config data.sample_prompts'}",
        f"- held-out prompts: {prompt_split.get('heldout_prompts_file') or 'missing'}",
        f"- held-out request count: {prompt_split.get('heldout_request_count')}",
        "",
        "## Tune",
        "",
        "```bash",
        "PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \\",
        "  scripts/run_method_tuning.py \\",
        f"  --output-dir {output_dir} \\",
        "  --continue-on-error",
        "```",
        "",
    ]
    if locked_config_written:
        text.extend(
            [
                "## Evaluate Locked Config",
                "",
                "```bash",
                "PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \\",
                "  scripts/run_experiment_matrix.py \\",
                f"  --base-config {locked_path} \\",
                "  --output-dir experiments/three_method_locked_matrix/latest \\",
                "  --methods target_only,specedge_pipeline,sled_async,dip_sd \\",
                "  --request-counts 1,2,4,8,16 \\",
                "  --draft-worker-counts 1,2,4,8 \\",
                "  --max-new-token-counts 8,16,32,64 \\",
                "  --depths locked \\",
                "  --network-profiles observe,low_uplink,high_rtt \\",
                "  --resume --rerun-mismatches --continue-on-error",
                "```",
                "",
            ]
        )
    else:
        text.extend(
            [
                "## Evaluate Locked Config",
                "",
                "No locked config was written yet. Finish at least one correctness-clean candidate for each method.",
                "Also make sure a held-out prompt file is available so final evaluation does not reuse tuning prompts.",
                "",
            ]
        )
    return "\n".join(text)


def _prompt_split(args: argparse.Namespace, base_config: dict[str, Any]) -> dict[str, Any]:
    tuning_prompts = _optional_path(args.tuning_prompts_file)
    if tuning_prompts is None:
        tuning_prompts = _optional_path(_nested(base_config, "data", "sample_prompts"))

    heldout_prompts = _optional_path(args.heldout_prompts_file)
    if heldout_prompts is None:
        heldout_prompts = _optional_path(_nested(base_config, "data", "heldout_sample_prompts"))
    if heldout_prompts is None:
        default_heldout = Path("data/sample_prompts_heldout.jsonl")
        if default_heldout.exists():
            heldout_prompts = default_heldout

    return {
        "tuning_prompts_file": tuning_prompts,
        "heldout_prompts_file": heldout_prompts,
        "heldout_request_count": int(args.heldout_request_count or args.request_count),
    }


def _run_candidate(command: list[str], *, log_path: Path, stream_output: bool) -> None:
    if stream_output:
        subprocess.run(command, check=True)
        return
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["candidate_id", "method", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _effective_total_ms(efficiency: dict[str, Any]) -> float | None:
    for key in ("effective_total_ms", "runtime_round_total_ms", "http_total_ms"):
        value = _optional_float(efficiency.get(key))
        if value is not None and value > 0:
            return value
    return None


def _param_slug(params: dict[str, Any]) -> str:
    parts = []
    for key in sorted(params):
        value = params[key]
        if value is None:
            value = "none"
        elif isinstance(value, tuple):
            value = "-".join("auto" if item is None else str(item) for item in value)
        elif isinstance(value, bool):
            value = "on" if value else "off"
        parts.append(f"{_short_key(key)}{str(value).replace('.', 'p').replace('/', '_')}")
    return "_".join(parts)


def _short_key(key: str) -> str:
    return {
        "batch_count": "bc",
        "batch_size": "bs",
        "branch_width": "bw",
        "calibration_profile": "cal",
        "confidence_threshold": "ct",
        "depth": "d",
        "max_budget": "mb",
        "max_draft_length": "mdl",
        "max_speculation_tokens": "mst",
        "official": "off",
        "proactive_depth": "pd",
        "proactive_tokens": "pt",
        "queue_max_wait_ms": "qmw",
        "solver": "sol",
    }.get(key, f"{key}=")


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _optional_float_list(value: str) -> list[float | None]:
    result: list[float | None] = []
    for part in str(value).split(","):
        raw = part.strip()
        if not raw:
            continue
        if raw.lower() in {"none", "null", "off"}:
            result.append(None)
        else:
            result.append(float(raw))
    return result or [None]


def _str_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _bool_list(value: str) -> list[bool]:
    mapping = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}
    result: list[bool] = []
    for part in str(value).split(","):
        raw = part.strip().lower()
        if not raw:
            continue
        if raw not in mapping:
            raise ValueError(f"Invalid boolean grid value: {part!r}")
        result.append(mapping[raw])
    return result or [True]


def _batch_count_list(value: str) -> list[tuple[int, int | None]]:
    result: list[tuple[int, int | None]] = []
    for part in str(value).split(","):
        raw = part.strip().lower()
        if not raw:
            continue
        if raw in {"auto", "none"}:
            result.append((2, None))
            continue
        if "-" in raw:
            left, right = raw.split("-", 1)
            result.append((int(left), None if right in {"auto", "none", "0"} else int(right)))
            continue
        count = int(raw)
        result.append((count, count))
    return result or [(2, None)]


def _calibration_profile_list(value: str) -> list[str | None]:
    profiles: list[str | None] = []
    for part in str(value).split(","):
        raw = part.strip()
        if not raw:
            continue
        if raw.lower() in {"none", "off", "false"}:
            profiles.append(None)
        else:
            profiles.append(raw)
    return profiles or [None]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _round_or_blank(value: Any) -> float | str:
    number = _optional_float(value)
    if number is None:
        return ""
    return round(number, 6)


if __name__ == "__main__":
    main()
