from __future__ import annotations

"""从已有 timing artifacts 离线生成诊断图。"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from specplatform.core import PhaseEvent
from specplatform.metrics import write_timing_charts


def build_parser() -> argparse.ArgumentParser:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="Render fixed timing or matrix comparison charts from artifacts.")
    parser.add_argument("--mode", choices=("single", "matrix"), default="single")
    parser.add_argument("--input-dir", default="experiments/3090_a100_smoke/latest")
    parser.add_argument("--events-file", default=None)
    parser.add_argument("--matrix-summary", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument(
        "--local-results-dir",
        default=None,
        help="Optional server-side directory that receives a timestamped copy of summaries and plots.",
    )
    parser.add_argument("--result-zip-dir", default="transfer")
    parser.add_argument("--no-result-zip", action="store_true")
    parser.add_argument("--sync-dest", default=None)
    return parser


def main() -> None:
    """读取 phase events，生成图表，并可选同步结果目录。"""
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
    if args.mode == "single":
        events_path = Path(args.events_file) if args.events_file else input_dir / "phase_events.jsonl"
        events = _read_phase_events_jsonl(events_path)
        written = write_timing_charts(events, output_dir, formats=formats, mode="single_result")
    else:
        summary_path = Path(args.matrix_summary) if args.matrix_summary else input_dir / "matrix_summary.json"
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        matrix_runner = _load_matrix_runner_module()
        written = matrix_runner._write_matrix_plots(rows, output_dir, formats=formats)
    print("mode:", args.mode)
    print("charts_dir:", output_dir)
    print("charts:", sum(len(paths) for paths in written.values()))
    matrix_runner = None
    if args.local_results_dir:
        matrix_runner = _load_matrix_runner_module()
        local_export_dir = matrix_runner._export_result_bundle(input_dir, Path(args.local_results_dir))
        print("server_results_dir:", local_export_dir)
    if not args.no_result_zip:
        if matrix_runner is None:
            matrix_runner = _load_matrix_runner_module()
        result_zip = matrix_runner._write_result_zip_bundle(input_dir, Path(args.result_zip_dir))
        print("result_zip:", result_zip)
    if args.sync_dest:
        matrix_runner = _load_matrix_runner_module()
        _sync_dir(input_dir, args.sync_dest)
        print("synced_to:", args.sync_dest)


def _read_phase_events_jsonl(path: Path) -> list[PhaseEvent]:
    """读取 EventLogger.write_jsonl 写出的 PhaseEvent。"""
    events: list[PhaseEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(PhaseEvent(**json.loads(line)))
    return events


def _load_matrix_runner_module() -> ModuleType:
    """Load the matrix renderer without requiring scripts/ to be a package."""
    path = Path(__file__).resolve().with_name("run_experiment_matrix.py")
    spec = importlib.util.spec_from_file_location("experiment_matrix_runner_for_plots", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load matrix runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _sync_dir(input_dir: Path, sync_dest: str) -> None:
    """用 rsync 同步整个实验输出目录。"""
    source = str(input_dir) + "/"
    subprocess.run(["rsync", "-a", source, sync_dest], check=True)


if __name__ == "__main__":
    main()
