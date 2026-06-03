from __future__ import annotations

"""Simulate SLED Poisson multi-device arrivals and central static batching."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from specplatform.schedulers import (
    PoissonArrivalConfig,
    StaticQueueBatchPlanner,
    generate_poisson_arrivals,
    summarize_queue_batches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SLED Poisson arrival + queue/batch planner simulation.")
    parser.add_argument("--device-counts", default="4,8,16,32")
    parser.add_argument("--arrival-rates", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--draft-lengths", default="2,4,8")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="experiments/sled_poisson_queue_sim")
    parser.add_argument("--plot-formats", default="png,svg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _simulate_rows(args)
    _write_rows(rows, output_dir)
    _write_plots(rows, output_dir / "plots", formats=_csv_values(args.plot_formats, str))
    print("output_dir:", output_dir)
    print("row_count:", len(rows))


def _simulate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device_counts = _csv_values(args.device_counts, int)
    arrival_rates = _csv_values(args.arrival_rates, float)
    draft_lengths = _csv_values(args.draft_lengths, int)
    for device_count in device_counts:
        for arrival_rate in arrival_rates:
            for draft_length in draft_lengths:
                config = PoissonArrivalConfig(
                    device_count=device_count,
                    arrival_rate_per_device_s=arrival_rate,
                    duration_s=float(args.duration_s),
                    seed=int(args.seed) + device_count * 1000 + int(arrival_rate * 100) + draft_length,
                    draft_length=draft_length,
                )
                arrivals = generate_poisson_arrivals(config)
                planner = StaticQueueBatchPlanner(
                    batch_size=int(args.batch_size),
                    max_wait_ms=float(args.max_wait_ms),
                )
                batches = planner.plan(arrivals)
                summary = summarize_queue_batches(batches)
                rows.append(
                    {
                        "device_count": device_count,
                        "arrival_rate_per_device_s": arrival_rate,
                        "total_arrival_rate_s": device_count * arrival_rate,
                        "duration_s": float(args.duration_s),
                        "draft_length": draft_length,
                        "batch_size": int(args.batch_size),
                        "max_wait_ms": float(args.max_wait_ms),
                        **summary,
                    }
                )
    return rows


def _write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    json_path = output_dir / "sled_poisson_queue_summary.json"
    csv_path = output_dir / "sled_poisson_queue_rows.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plots(rows: list[dict[str, Any]], output_dir: Path, *, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting environment
        (output_dir / "plot_error.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    builders = {
        "sled_queue_throughput_capacity": lambda: _line_plot(
            plt,
            rows,
            x_key="device_count",
            y_key="throughput_requests_per_s",
            series_key="arrival_rate_per_device_s",
            xlabel="device count",
            ylabel="throughput requests/s",
            title="SLED Poisson Queue Capacity",
        ),
        "sled_queue_wait_vs_rate": lambda: _line_plot(
            plt,
            rows,
            x_key="total_arrival_rate_s",
            y_key="p95_queue_wait_ms",
            series_key="draft_length",
            xlabel="total arrival rate requests/s",
            ylabel="p95 queue wait ms",
            title="SLED Queue Wait Under Poisson Arrivals",
        ),
        "sled_queue_batch_fill": lambda: _line_plot(
            plt,
            rows,
            x_key="total_arrival_rate_s",
            y_key="avg_batch_size",
            series_key="device_count",
            xlabel="total arrival rate requests/s",
            ylabel="average batch size",
            title="SLED Static Batch Fill",
        ),
        "sled_queue_padding_overhead": lambda: _line_plot(
            plt,
            rows,
            x_key="draft_length",
            y_key="padding_overhead_ratio",
            series_key="device_count",
            xlabel="draft length",
            ylabel="padding overhead ratio",
            title="SLED Padding Overhead",
        ),
    }
    for name, builder in builders.items():
        fig = builder()
        for fmt in formats:
            fig.savefig(output_dir / f"{name}.{fmt}", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _line_plot(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        x_raw = row.get(x_key)
        y_raw = row.get(y_key)
        if x_raw is None or y_raw is None:
            continue
        series = str(row.get(series_key))
        grouped.setdefault(series, {}).setdefault(float(x_raw), []).append(float(y_raw))
    if not grouped:
        ax.text(0.5, 0.5, f"No rows for {y_key}", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for series, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [sum(points[x]) / len(points[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=series)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title=series_key)
    fig.tight_layout()
    return fig


def _csv_values(raw: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


if __name__ == "__main__":
    main()
