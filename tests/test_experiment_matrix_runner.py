"""Experiment matrix runner helper tests."""

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType


def _load_matrix_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("experiment_matrix_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_experiment_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


class ExperimentMatrixRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_matrix_module()

    def test_matrix_outputs_include_best_method_table(self) -> None:
        rows = [
            _row("run-a", "tree_stop_wait", 100.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-a", "dip_sd", 90.0),
            _row("run-a", "sled", 70.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self.runner._write_matrix_outputs(rows, output_dir)

            self.assertGreater((output_dir / "matrix_summary.csv").stat().st_size, 0)
            self.assertGreater((output_dir / "matrix_summary.json").stat().st_size, 0)
            best_csv = (output_dir / "matrix_best_methods.csv").read_text(encoding="utf-8")
            self.assertIn("sled", best_csv)
            comparison_csv = (output_dir / "matrix_comparison.csv").read_text(encoding="utf-8")
            self.assertIn("specedge_pipeline_speedup_vs_tree_stop_wait", comparison_csv)
            self.assertIn("best_method", comparison_csv)
            aggregate_csv = (output_dir / "matrix_method_aggregate.csv").read_text(encoding="utf-8")
            self.assertIn("mean_speedup_vs_tree_stop_wait", aggregate_csv)
            self.assertIn("mean_setup_total_ms", aggregate_csv)
            self.assertIn("reproduction_execution_modes", aggregate_csv)
            self.assertIn("winning_cell_count", aggregate_csv)
            phase_csv = (output_dir / "matrix_phase_distribution.csv").read_text(encoding="utf-8")
            self.assertIn("mean_phase_draft_pct_of_leaf", phase_csv)
            self.assertIn("specedge_pipeline", phase_csv)
            self.assertGreater((output_dir / "matrix_status.json").stat().st_size, 0)
            report = (output_dir / "matrix_report.md").read_text(encoding="utf-8")
            self.assertIn("Speculative Winner Counts", report)
            self.assertIn("Top Speedup Cells", report)
            self.assertIn("Method Reproduction", report)
            self.assertIn("Phase Distribution", report)

    def test_matrix_plots_are_optional_and_nonempty_when_matplotlib_exists(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib not installed")
        rows = [
            _row("run-a", "tree_stop_wait", 100.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-a", "dip_sd", 90.0),
            _row("run-a", "sled", 70.0),
            _row("run-b", "tree_stop_wait", 120.0, request_count=2),
            _row("run-b", "specedge_pipeline", 110.0, request_count=2),
            _row("run-b", "dip_sd", 95.0, request_count=2),
            _row("run-b", "sled", 105.0, request_count=2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            written = self.runner._write_matrix_plots(rows, output_dir, formats=("png",))

            self.assertEqual(tuple(written), self.runner.MATRIX_COMPARISON_PLOT_NAMES)
            self.assertIn("matrix_runtime_by_method", written)
            self.assertIn("matrix_runtime_by_depth", written)
            self.assertIn("matrix_speedup_vs_target_only", written)
            self.assertIn("matrix_verify_batch_size", written)
            self.assertIn("matrix_phase_distribution", written)
            for paths in written.values():
                for path in paths:
                    self.assertGreater(Path(path).stat().st_size, 0)

    def test_export_result_bundle_copies_summaries_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "matrix-run"
            output_dir.mkdir()
            (output_dir / "matrix_summary.csv").write_text("method,total\n", encoding="utf-8")
            (output_dir / "matrix_report.md").write_text("# report\n", encoding="utf-8")
            plot_dir = output_dir / "plots"
            plot_dir.mkdir()
            (plot_dir / "matrix_runtime_by_method.png").write_bytes(b"plot")
            method_dir = output_dir / "runs" / "rc1" / "specedge_pipeline"
            (method_dir / "plots").mkdir(parents=True)
            (method_dir / "phase_summary.csv").write_text("phase,total\n", encoding="utf-8")
            (method_dir / "plots" / "worker_batch_lanes.png").write_bytes(b"lanes")
            dip_dir = output_dir / "runs" / "rc1" / "dip_sd"
            dip_dir.mkdir(parents=True)
            (dip_dir / "dip_sd_solver_trace.json").write_text("{}\n", encoding="utf-8")
            (dip_dir / "dip_sd_offline_plan_table.json").write_text("{}\n", encoding="utf-8")
            (dip_dir / "dip_sd_stage_plan.csv").write_text("stage_index\n", encoding="utf-8")
            (dip_dir / "pipeline_stage_timeline.csv").write_text("phase\n", encoding="utf-8")

            export_dir = self.runner._export_result_bundle(output_dir, root / "specdec_results")

            self.assertTrue((export_dir / "matrix_summary.csv").exists())
            self.assertTrue((export_dir / "matrix_report.md").exists())
            self.assertTrue((export_dir / "plots" / "matrix_runtime_by_method.png").exists())
            self.assertTrue((export_dir / "runs" / "rc1" / "specedge_pipeline" / "phase_summary.csv").exists())
            self.assertTrue(
                (export_dir / "runs" / "rc1" / "specedge_pipeline" / "plots" / "worker_batch_lanes.png").exists()
            )
            self.assertTrue((export_dir / "runs" / "rc1" / "dip_sd" / "dip_sd_solver_trace.json").exists())
            self.assertTrue((export_dir / "runs" / "rc1" / "dip_sd" / "dip_sd_offline_plan_table.json").exists())
            self.assertTrue((export_dir / "runs" / "rc1" / "dip_sd" / "dip_sd_stage_plan.csv").exists())
            self.assertTrue((export_dir / "runs" / "rc1" / "dip_sd" / "pipeline_stage_timeline.csv").exists())
            self.assertTrue((export_dir / "export_manifest.json").exists())

    def test_write_result_zip_bundle_includes_summaries_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "matrix-run"
            output_dir.mkdir()
            (output_dir / "matrix_summary.csv").write_text("method,total\n", encoding="utf-8")
            plot_dir = output_dir / "plots"
            plot_dir.mkdir()
            (plot_dir / "matrix_runtime_by_method.png").write_bytes(b"plot")
            method_dir = output_dir / "runs" / "rc1" / "sled_async"
            (method_dir / "plots").mkdir(parents=True)
            (method_dir / "plots" / "worker_batch_lanes.png").write_bytes(b"lanes")
            dip_dir = output_dir / "runs" / "rc1" / "dip_sd"
            dip_dir.mkdir(parents=True)
            (dip_dir / "dip_sd_solver_trace.json").write_text("{}\n", encoding="utf-8")
            (dip_dir / "dip_sd_offline_plan_table.json").write_text("{}\n", encoding="utf-8")
            (dip_dir / "dip_sd_stage_plan.csv").write_text("stage_index\n", encoding="utf-8")
            (dip_dir / "pipeline_stage_timeline.csv").write_text("phase\n", encoding="utf-8")

            zip_path = self.runner._write_result_zip_bundle(output_dir, root / "transfer")

            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
            self.assertIn("matrix-run/matrix_summary.csv", names)
            self.assertIn("matrix-run/plots/matrix_runtime_by_method.png", names)
            self.assertIn("matrix-run/runs/rc1/sled_async/plots/worker_batch_lanes.png", names)
            self.assertIn("matrix-run/runs/rc1/dip_sd/dip_sd_solver_trace.json", names)
            self.assertIn("matrix-run/runs/rc1/dip_sd/dip_sd_offline_plan_table.json", names)
            self.assertIn("matrix-run/runs/rc1/dip_sd/dip_sd_stage_plan.csv", names)
            self.assertIn("matrix-run/runs/rc1/dip_sd/pipeline_stage_timeline.csv", names)

    def test_effective_total_uses_http_for_target_only(self) -> None:
        row = _row("run-a", "target_only", 0.0, http_ms=123.0)
        self.assertEqual(self.runner._effective_total_ms(row), 123.0)

    def test_comparison_rows_widen_method_metrics_per_cell(self) -> None:
        rows = [
            _row("run-a", "target_only", 0.0, http_ms=160.0),
            _row("run-a", "tree_stop_wait", 100.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-a", "dip_sd", 50.0),
            _row("run-a", "sled", 125.0),
        ]

        [comparison] = self.runner._comparison_rows(rows)

        self.assertEqual(comparison["run_id"], "run-a")
        self.assertEqual(comparison["best_method"], "dip_sd")
        self.assertEqual(comparison["specedge_pipeline_speedup_vs_tree_stop_wait"], 1.25)
        self.assertEqual(comparison["dip_sd_speedup_vs_target_only"], 3.2)
        self.assertEqual(comparison["best_speedup_vs_tree_stop_wait"], 2.0)
        self.assertEqual(comparison["setup_total_ms"], 35.0)
        self.assertEqual(comparison["specedge_pipeline_setup_total_ms"], 35.0)
        self.assertEqual(comparison["specedge_pipeline_reproduction_execution_mode"], "async_pipeline")
        self.assertEqual(comparison["dip_sd_reproduction_partial_or_missing_count"], 1)

    def test_comparison_rows_alias_sled_async_to_sled_columns(self) -> None:
        rows = [
            _row("run-a", "target_only", 0.0, http_ms=160.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-a", "sled_async", 70.0),
        ]

        [comparison] = self.runner._comparison_rows(rows)

        self.assertEqual(comparison["sled_effective_total_ms"], 70.0)
        self.assertEqual(comparison["sled_speedup_vs_target_only"], 160.0 / 70.0)
        self.assertEqual(comparison["best_method"], "sled")
        self.assertEqual(comparison["best_speculative_method"], "sled")

        report = self.runner._matrix_report_text(
            rows,
            best_rows=self.runner._best_method_rows(rows),
            comparison_rows=[comparison],
            aggregate_rows=self.runner._aggregate_method_rows(rows),
            phase_rows=self.runner._phase_distribution_rows(rows),
            status=self.runner._matrix_status(rows),
        )
        self.assertIn("### vs target_only", report)
        self.assertIn("#### sled", report)
        self.assertNotIn("No speedup rows.", report)

    def test_comparison_rows_best_method_includes_target_only(self) -> None:
        rows = [
            _row("run-a", "target_only", 0.0, http_ms=60.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-a", "sled_async", 70.0),
        ]

        [comparison] = self.runner._comparison_rows(rows)

        self.assertEqual(comparison["best_method"], "target_only")
        self.assertEqual(comparison["best_effective_total_ms"], 60.0)
        self.assertEqual(comparison["best_speculative_method"], "sled")
        self.assertEqual(comparison["best_speculative_effective_total_ms"], 70.0)

    def test_report_includes_depth_effect_when_depth_varies(self) -> None:
        rows = [
            _row("d2", "target_only", 0.0, http_ms=160.0, depth=2),
            _row("d2", "tree_stop_wait", 100.0, depth=2),
            _row("d2", "specedge_pipeline", 80.0, depth=2),
            _row("d4", "target_only", 0.0, http_ms=170.0, depth=4),
            _row("d4", "tree_stop_wait", 120.0, depth=4),
            _row("d4", "specedge_pipeline", 90.0, depth=4),
        ]
        comparison_rows = self.runner._comparison_rows(rows)
        report = self.runner._matrix_report_text(
            rows,
            best_rows=self.runner._best_method_rows(rows),
            comparison_rows=comparison_rows,
            aggregate_rows=self.runner._aggregate_method_rows(rows),
            phase_rows=self.runner._phase_distribution_rows(rows),
            status=self.runner._matrix_status(rows),
        )

        self.assertIn("Depth Effect", report)
        self.assertIn("| d2 |", report)
        self.assertIn("| d4 |", report)

    def test_summary_rows_preserve_setup_metrics_without_changing_effective_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_output = Path(tmp)
            (run_output / "combined_summary.json").write_text(
                """
{
  "matches_target_only": {"specedge_pipeline": true},
  "method_efficiency": {
    "specedge_pipeline": {
      "runtime_round_total_ms": 80.0,
      "setup_load_draft_model_ms": 1000.0,
      "setup_warm_draft_workers_ms": 200.0,
      "setup_warm_draft_worker_event_count": 8,
      "setup_total_ms": 1200.0
    }
  },
  "method_reproduction": {
    "specedge_pipeline": {
      "reference_scope": "SpecEdge original core",
      "execution_mode": "async_pipeline",
      "implemented": ["proactive edge drafting", "draft/verify overlap"],
      "partial_or_missing": [],
      "not_counted_as_original": [],
      "signals": {"scheduler_method_family": null}
    }
  }
}
""".strip(),
                encoding="utf-8",
            )

            [row] = self.runner._summary_rows(
                "run-a",
                8,
                8,
                "explicit",
                "model_size",
                8,
                "observe",
                run_output,
                worker_summary={},
            )

        self.assertEqual(row["setup_load_draft_model_ms"], 1000.0)
        self.assertEqual(row["setup_warm_draft_workers_ms"], 200.0)
        self.assertEqual(row["setup_warm_draft_worker_event_count"], 8)
        self.assertEqual(row["setup_total_ms"], 1200.0)
        self.assertEqual(row["effective_total_ms"], 80.0)
        self.assertEqual(row["reproduction_execution_mode"], "async_pipeline")
        self.assertEqual(row["reproduction_implemented_count"], 2)
        self.assertEqual(row["reproduction_partial_or_missing_count"], 0)

    def test_summary_rows_include_system_leaf_phase_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_output = Path(tmp)
            (run_output / "combined_summary.json").write_text(
                """
{
  "matches_target_only": {"specedge_pipeline": true},
  "method_efficiency": {
    "specedge_pipeline": {"runtime_round_total_ms": 80.0}
  }
}
""".strip(),
                encoding="utf-8",
            )
            method_dir = run_output / "specedge_pipeline"
            method_dir.mkdir()
            (method_dir / "phase_summary.csv").write_text(
                "\n".join(
                    [
                        "summary_view,method,phase,phase_category,event_scope,span_kind,count,total_measured_duration_ms,total_attributed_duration_ms,mean_measured_duration_ms,mean_attributed_duration_ms",
                        "system_leaf_summary,specedge_pipeline,draft.generate,draft,system,leaf,1,30.0,30.0,30.0,30.0",
                        "system_leaf_summary,specedge_pipeline,draft.topk,draft,system,detail,1,999.0,999.0,999.0,999.0",
                        "system_leaf_summary,specedge_pipeline,verify.batch_total,verify,system,leaf,1,70.0,70.0,70.0,70.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            [row] = self.runner._summary_rows(
                "run-a",
                8,
                8,
                "explicit",
                "model_size",
                8,
                "observe",
                run_output,
                worker_summary={},
            )

        self.assertEqual(row["phase_leaf_total_ms"], 100.0)
        self.assertEqual(row["phase_draft_ms"], 30.0)
        self.assertEqual(row["phase_verify_ms"], 70.0)
        self.assertEqual(row["phase_draft_pct_of_leaf"], 0.3)
        self.assertEqual(row["phase_busy_over_wall_ratio"], 1.25)
        self.assertNotIn("phase_name_draft_topk_ms", row)

    def test_matrix_cell_key_separates_single_and_multi_model_workers(self) -> None:
        rows = [
            _row("single", "tree_stop_wait", 100.0, model_mode="single_model", model_id_count=1),
            _row("single", "specedge_pipeline", 80.0, model_mode="single_model", model_id_count=1),
            _row("multi", "tree_stop_wait", 120.0, model_mode="multi_model", model_id_count=2),
            _row("multi", "specedge_pipeline", 90.0, model_mode="multi_model", model_id_count=2),
        ]

        comparisons = self.runner._comparison_rows(rows)
        status = self.runner._matrix_status(rows)

        self.assertEqual(len(comparisons), 2)
        self.assertEqual(status["completed_cell_count"], 2)
        self.assertEqual(
            sorted(row["draft_worker_model_mode"] for row in comparisons),
            ["multi_model", "single_model"],
        )

    def test_matrix_cell_key_separates_single_and_multi_device_workers(self) -> None:
        rows = [
            _row("cuda0", "tree_stop_wait", 100.0, device_count=1, device_set="cuda:0"),
            _row("cuda0", "specedge_pipeline", 80.0, device_count=1, device_set="cuda:0"),
            _row("cuda01", "tree_stop_wait", 120.0, device_count=2, device_set="cuda:0;cuda:1"),
            _row("cuda01", "specedge_pipeline", 90.0, device_count=2, device_set="cuda:0;cuda:1"),
        ]

        comparisons = self.runner._comparison_rows(rows)
        status = self.runner._matrix_status(rows)

        self.assertEqual(len(comparisons), 2)
        self.assertEqual(status["completed_cell_count"], 2)
        self.assertEqual(
            sorted(row["draft_worker_device_count"] for row in comparisons),
            [1, 2],
        )

    def test_matrix_cell_key_separates_max_new_tokens(self) -> None:
        rows = [
            _row("mt8", "target_only", 0.0, http_ms=80.0, max_new_tokens=8),
            _row("mt8", "sled_async", 40.0, max_new_tokens=8),
            _row("mt16", "target_only", 0.0, http_ms=160.0, max_new_tokens=16),
            _row("mt16", "sled_async", 80.0, max_new_tokens=16),
        ]

        comparisons = self.runner._comparison_rows(rows)
        status = self.runner._matrix_status(rows)

        self.assertEqual(len(comparisons), 2)
        self.assertEqual(status["completed_cell_count"], 2)
        self.assertEqual(status["max_new_tokens"], [8, 16])
        self.assertEqual(
            sorted(row["max_new_tokens"] for row in comparisons),
            [8, 16],
        )

    def test_failure_row_marks_failed_cell_for_continue_on_error(self) -> None:
        row = self.runner._failure_row(
            "run-a",
            1,
            2,
            "explicit",
            "heterogeneous",
            4,
            None,
            "high_rtt",
            Path("/tmp/config.yaml"),
            log_path=Path("/tmp/run-a.log"),
            returncode=7,
        )

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["returncode"], 7)
        self.assertEqual(row["network_profile"], "high_rtt")
        self.assertEqual(row["log_path"], "/tmp/run-a.log")

    def test_run_cell_command_writes_subprocess_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "cell.log"

            self.runner._run_cell_command(
                [sys.executable, "-c", "print('hello-cell')"],
                log_path=log_path,
                stream_output=False,
            )

            self.assertIn("hello-cell", log_path.read_text(encoding="utf-8"))

    def test_summary_has_mismatch_detects_failed_target_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_output = Path(tmp)
            (run_output / "combined_summary.json").write_text(
                '{"matches_target_only": {"specedge_pipeline": true, "sled": false}}',
                encoding="utf-8",
            )

            self.assertTrue(self.runner._summary_has_mismatch(run_output))

    def test_matrix_status_reports_failed_and_incomplete_cells(self) -> None:
        rows = [
            _row("run-a", "tree_stop_wait", 100.0),
            _row("run-a", "specedge_pipeline", 80.0),
            _row("run-b", "tree_stop_wait", 90.0, request_count=2),
            self.runner._failure_row(
                "run-c",
                4,
                1,
                "shared",
                "homogeneous",
                2,
                None,
                "observe",
                Path("/tmp/config.yaml"),
                returncode=3,
            ),
        ]

        status = self.runner._matrix_status(rows)

        self.assertEqual(status["failed_cell_count"], 1)
        self.assertEqual(status["completed_cell_count"], 2)
        self.assertEqual(status["incomplete_cell_count"], 1)

    def test_explicit_worker_mode_generates_registry_configs(self) -> None:
        config = self.runner._matrix_config(
            {
                "models": {"draft": "/models/draft"},
                "draft": {"device": "cuda:0", "backend": "hf_eager", "torch_dtype": "fp16"},
            },
            run_id="run-a",
            run_output=Path("/tmp/run-a"),
            request_count=4,
            worker_count=3,
            worker_mode="explicit",
            worker_speed_profile="heterogeneous",
            depth=2,
            network_profile_name="observe",
            methods=["specedge_pipeline"],
            plot_formats="png",
            disable_plots=True,
        )

        workers = config["draft"]["workers"]
        self.assertEqual(len(workers), 3)
        self.assertEqual(workers[0]["model_path"], "/models/draft")
        self.assertEqual(workers[0]["draft_type"], "both")
        self.assertNotEqual(
            workers[0]["speed_profile"]["relative_speed"],
            workers[2]["speed_profile"]["relative_speed"],
        )
        self.assertIn("quality", workers[0]["speed_profile"])

    def test_locked_depth_preserves_base_tree_and_pipeline_depth(self) -> None:
        config = self.runner._matrix_config(
            {
                "tree": {"max_depth": 4, "branch_width": 8},
                "pipeline": {"max_depth": 8, "proactive_depth": 4},
                "draft": {"device": "cuda:0", "backend": "hf_eager"},
            },
            run_id="run-a",
            run_output=Path("/tmp/run-a"),
            request_count=4,
            worker_count=1,
            worker_mode="shared",
            worker_speed_profile="homogeneous",
            depth=None,
            network_profile_name="observe",
            methods=["specedge_pipeline"],
            plot_formats="png",
            disable_plots=True,
        )

        self.assertEqual(config["tree"]["max_depth"], 4)
        self.assertEqual(config["pipeline"]["max_depth"], 8)
        self.assertEqual(config["pipeline"]["proactive_depth"], 4)
        self.assertEqual(self.runner._configured_depth(config, requested_depth=None), 4)

    def test_depth_values_accept_locked_marker(self) -> None:
        self.assertEqual(self.runner._depth_values("locked,4,base"), [None, 4, None])

    def test_matrix_config_resolves_offline_plan_table_file_relative_to_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config_path = root / "base.yaml"
            base_config_path.write_text("dip_sd: {}\n", encoding="utf-8")

            config = self.runner._matrix_config(
                {
                    "draft": {"device": "cuda:0", "backend": "hf_eager"},
                    "dip_sd": {
                        "solver": "offline_table",
                        "offline_plan_table_file": "plans/table.json",
                        "calibration_profile": "calibration/profile.json",
                    },
                },
                base_config_path=base_config_path,
                run_id="run-a",
                run_output=Path("/tmp/run-a"),
                request_count=2,
                worker_count=1,
                worker_mode="shared",
                worker_speed_profile="homogeneous",
                depth=None,
                network_profile_name="observe",
                methods=["dip_sd"],
                plot_formats="png",
                disable_plots=True,
            )

        self.assertEqual(
            config["dip_sd"]["offline_plan_table_file"],
            str((root / "plans" / "table.json").resolve()),
        )
        self.assertEqual(
            config["dip_sd"]["calibration_profile"],
            str((root / "calibration" / "profile.json").resolve()),
        )

    def test_explicit_worker_mode_reuses_base_worker_templates(self) -> None:
        config = self.runner._matrix_config(
            {
                "models": {"draft": "/models/default"},
                "draft": {
                    "device": "cuda:0",
                    "backend": "hf_eager",
                    "torch_dtype": "fp16",
                    "disable_backend_fallback": True,
                    "workers": [
                        {
                            "worker_id": "edge-0",
                            "model_path": "/models/qwen06",
                            "device": "cuda:0",
                            "backend": "qwen3_graph",
                            "torch_dtype": "fp16",
                            "draft_type": "both",
                            "speed_profile": {"name": "qwen06", "quality": 0.8},
                        },
                        {
                            "worker_id": "edge-1",
                            "model_path": "/models/qwen17",
                            "device": "cuda:1",
                            "backend": "qwen3_graph",
                            "torch_dtype": "fp16",
                            "draft_type": "both",
                            "speed_profile": {"name": "qwen17", "quality": 0.97},
                        },
                    ],
                },
            },
            run_id="run-a",
            run_output=Path("/tmp/run-a"),
            request_count=2,
            worker_count=2,
            worker_mode="explicit",
            worker_speed_profile="homogeneous",
            depth=2,
            network_profile_name="observe",
            methods=["specedge_pipeline", "sled_async", "dip_sd"],
            plot_formats="png",
            disable_plots=True,
        )

        workers = config["draft"]["workers"]
        self.assertEqual([worker["model_path"] for worker in workers], ["/models/qwen06", "/models/qwen17"])
        self.assertEqual([worker["device"] for worker in workers], ["cuda:0", "cuda:1"])
        self.assertEqual([worker["backend"] for worker in workers], ["qwen3_graph", "qwen3_graph"])
        self.assertEqual(workers[0]["speed_profile"]["name"], "qwen06")
        self.assertEqual(config["draft"]["audit"]["required_devices"], ["cuda:0", "cuda:1"])
        self.assertTrue(config["draft"]["audit"]["forbid_backend_fallback"])

    def test_explicit_worker_mode_cycles_real_heterogeneous_model_paths(self) -> None:
        config = self.runner._matrix_config(
            {
                "models": {"draft": "/models/default"},
                "draft": {"device": "cuda:0", "backend": "hf_eager", "torch_dtype": "fp16"},
            },
            run_id="run-a",
            run_output=Path("/tmp/run-a"),
            request_count=4,
            worker_count=3,
            worker_mode="explicit",
            worker_speed_profile="heterogeneous",
            worker_model_paths=["/models/qwen06", "/models/qwen17"],
            worker_devices=["cuda:0", "cuda:1"],
            worker_backends=["hf_cached", "qwen3_graph"],
            worker_torch_dtypes=["fp16"],
            worker_draft_types=["tree", "both"],
            depth=2,
            network_profile_name="observe",
            methods=["specedge_pipeline"],
            plot_formats="png",
            disable_plots=True,
        )

        workers = config["draft"]["workers"]
        self.assertEqual([worker["model_path"] for worker in workers], ["/models/qwen06", "/models/qwen17", "/models/qwen06"])
        self.assertEqual([worker["device"] for worker in workers], ["cuda:0", "cuda:1", "cuda:0"])
        self.assertEqual([worker["backend"] for worker in workers], ["hf_cached", "qwen3_graph", "hf_cached"])
        self.assertEqual([worker["draft_type"] for worker in workers], ["tree", "both", "tree"])
        summary = self.runner._worker_config_summary_from_config(config)
        self.assertEqual(summary["draft_worker_model_mode"], "multi_model")
        self.assertEqual(summary["draft_worker_model_id_count"], 2)
        self.assertEqual(summary["draft_worker_model_names"], "qwen06;qwen17;qwen06")
        self.assertEqual(summary["draft_worker_device_count"], 2)
        self.assertEqual(summary["draft_worker_device_set"], "cuda:0;cuda:1")

    def test_model_size_speed_profile_uses_model_path_scale(self) -> None:
        config = self.runner._matrix_config(
            {
                "models": {"draft": "/models/default"},
                "draft": {"device": "cuda:0", "backend": "hf_eager", "torch_dtype": "fp16"},
            },
            run_id="run-a",
            run_output=Path("/tmp/run-a"),
            request_count=4,
            worker_count=2,
            worker_mode="explicit",
            worker_speed_profile="model_size",
            worker_model_paths=["/models/Qwen3-0.6B", "/models/Qwen3-1.7B"],
            depth=2,
            network_profile_name="observe",
            methods=["specedge_pipeline"],
            plot_formats="png",
            disable_plots=True,
        )

        small_profile = config["draft"]["workers"][0]["speed_profile"]
        large_profile = config["draft"]["workers"][1]["speed_profile"]
        self.assertGreater(small_profile["relative_speed"], large_profile["relative_speed"])
        self.assertLess(small_profile["quality"], large_profile["quality"])
        self.assertEqual(small_profile["metadata"]["model_size_billions"], 0.6)
        self.assertEqual(large_profile["metadata"]["model_size_billions"], 1.7)


def _row(
    run_id: str,
    method: str,
    runtime_ms: float,
    *,
    request_count: int = 1,
    depth: int = 2,
    http_ms: float | None = None,
    max_new_tokens: int | None = None,
    model_mode: str = "single_model",
    model_id_count: int = 1,
    device_count: int = 1,
    device_set: str = "cuda:0",
) -> dict[str, object]:
    effective_ms = runtime_ms if runtime_ms > 0 else float(http_ms or 0.0)
    if method == "target_only":
        phase_values = {
            "scheduler": 0.0,
            "draft": 0.0,
            "verify": 0.0,
            "accept": 0.0,
            "session": 0.0,
            "runtime": effective_ms,
            "other": 0.0,
        }
    else:
        phase_values = {
            "scheduler": effective_ms * 0.02,
            "draft": effective_ms * 0.55,
            "verify": effective_ms * 0.35,
            "accept": effective_ms * 0.02,
            "session": effective_ms * 0.01,
            "runtime": effective_ms * 0.05,
            "other": 0.0,
        }
    phase_total = sum(phase_values.values())
    row = {
        "run_id": run_id,
        "method": method,
        "request_count": request_count,
        "draft_worker_count": 1,
        "draft_worker_mode": "explicit",
        "worker_speed_profile": "homogeneous",
        "draft_worker_model_mode": model_mode,
        "draft_worker_model_id_count": model_id_count,
        "draft_worker_device_count": device_count,
        "draft_worker_devices": device_set,
        "draft_worker_device_set": device_set,
        "draft_worker_backend_set": "hf_eager",
        "draft_worker_draft_type_set": "both",
        "depth": depth,
        "max_new_tokens": max_new_tokens,
        "network_profile": "observe",
        "runtime_round_total_ms": runtime_ms,
        "server_idle_gap_ms": runtime_ms / 10,
        "setup_load_draft_model_ms": 20.0,
        "setup_warm_draft_workers_ms": 15.0,
        "setup_warm_draft_worker_event_count": 1,
        "setup_total_ms": 35.0,
        "matches_target_only": True,
        "reproduction_reference_scope": f"{method} scope",
        "reproduction_execution_mode": "async_pipeline" if method == "specedge_pipeline" else "stop_wait_round_runtime",
        "reproduction_implemented": "core feature",
        "reproduction_implemented_count": 1,
        "reproduction_partial_or_missing": "phase-level pipeline missing" if method == "dip_sd" else "",
        "reproduction_partial_or_missing_count": 1 if method == "dip_sd" else 0,
        "reproduction_not_counted_as_original": "future optimization" if method == "sled" else "",
        "reproduction_not_original_count": 1 if method == "sled" else 0,
        "reproduction_scheduler_method_family": method if method in {"dip_sd", "sled"} else None,
        "phase_leaf_total_ms": phase_total,
        "phase_busy_over_wall_ratio": phase_total / effective_ms if effective_ms > 0 else None,
    }
    for category, value in phase_values.items():
        row[f"phase_{category}_ms"] = value
        row[f"phase_{category}_pct_of_leaf"] = value / phase_total if phase_total > 0 else None
    if http_ms is not None:
        row["http_total_ms"] = http_ms
    return row


if __name__ == "__main__":
    unittest.main()
