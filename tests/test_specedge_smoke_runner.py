"""SpecEdge smoke runner 的轻量行为测试。"""

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from specplatform.core import PhaseEvent
from specplatform.metrics import EventLogger
from specplatform.runtime import RuntimeRequestResult, RuntimeRunResult
from specplatform.timing import TimingRecorder


def _load_runner_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "3090_specedge_smoke.py"
    spec = importlib.util.spec_from_file_location("specedge_smoke_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 3090_specedge_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


class SpecEdgeSmokeRunnerTest(unittest.TestCase):
    """不加载真实模型，只验证 runner helper 的数据契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner_module()

    def test_shared_draft_loader_passes_graph_capture_limits(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeRunner:
            def backend_capabilities(self) -> SimpleNamespace:
                return SimpleNamespace(to_dict=lambda: {"backend_name": "fake"})

        original_loader = self.runner.load_causal_lm_runner

        def fake_loader(model_path: str, **kwargs: object) -> FakeRunner:
            calls.append({"model_path": model_path, **kwargs})
            return FakeRunner()

        self.runner.load_causal_lm_runner = fake_loader
        try:
            registry = self.runner._load_draft_registry(
                {
                    "draft_workers": [],
                    "draft_model_path": "/models/qwen3-small",
                    "draft_backend": "qwen3_graph",
                    "device": "cuda:0",
                    "torch_dtype": "fp16",
                    "device_map": None,
                    "allow_backend_fallback": False,
                    "draft_worker_count": 2,
                    "draft_max_graph_len": 128,
                    "draft_max_graph_tokens": 12,
                    "draft_max_graph_batch_size": 1,
                }
            )
        finally:
            self.runner.load_causal_lm_runner = original_loader

        self.assertEqual(calls[0]["max_graph_len"], 128)
        self.assertEqual(calls[0]["max_graph_tokens"], 12)
        self.assertEqual(calls[0]["max_graph_batch_size"], 1)
        workers = registry.to_metadata()["draft_workers"]
        self.assertEqual(len(workers), 2)
        self.assertEqual(workers[0]["max_graph_len"], 128)
        self.assertEqual(workers[0]["max_graph_tokens"], 12)
        self.assertEqual(workers[0]["max_graph_batch_size"], 1)

    def test_combined_summary_reports_per_request_matches(self) -> None:
        summary = self.runner._build_combined_summary(
            {
                "target_only": {
                    "req-1": [1, 2],
                    "req-2": [3],
                },
                "linear": {
                    "req-1": [1, 2],
                    "req-2": [3],
                },
                "tree": {
                    "req-1": [1, 2],
                    "req-2": [4],
                },
            }
        )

        self.assertTrue(summary["matches_target_only"]["linear"])
        self.assertFalse(summary["matches_target_only"]["tree"])
        self.assertTrue(summary["matches_by_request"]["req-1"]["tree"])
        self.assertFalse(summary["matches_by_request"]["req-2"]["tree"])
        self.assertEqual(summary["request_count"], 2)

    def test_combined_summary_reports_theory_efficiency_metrics(self) -> None:
        tree_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="tree",
                        phase="verify.http_total",
                        duration_ms=10.0,
                        span_kind="detail",
                        metadata={
                            "response_timing": {
                                "target_tree_forward_total_ms": 7.5,
                                "target_tree_forward_events": [
                                    {"kind": "tree_choice_batch", "batch_size": 3, "duration_ms": 7.5}
                                ],
                            }
                        },
                    )
                ]
            ),
        )
        linear_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="linear",
                        phase="verify.http_total",
                        duration_ms=12.0,
                        span_kind="detail",
                        metadata={
                            "response_timing": {
                                "target_forward_total_ms": 9.0,
                                "target_forward_events": [
                                    {"kind": "draft", "duration_ms": 4.5},
                                    {"kind": "draft", "duration_ms": 4.5},
                                ],
                            }
                        },
                    )
                ]
            ),
        )

        summary = self.runner._build_combined_summary(
            {
                "target_only": {"req-1": [1, 2]},
                "linear": {"req-1": [1, 2]},
                "tree": {"req-1": [1, 2]},
            },
            method_results={"linear": linear_result, "tree": tree_result},
        )

        self.assertEqual(summary["method_efficiency"]["tree"]["target_forward_event_count"], 1)
        self.assertEqual(summary["method_efficiency"]["tree"]["tree_choice_prefix_count"], 3)
        self.assertEqual(summary["method_efficiency"]["tree"]["tree_batch_compression_ratio"], 3.0)
        self.assertTrue(summary["theory_checks"]["tree_uses_batched_choice_forward"])
        self.assertTrue(summary["theory_checks"]["tree_target_call_count_lte_linear"])

    def test_theory_metrics_separate_tree_main_path_from_guard_overhead(self) -> None:
        tree_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="tree",
                        phase="verify.http_total",
                        duration_ms=10.0,
                        span_kind="detail",
                        metadata={
                            "response_timing": {
                                "target_tree_forward_total_ms": 10.0,
                                "target_tree_forward_events": [
                                    {
                                        "kind": "tree_attention",
                                        "choice_count": 3,
                                        "duration_ms": 5.0,
                                    },
                                    {
                                        "kind": "tree_root_guard",
                                        "duration_ms": 1.0,
                                    },
                                    {
                                        "kind": "tree_choice_batch",
                                        "batch_size": 3,
                                        "duration_ms": 4.0,
                                        "metadata": {"fallback_reason": "tree_root_guard_mismatch"},
                                    },
                                ],
                            }
                        },
                    )
                ]
            ),
        )
        linear_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="linear",
                        phase="verify.http_total",
                        duration_ms=12.0,
                        span_kind="detail",
                        metadata={
                            "response_timing": {
                                "target_forward_total_ms": 9.0,
                                "target_forward_events": [
                                    {"kind": "draft", "duration_ms": 4.5},
                                    {"kind": "draft", "duration_ms": 4.5},
                                ],
                            }
                        },
                    )
                ]
            ),
        )

        summary = self.runner._build_combined_summary(
            {
                "target_only": {"req-1": [1, 2]},
                "linear": {"req-1": [1, 2]},
                "tree": {"req-1": [1, 2]},
            },
            method_results={"linear": linear_result, "tree": tree_result},
        )

        tree_efficiency = summary["method_efficiency"]["tree"]
        self.assertEqual(tree_efficiency["target_forward_event_count"], 3)
        self.assertEqual(tree_efficiency["target_forward_call_count"], 3)
        self.assertEqual(tree_efficiency["main_target_forward_call_count"], 1)
        self.assertEqual(tree_efficiency["main_tree_choice_prefix_count"], 3)
        self.assertEqual(tree_efficiency["tree_choice_prefix_count"], 6)
        self.assertEqual(tree_efficiency["tree_root_guard_event_count"], 1)
        self.assertEqual(tree_efficiency["tree_corrective_fallback_event_count"], 1)
        self.assertEqual(tree_efficiency["main_tree_batch_compression_ratio"], 3.0)
        self.assertTrue(summary["theory_checks"]["tree_main_target_call_count_lte_linear"])
        self.assertTrue(summary["theory_checks"]["tree_target_call_count_lte_linear"])
        self.assertFalse(summary["theory_checks"]["tree_raw_target_call_count_lte_linear"])
        self.assertEqual(summary["theory_checks"]["tree_call_reduction_vs_linear"], 0.5)
        self.assertEqual(summary["theory_checks"]["tree_raw_call_reduction_vs_linear"], -0.5)

    def test_combined_summary_reports_specedge_pipeline_checks(self) -> None:
        linear_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["l1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="linear",
                        phase="verify.http_total",
                        duration_ms=20.0,
                        span_kind="detail",
                        metadata={
                            "response_timing": {
                                "batch_size": 1,
                                "target_forward_total_ms": 18.0,
                                "target_forward_events": [
                                    {"kind": "draft", "duration_ms": 9.0},
                                    {"kind": "draft", "duration_ms": 9.0},
                                ],
                            }
                        },
                    ),
                ],
            ),
        )
        stop_wait_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["t1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="tree_stop_wait",
                        phase="runtime.round_total",
                        duration_ms=100.0,
                        span_kind="aggregate",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="tree_stop_wait",
                        phase="verify.http_total",
                        duration_ms=20.0,
                        span_kind="detail",
                        start_ns=0,
                        end_ns=20_000_000,
                        metadata={
                            "response_timing": {
                                "batch_size": 2,
                                "target_tree_forward_total_ms": 12.0,
                                "target_tree_forward_events": [
                                    {"kind": "tree_attention", "choice_count": 2, "duration_ms": 12.0}
                                ],
                            }
                        },
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="tree_stop_wait",
                        phase="verify.http_total",
                        duration_ms=20.0,
                        span_kind="detail",
                        start_ns=40_000_000,
                        end_ns=60_000_000,
                        metadata={
                            "response_timing": {
                                "batch_size": 2,
                                "target_tree_forward_total_ms": 12.0,
                                "target_tree_forward_events": [
                                    {"kind": "tree_attention", "choice_count": 2, "duration_ms": 12.0}
                                ],
                            }
                        },
                    ),
                ],
            ),
        )
        pipeline_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="runtime.round_total",
                        duration_ms=90.0,
                        span_kind="aggregate",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="draft.proactive",
                        duration_ms=10.0,
                        span_kind="leaf",
                        start_ns=10_000_000,
                        end_ns=20_000_000,
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="pipeline.reconcile",
                        duration_ms=1.0,
                        span_kind="leaf",
                        metadata={
                            "aligned": True,
                            "reused_token_count": 2,
                            "discarded_token_count": 1,
                        },
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="verify.http_total",
                        duration_ms=30.0,
                        span_kind="detail",
                        start_ns=0,
                        end_ns=30_000_000,
                        metadata={
                            "network_or_queue_residual_ms": 1.0,
                            "response_timing": {
                                "batch_size": 4,
                                "tree_forward_batch_kind": "fallback_sequential",
                                "target_tree_forward_total_ms": 10.0,
                                "target_tree_forward_events": [
                                    {"kind": "tree_attention", "choice_count": 2, "duration_ms": 10.0}
                                ],
                            }
                        },
                    ),
                ],
            ),
        )

        summary = self.runner._build_combined_summary(
            {
                "target_only": {"req-1": [1, 2]},
                "linear": {"req-1": [1, 2]},
                "tree_stop_wait": {"req-1": [1, 2]},
                "specedge_pipeline": {"req-1": [1, 2]},
            },
            method_results={
                "linear": linear_result,
                "tree_stop_wait": stop_wait_result,
                "specedge_pipeline": pipeline_result,
            },
        )

        pipeline_efficiency = summary["method_efficiency"]["specedge_pipeline"]
        self.assertEqual(pipeline_efficiency["runtime_round_total_ms"], 90.0)
        self.assertEqual(pipeline_efficiency["tree_forward_batch_kinds"], ["fallback_sequential"])
        self.assertTrue(summary["theory_checks"]["specedge_has_proactive_overlap"])
        self.assertTrue(summary["theory_checks"]["specedge_avg_verify_batch_size_gt_one"])
        self.assertTrue(summary["theory_checks"]["specedge_main_target_call_count_lte_linear"])
        self.assertTrue(summary["theory_checks"]["specedge_server_idle_gap_lt_stop_wait"])
        self.assertTrue(summary["theory_checks"]["specedge_runtime_total_ms_lte_stop_wait"])
        self.assertTrue(summary["theory_checks"]["specedge_tree_forward_batch_kind_recorded"])
        self.assertTrue(summary["theory_checks"]["specedge_timing_residual_nonnegative"])
        self.assertEqual(
            summary["method_reproduction"]["specedge_pipeline"]["execution_mode"],
            "async_pipeline",
        )
        self.assertIn(
            "proactive edge drafting",
            summary["method_reproduction"]["specedge_pipeline"]["implemented"],
        )

    def test_combined_summary_extracts_nested_qwen3_tree_batch_kind(self) -> None:
        pipeline_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="runtime.round_total",
                        duration_ms=30.0,
                        span_kind="aggregate",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="draft.proactive",
                        duration_ms=5.0,
                        span_kind="leaf",
                        start_ns=0,
                        end_ns=5_000_000,
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="verify.http_total",
                        duration_ms=10.0,
                        span_kind="detail",
                        start_ns=0,
                        end_ns=10_000_000,
                        metadata={
                            "response_timing": {
                                "batch_size": 1,
                                "target_tree_forward_total_ms": 8.0,
                                "target_tree_forward_events": [
                                    {
                                        "kind": "tree_attention_qwen3_graph",
                                        "choice_count": 4,
                                        "duration_ms": 8.0,
                                        "metadata": {
                                            "tree_forward_batch_kind": "tree_attention_batch_qwen3_graph"
                                        },
                                    }
                                ],
                            }
                        },
                    ),
                ]
            ),
        )

        summary = self.runner._build_combined_summary(
            {
                "target_only": {"req-1": [1, 2]},
                "specedge_pipeline": {"req-1": [1, 2]},
            },
            method_results={"specedge_pipeline": pipeline_result},
        )

        pipeline_efficiency = summary["method_efficiency"]["specedge_pipeline"]
        self.assertEqual(
            pipeline_efficiency["tree_forward_batch_kinds"],
            ["tree_attention_batch_qwen3_graph"],
        )
        self.assertEqual(pipeline_efficiency["tree_backend_fallback_event_count"], 0)
        self.assertTrue(summary["theory_checks"]["specedge_tree_forward_batch_kind_recorded"])

    def test_method_reproduction_report_separates_original_scope_from_optimization(self) -> None:
        dip_sd_result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1, 2], proposals=["dip-p1", "dip-p2"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="",
                        method="dip_sd",
                        phase="scheduler.plan",
                        duration_ms=0.1,
                        span_kind="leaf",
                        metadata={
                            "hints_metadata": {
                                "method_family": "dip_sd",
                                "joint_batch_assignment": True,
                                "joint_draft_length": True,
                                "solver_active": True,
                            }
                        },
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="runtime.round_total",
                        duration_ms=40.0,
                        span_kind="aggregate",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="draft.generate",
                        duration_ms=5.0,
                        span_kind="leaf",
                        worker_id="draft-0",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-2",
                        method="dip_sd",
                        phase="draft.generate",
                        duration_ms=5.0,
                        span_kind="leaf",
                        worker_id="draft-1",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="pipeline.stage",
                        duration_ms=25.0,
                        span_kind="aggregate",
                        batch_id="batch0",
                        metadata={"stage_index": 0, "runtime_engine": "distributed_batch_pipeline"},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="verify.http_total",
                        duration_ms=20.0,
                        span_kind="detail",
                        start_ns=0,
                        end_ns=20_000_000,
                        metadata={"response_timing": {"batch_size": 2}},
                    ),
                ],
            ),
        )
        sled_result = SimpleNamespace(
            request_results=[
                SimpleNamespace(output_token_ids=[1, 2], proposals=["sled-p1"]),
                SimpleNamespace(output_token_ids=[3, 4], proposals=["sled-p2"]),
            ],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="",
                        method="sled",
                        phase="scheduler.plan",
                        duration_ms=0.1,
                        span_kind="leaf",
                        metadata={
                            "hints_metadata": {
                                "method_family": "sled",
                                "edge_device_worker_assignment": True,
                                "heterogeneous_worker_assignment": True,
                                "single_edge_device_per_request": True,
                                "dynamic_drafting": True,
                                "confidence_threshold": 0.5,
                            }
                        },
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="sled",
                        phase="runtime.round_total",
                        duration_ms=50.0,
                        span_kind="aggregate",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="sled",
                        phase="draft.generate",
                        duration_ms=5.0,
                        span_kind="leaf",
                        worker_id="edge-a",
                        metadata={"dynamic_drafting": True, "dynamic_stop_reason": "confidence_below_threshold"},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-2",
                        method="sled",
                        phase="draft.generate",
                        duration_ms=6.0,
                        span_kind="leaf",
                        worker_id="edge-b",
                        metadata={"dynamic_drafting": True, "dynamic_stop_reason": "max_tokens"},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="sled",
                        phase="accept.apply",
                        duration_ms=0.1,
                        span_kind="leaf",
                        metadata={"candidate_count": 1, "candidate_winner": True},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="sled",
                        phase="verify.http_total",
                        duration_ms=20.0,
                        span_kind="detail",
                        start_ns=0,
                        end_ns=20_000_000,
                        metadata={"response_timing": {"batch_size": 2}},
                    ),
                ],
            ),
        )

        summary = self.runner._build_combined_summary(
            {
                "target_only": {"req-1": [1, 2]},
                "dip_sd": {"req-1": [1, 2]},
                "sled": {"req-1": [1, 2]},
            },
            method_results={"dip_sd": dip_sd_result, "sled": sled_result},
        )

        dip_status = summary["method_reproduction"]["dip_sd"]
        self.assertEqual(dip_status["execution_mode"], "distributed_batch_pipeline")
        self.assertIn("joint batch assignment planning", dip_status["implemented"])
        self.assertIn("phase-level draft/verify pipeline", dip_status["implemented"])
        self.assertIn(
            "SpecEdge proactive single-head drafting is not part of the DiP-SD reproduction report",
            dip_status["not_counted_as_original"],
        )

        sled_status = summary["method_reproduction"]["sled"]
        self.assertEqual(sled_status["execution_mode"], "stop_wait_round_runtime")
        self.assertIn("single edge-device draft stream per request", sled_status["implemented"])
        self.assertIn("confidence-triggered dynamic drafting", sled_status["implemented"])
        self.assertIn(
            "same-request multi-worker candidate selection is not counted as SLED original reproduction",
            sled_status["not_counted_as_original"],
        )
        self.assertIn(
            "async/proactive overlap is absent from this stop-wait SLED run",
            sled_status["not_counted_as_original"],
        )

    def test_dip_sd_method_artifacts_include_solver_and_pipeline_files(self) -> None:
        events = EventLogger()
        events.record(
            PhaseEvent(
                run_id="run",
                request_id="",
                method="dip_sd",
                phase="scheduler.plan",
                duration_ms=0.1,
                span_kind="leaf",
                metadata={
                    "hints_metadata": {
                        "method_family": "dip_sd",
                        "offline_plan_shape_key": "requests=2|workers=2|max_new=4|remaining=4|prefix<=8",
                        "solver_planned_batch_count": 2,
                        "solver_mode": "enumerate",
                        "requested_solver_mode": "enumerate",
                        "hybrid_single_batch_applied": True,
                        "hybrid_single_batch_reason": "small_request_batching",
                        "dip_sd_solution": {
                            "batch_count": 1,
                            "draft_lengths": {"req-1": 2, "req-2": 3},
                            "throughput_tokens_per_s": 10.0,
                        },
                        "preferred_batch_metadata": [
                            {
                                "stage_index": 0,
                                "planned_batch_count": 1,
                                "request_ids": ["req-1", "req-2"],
                                "max_draft_len": 3,
                                "max_prefix_len": 4,
                                "estimated_verify_ms": 1.0,
                                "estimated_memory_bytes": 2.0,
                            }
                        ],
                    },
                    "draft_lengths": {"req-1": 2, "req-2": 3},
                    "preferred_batches": [["req-1", "req-2"]],
                },
            )
        )
        events.record(
            PhaseEvent(
                run_id="run",
                request_id="req-1",
                method="dip_sd",
                phase="pipeline.stage",
                duration_ms=2.0,
                span_kind="aggregate",
                batch_id="batch0",
                metadata={"stage_index": 0, "runtime_engine": "distributed_batch_pipeline"},
            )
        )
        result = RuntimeRunResult(
            request_results=[RuntimeRequestResult(request_id="req-1", output_token_ids=[1, 2])],
            events=events,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self.runner._write_method_artifacts(
                result,
                output_dir,
                render_plots=False,
                plot_formats=("png",),
                metadata={"method": "dip_sd", "run_id": "run"},
            )

            self.assertTrue((output_dir / "dip_sd_solver_trace.json").exists())
            self.assertIn("throughput_tokens_per_s", (output_dir / "dip_sd_solver_trace.json").read_text())
            self.assertIn("stage_index", (output_dir / "dip_sd_stage_plan.csv").read_text())
            self.assertIn("pipeline.stage", (output_dir / "pipeline_stage_timeline.csv").read_text())
            offline_table = json.loads((output_dir / "dip_sd_offline_plan_table.json").read_text())
            entry = offline_table["entries"]["requests=2|workers=2|max_new=4|remaining=4|prefix<=8"]
            self.assertEqual(entry["request_order"], ["req-1", "req-2"])
            self.assertEqual(entry["draft_lengths"], [2, 3])
            self.assertEqual(entry["preferred_batches"], [[0, 1]])
            self.assertTrue(entry["hybrid_single_batch_applied"])

    def test_dip_sd_offline_plan_table_file_is_loaded_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table_dir = root / "plans"
            table_dir.mkdir()
            table_path = table_dir / "dip_sd_offline_plan_table.json"
            table_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": {
                            "shape": {
                                "draft_lengths": [2, 2],
                                "preferred_batches": [[0, 1]],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "probe.yaml"
            config_path.write_text("dip_sd: {}\n", encoding="utf-8")

            table = self.runner._dip_sd_offline_plan_table_config(
                {"dip_sd": {"offline_plan_table_file": "plans/dip_sd_offline_plan_table.json"}},
                config_path=str(config_path),
            )

        self.assertEqual(table["entries"]["shape"]["draft_lengths"], [2, 2])

    def test_dip_sd_settings_apply_relative_calibration_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("", encoding="utf-8")
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "recommended_method_config": {
                            "dip_sd_draft_c": 1.0,
                            "dip_sd_draft_beta": 2.0,
                            "dip_sd_verify_c": 3.0,
                            "dip_sd_verify_beta": 4.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = self.runner.build_parser().parse_args(["--config", str(config_path)])

            settings = self.runner._settings(
                args,
                {
                    "dip_sd": {
                        "calibration_profile": "profile.json",
                    }
                },
            )

        self.assertEqual(settings["dip_sd_draft_c"], 1.0)
        self.assertEqual(settings["dip_sd_draft_beta"], 2.0)
        self.assertEqual(settings["dip_sd_verify_c"], 3.0)
        self.assertEqual(settings["dip_sd_verify_beta"], 4.0)
        self.assertTrue(settings["dip_sd_calibration_applied"])
        self.assertEqual(settings["dip_sd_calibration_profile"], str(profile_path.resolve()))

    def test_dip_sd_cli_calibration_profile_resolves_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps({"recommended_method_config": {"dip_sd_draft_beta": 9.0}}),
                encoding="utf-8",
            )
            config_path = root / "configs" / "config.yaml"
            config_path.parent.mkdir()
            config_path.write_text("", encoding="utf-8")
            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                args = self.runner.build_parser().parse_args(
                    ["--config", str(config_path), "--dip-sd-calibration-profile", "profile.json"]
                )
                settings = self.runner._settings(args, {})
            finally:
                os.chdir(cwd)

        self.assertEqual(settings["dip_sd_draft_beta"], 9.0)
        self.assertEqual(settings["dip_sd_calibration_profile"], str(profile_path.resolve()))

    def test_warm_draft_registry_records_setup_events_and_resets(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.reset_calls: list[str | None] = []

            def reset(self, request_id: str | None = None) -> None:
                self.reset_calls.append(request_id)

        class FakeTreeRunner:
            runner_id = "draft-a"

            def __init__(self) -> None:
                self.model = FakeModel()
                self.calls: list[dict[str, object]] = []

            def generate_tree(self, **kwargs):
                self.calls.append(dict(kwargs))
                return SimpleNamespace(tree=SimpleNamespace(nodes=[1, 2]))

        class FakeRegistry:
            def __init__(self) -> None:
                self.runner = FakeTreeRunner()

            def runners_for(self, draft_type: str):
                if draft_type != "tree":
                    raise AssertionError(f"unexpected draft_type={draft_type}")
                return {"draft-a": self.runner}

        registry = FakeRegistry()

        events = self.runner._warm_draft_registry(
            settings={
                "run_id": "run",
                "draft_warmup_enabled": True,
                "draft_warmup_tokens": 1,
                "draft_warmup_tree_depth": 1,
                "draft_warmup_branch_width": 2,
                "draft_warmup_max_budget": 2,
            },
            draft_registry=registry,
            prompts=[self.runner.PromptSpec("req-1", "hello", [1, 2, 3])],
            methods=["specedge_pipeline"],
            recorder=TimingRecorder(),
        )

        phases = [event.phase for event in events]
        self.assertEqual(phases[0], "setup.warm_draft_workers")
        self.assertIn("setup.warm_draft_worker", phases)
        child = next(event for event in events if event.phase == "setup.warm_draft_worker")
        self.assertEqual(child.worker_id, "draft-a")
        self.assertEqual(child.metadata["warmup_type"], "tree")
        self.assertEqual(child.metadata["produced_count"], 2)
        self.assertTrue(child.metadata["reset_after_warmup"])
        self.assertEqual(registry.runner.model.reset_calls, ["setup-warmup"])
        self.assertEqual(registry.runner.calls[0]["prefix_ids"], [1, 2, 3])

    def test_warmup_serializes_tasks_that_share_model_state(self) -> None:
        class SharedModel:
            def __init__(self) -> None:
                self.active = False
                self.overlap_count = 0
                self.reset_calls = 0

            def enter(self) -> None:
                if self.active:
                    self.overlap_count += 1
                self.active = True
                time.sleep(0.01)
                self.active = False

            def reset(self, request_id: str | None = None) -> None:
                self.reset_calls += 1

        class FakeGreedyRunner:
            runner_id = "shared-greedy"

            def __init__(self, model: SharedModel) -> None:
                self.model = model

            def generate_tokens(self, **kwargs):
                self.model.enter()
                return SimpleNamespace(tokens=[1])

        class FakeTreeRunner:
            runner_id = "shared-tree"

            def __init__(self, model: SharedModel) -> None:
                self.model = model

            def generate_tree(self, **kwargs):
                self.model.enter()
                return SimpleNamespace(tree=SimpleNamespace(nodes=[1, 2]))

        shared_model = SharedModel()

        results = self.runner._run_draft_warmup_tasks(
            settings={
                "draft_warmup_parallelism": 2,
                "draft_warmup_tokens": 1,
                "draft_warmup_tree_depth": 1,
                "draft_warmup_branch_width": 2,
                "draft_warmup_max_budget": 2,
            },
            tasks=[
                ("greedy", "worker-a", FakeGreedyRunner(shared_model)),
                ("tree", "worker-a", FakeTreeRunner(shared_model)),
            ],
            prefix_ids=[1, 2, 3],
        )

        self.assertEqual([result.draft_type for result in results], ["greedy", "tree"])
        self.assertEqual(shared_model.overlap_count, 0)
        self.assertEqual(shared_model.reset_calls, 2)

    def test_dip_sd_and_sled_warm_greedy_draft_workers(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.reset_calls: list[str | None] = []

            def reset(self, request_id: str | None = None) -> None:
                self.reset_calls.append(request_id)

        class FakeGreedyRunner:
            runner_id = "draft-greedy"

            def __init__(self) -> None:
                self.model = FakeModel()
                self.calls: list[dict[str, object]] = []

            def generate_tokens(self, **kwargs):
                self.calls.append(dict(kwargs))
                return SimpleNamespace(tokens=[7])

        class FakeRegistry:
            def __init__(self) -> None:
                self.runner = FakeGreedyRunner()

            def runners_for(self, draft_type: str):
                if draft_type != "greedy":
                    raise AssertionError(f"unexpected draft_type={draft_type}")
                return {"draft-greedy": self.runner}

        registry = FakeRegistry()

        events = self.runner._warm_draft_registry(
            settings={
                "run_id": "run",
                "draft_warmup_enabled": True,
                "draft_warmup_tokens": 1,
                "draft_warmup_tree_depth": 1,
                "draft_warmup_branch_width": 2,
                "draft_warmup_max_budget": 2,
            },
            draft_registry=registry,
            prompts=[self.runner.PromptSpec("req-1", "hello", [1, 2, 3])],
            methods=["dip_sd", "sled"],
            recorder=TimingRecorder(),
        )

        child = next(event for event in events if event.phase == "setup.warm_draft_worker")
        self.assertEqual(child.metadata["warmup_type"], "greedy")
        self.assertEqual(child.metadata["produced_count"], 1)
        self.assertEqual(registry.runner.model.reset_calls, ["setup-warmup"])
        self.assertEqual(registry.runner.calls[0]["max_tokens"], 1)

    def test_setup_metrics_are_reported_outside_runtime_total(self) -> None:
        result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id=None,
                        method="specedge_smoke",
                        phase="setup.load_draft_model",
                        duration_ms=100.0,
                        span_kind="setup",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id=None,
                        method="specedge_smoke",
                        phase="setup.warm_draft_workers",
                        duration_ms=25.0,
                        span_kind="setup",
                        metadata={"warmup_worker_event_count": 2},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="specedge_pipeline",
                        phase="runtime.round_total",
                        duration_ms=10.0,
                        span_kind="aggregate",
                    ),
                ],
            ),
        )

        metrics = self.runner._method_efficiency_metrics(result)

        self.assertEqual(metrics["runtime_round_total_ms"], 10.0)
        self.assertEqual(metrics["setup_load_draft_model_ms"], 100.0)
        self.assertEqual(metrics["setup_warm_draft_workers_ms"], 25.0)
        self.assertEqual(metrics["setup_total_ms"], 125.0)
        self.assertEqual(metrics["setup_warm_draft_worker_event_count"], 2)

    def test_method_efficiency_reports_dip_sd_steady_state_prefetch(self) -> None:
        result = SimpleNamespace(
            request_results=[SimpleNamespace(output_token_ids=[1], proposals=["p1"])],
            events=SimpleNamespace(
                events=[
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="pipeline.steady_state_prefetch_submit",
                        duration_ms=0.01,
                        span_kind="leaf",
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="pipeline.steady_state_prefetch_reuse",
                        duration_ms=0.01,
                        span_kind="leaf",
                        metadata={"prefetch_hit_count": 1},
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-1",
                        method="dip_sd",
                        phase="draft.generate",
                        duration_ms=1.0,
                        span_kind="leaf",
                        worker_id="w0",
                        tokens_out=1,
                        metadata={
                            "steady_state_prefetch": True,
                            "prefetch_truncated": True,
                            "prefetch_original_draft_length": 3,
                            "prefetch_reused_budget_tokens": 1,
                        },
                    ),
                    PhaseEvent(
                        run_id="run",
                        request_id="req-2",
                        method="dip_sd",
                        phase="draft.prefetch_discard",
                        duration_ms=1.0,
                        span_kind="leaf",
                        worker_id="w0",
                    ),
                ]
            ),
        )

        metrics = self.runner._method_efficiency_metrics(result)

        self.assertEqual(metrics["steady_state_prefetch_submit_count"], 1)
        self.assertEqual(metrics["steady_state_prefetch_reuse_event_count"], 1)
        self.assertEqual(metrics["steady_state_prefetch_reused_draft_count"], 1)
        self.assertEqual(metrics["steady_state_prefetch_truncated_reuse_count"], 1)
        self.assertEqual(metrics["steady_state_prefetch_original_draft_token_count"], 3)
        self.assertEqual(metrics["steady_state_prefetch_reused_draft_token_count"], 1)
        self.assertEqual(metrics["steady_state_prefetch_discard_count"], 1)

    def test_read_prompt_file_supports_jsonl_objects_and_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prompts.jsonl"
            path.write_text(
                json.dumps({"id": "custom", "prompt": "hello"}) + "\n"
                + json.dumps("world") + "\n",
                encoding="utf-8",
            )

            rows = self.runner._read_prompt_file(path)

        self.assertEqual(rows[0]["id"], "custom")
        self.assertEqual(rows[0]["prompt"], "hello")
        self.assertEqual(rows[1]["id"], "sample-001")
        self.assertEqual(rows[1]["prompt"], "world")

    def test_prompt_specs_repeats_sample_prompts_to_requested_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prompts.jsonl"
            path.write_text(
                json.dumps({"id": "a", "prompt": "one"}) + "\n"
                + json.dumps({"id": "b", "prompt": "two"}) + "\n",
                encoding="utf-8",
            )

            prompts = self.runner._prompt_specs(
                {
                    "use_sample_prompts": True,
                    "prompts_file": str(path),
                    "sample_count": 5,
                },
                SimpleNamespace(encode=lambda text: [len(text)]),
            )

        self.assertEqual(len(prompts), 5)
        self.assertEqual(
            [prompt.request_id for prompt in prompts],
            ["a", "b", "a-repeat1", "b-repeat1", "a-repeat2"],
        )
        self.assertEqual([prompt.prompt for prompt in prompts], ["one", "two", "one", "two", "one"])


if __name__ == "__main__":
    unittest.main()
