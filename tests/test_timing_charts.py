"""timing 图表和 audit 行为测试。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from specplatform.core import PhaseEvent
from specplatform.metrics import build_timing_audit, write_timing_charts
from specplatform.metrics import plots as timing_plots


class TimingChartsTest(unittest.TestCase):
    """验证图表模块不影响核心 import，并能消费 PhaseEvent。"""

    def test_build_timing_audit_reports_round_coverage_and_residual_warning(self) -> None:
        events = [
            PhaseEvent(
                event_id="evt_round",
                span_id="span_round",
                run_id="run",
                request_id="",
                method="linear",
                plan_id="run:round0",
                phase="runtime.round_total",
                span_kind="aggregate",
                round=0,
                start_ns=0,
                end_ns=100_000_000,
                duration_ms=100.0,
            ),
            PhaseEvent(
                event_id="evt_draft",
                span_id="span_draft",
                run_id="run",
                request_id="request-1",
                method="linear",
                plan_id="run:round0",
                phase="draft.generate",
                span_kind="leaf",
                round=0,
                start_ns=0,
                end_ns=40_000_000,
                duration_ms=40.0,
            ),
            PhaseEvent(
                event_id="evt_http",
                span_id="span_http",
                run_id="run",
                request_id="request-1",
                method="linear",
                plan_id="run:round0",
                phase="verify.http_total",
                span_kind="detail",
                round=0,
                start_ns=40_000_000,
                end_ns=50_000_000,
                duration_ms=10.0,
                metadata={"network_or_queue_residual_ms": -1.0},
            ),
        ]

        audit = build_timing_audit(events)

        self.assertEqual(audit["system_leaf_count"], 1)
        self.assertEqual(audit["system_detail_count"], 1)
        self.assertEqual(audit["rounds"][0]["system_leaf_sum_ms"], 40.0)
        self.assertTrue(audit["warnings"])

    def test_compact_timeline_buckets_include_proactive_draft(self) -> None:
        events = [
            PhaseEvent(
                event_id="evt_draft",
                span_id="span_draft",
                run_id="run",
                request_id="request-1",
                method="specedge_pipeline",
                plan_id="run:round0",
                phase="draft.generate",
                span_kind="leaf",
                round=0,
                start_ns=0,
                end_ns=10_000_000,
                duration_ms=10.0,
            ),
            PhaseEvent(
                event_id="evt_proactive",
                span_id="span_proactive",
                run_id="run",
                request_id="request-1",
                method="specedge_pipeline",
                plan_id="run:round0",
                phase="draft.proactive",
                span_kind="leaf",
                round=0,
                start_ns=10_000_000,
                end_ns=30_000_000,
                duration_ms=20.0,
            ),
            PhaseEvent(
                event_id="evt_reuse",
                span_id="span_reuse",
                run_id="run",
                request_id="request-1",
                method="specedge_pipeline",
                plan_id="run:round1",
                phase="draft.reuse_proactive",
                span_kind="leaf",
                round=1,
                start_ns=30_000_000,
                end_ns=35_000_000,
                duration_ms=5.0,
            ),
            PhaseEvent(
                event_id="evt_detail",
                span_id="span_detail",
                run_id="run",
                request_id="request-1",
                method="specedge_pipeline",
                plan_id="run:round1",
                phase="draft.topk",
                span_kind="detail",
                round=1,
                start_ns=35_000_000,
                end_ns=36_000_000,
                duration_ms=1.0,
            ),
        ]

        selected = timing_plots._main_leaf_events(events)
        buckets = [timing_plots._compact_timeline_bucket(event.phase) for event in selected]

        self.assertEqual(buckets, ["draft.busy", "draft.busy", "draft.busy"])

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_write_timing_charts_creates_png_and_svg(self) -> None:
        events = [
            PhaseEvent(
                event_id="evt_1",
                span_id="span_1",
                run_id="run",
                request_id="request-1",
                method="linear",
                plan_id="run:round0",
                phase="draft.generate",
                span_kind="leaf",
                round=0,
                start_ns=0,
                end_ns=10_000_000,
                duration_ms=10.0,
            ),
            PhaseEvent(
                event_id="evt_2",
                span_id="span_2",
                run_id="run",
                request_id="request-1",
                method="linear",
                plan_id="run:round0",
                phase="verify.http_total",
                span_kind="detail",
                round=0,
                start_ns=10_000_000,
                end_ns=25_000_000,
                duration_ms=15.0,
                metadata={
                    "client_serialize_ms": 1.0,
                    "client_deserialize_ms": 0.5,
                    "modeled_upload_ms": 2.0,
                    "modeled_downlink_ms": 1.5,
                    "response_timing": {
                        "server_total_ms": 8.0,
                        "target_forward_total_ms": 6.0,
                    },
                    "network_or_queue_residual_ms": 7.0,
                },
            ),
            PhaseEvent(
                event_id="evt_3",
                span_id="span_3",
                run_id="run",
                request_id="request-1",
                method="linear",
                plan_id="run:round1",
                phase="verify.batch_total",
                span_kind="leaf",
                round=1,
                start_ns=25_000_000,
                end_ns=45_000_000,
                duration_ms=20.0,
            ),
            PhaseEvent(
                event_id="evt_4",
                span_id="span_4",
                run_id="run",
                request_id="request-1",
                method="specedge_pipeline",
                plan_id="run:round1",
                phase="pipeline.reconcile",
                span_kind="leaf",
                round=1,
                start_ns=45_000_000,
                end_ns=46_000_000,
                duration_ms=1.0,
                metadata={"reused_token_count": 2, "discarded_token_count": 3},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            written = write_timing_charts(events, output_dir, formats=("png", "svg"))

            self.assertEqual(
                tuple(name for name in written if name != "audit"),
                timing_plots.SINGLE_RESULT_CHART_NAMES,
            )
            self.assertGreater((output_dir / "timeline_gantt.png").stat().st_size, 0)
            self.assertGreater((output_dir / "timeline_gantt.svg").stat().st_size, 0)
            self.assertGreater((output_dir / "compact_timeline_distribution.png").stat().st_size, 0)
            self.assertGreater((output_dir / "compact_timeline_distribution.svg").stat().st_size, 0)
            self.assertGreater((output_dir / "worker_batch_lanes.png").stat().st_size, 0)
            self.assertGreater((output_dir / "worker_batch_lanes.svg").stat().st_size, 0)
            self.assertGreater((output_dir / "network_breakdown.png").stat().st_size, 0)
            self.assertGreater((output_dir / "network_breakdown.svg").stat().st_size, 0)
            self.assertGreater((output_dir / "proactive_reuse_chart.png").stat().st_size, 0)
            self.assertGreater((output_dir / "proactive_reuse_chart.svg").stat().st_size, 0)
            self.assertGreater((output_dir / "timing_audit.json").stat().st_size, 0)

    def test_write_timing_charts_rejects_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_timing_charts([], Path(tmp), mode="matrix")


if __name__ == "__main__":
    unittest.main()
