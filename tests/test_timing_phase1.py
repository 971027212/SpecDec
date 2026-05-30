"""timing / metrics Phase 1 行为测试。"""

import csv
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from specplatform.core import CandidateProposal
from specplatform.metrics import write_phase_events_csv, write_phase_summary_csv
from specplatform.timing import TimingAttributor, TimingRecorder, TimingSpan, event_from_span
from specplatform.timing.summary import summarize_timing_events


class FakeClock:
    """用固定 ns 序列替代真实时钟，保证 timing 测试稳定。"""

    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    def __call__(self) -> int:
        """返回下一个预置时间戳。"""
        if not self.values:
            raise AssertionError("fake clock exhausted")
        return self.values.pop(0)


class TimingPhase1Test(unittest.TestCase):
    """验证 span、归因和 artifact 输出不会重复计算共享 batch。"""

    def test_timing_span_measures_finished_duration(self) -> None:
        """已结束 span 应能计算毫秒耗时。"""
        span = TimingSpan(
            span_id="span_001",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=1_000,
            end_ns=101_000,
        )

        self.assertEqual(span.measured_duration_ms, 0.1)

    def test_timing_span_rejects_unfinished_or_negative_duration(self) -> None:
        """未结束或时间倒退的 span 应被拒绝。"""
        unfinished = TimingSpan(
            span_id="span_001",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=1_000,
        )
        with self.assertRaises(ValueError):
            _ = unfinished.measured_duration_ms

        backwards = TimingSpan(
            span_id="span_002",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=2_000,
            end_ns=1_000,
        )
        with self.assertRaises(ValueError):
            _ = backwards.measured_duration_ms

    def test_recorder_span_generates_ids_and_bounds(self) -> None:
        """TimingRecorder.span 应生成 id 并自动记录 end_ns。"""
        recorder = TimingRecorder(clock=FakeClock([10, 110]))

        with recorder.span(phase="scheduler.plan", method="fake_linear", plan_id="plan0") as span:
            self.assertEqual(span.span_id, "span_000001")
            self.assertEqual(span.start_ns, 10)

        self.assertEqual(span.end_ns, 110)
        self.assertEqual(span.measured_duration_ms, 0.0001)

    def test_timing_span_does_not_carry_attribution_only_fields(self) -> None:
        """TimingSpan 本身不携带 attribution-only 字段。"""
        names = {field.name for field in fields(TimingSpan)}

        self.assertNotIn("event_scope", names)
        self.assertNotIn("span_kind", names)
        self.assertNotIn("attribution", names)
        self.assertNotIn("parent_span_id", names)

    def test_event_from_span_creates_system_event(self) -> None:
        """真实 span 转出的事件默认是 system event。"""
        recorder = TimingRecorder(clock=FakeClock([]))
        span = TimingSpan(
            span_id="span_001",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=0,
            end_ns=100_000_000,
            run_id="run",
            round_id=0,
            batch_id="batch0",
            shared=True,
        )

        event = event_from_span(
            span,
            event_id_factory=recorder.next_event_id,
            span_kind="leaf",
            attribution="batch",
        )

        self.assertEqual(event.event_id, "evt_000001")
        self.assertEqual(event.span_id, span.span_id)
        self.assertIsNone(event.parent_span_id)
        self.assertEqual(event.event_scope, "system")
        self.assertEqual(event.span_kind, "leaf")
        self.assertEqual(event.measured_duration_ms, 100.0)
        self.assertEqual(event.attributed_duration_ms, 100.0)

    def test_batch_attribution_links_to_parent_span(self) -> None:
        """batch verifier 耗时应平均归因到每个 request。"""
        recorder = TimingRecorder(clock=FakeClock([]))
        span = TimingSpan(
            span_id="span_batch",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=0,
            end_ns=100_000_000,
            run_id="run",
            round_id=0,
            batch_id="batch0",
            shared=True,
        )
        proposals = [
            CandidateProposal(
                proposal_id=f"proposal{index}",
                request_id=f"req{index}",
                worker_id="draft0",
                shape="linear",
                tokens=[index],
            )
            for index in range(4)
        ]

        events = TimingAttributor().attribute_batch_average(
            parent_span=span,
            proposals=proposals,
            event_id_factory=recorder.next_event_id,
        )

        self.assertEqual(len(events), 4)
        for event in events:
            self.assertEqual(event.span_id, span.span_id)
            self.assertEqual(event.parent_span_id, span.span_id)
            self.assertEqual(event.event_scope, "request")
            self.assertEqual(event.span_kind, "attribution")
            self.assertEqual(event.phase, "verify.request_attributed")
            self.assertEqual(event.phase_category, "verify")
            self.assertEqual(event.measured_duration_ms, 100.0)
            self.assertEqual(event.attributed_duration_ms, 25.0)
        self.assertEqual(sum(event.attributed_duration_ms for event in events), 100.0)

    def test_summary_views_do_not_double_count_shared_batch(self) -> None:
        """summary 视图区分真实耗时和 request 归因，避免双算。"""
        recorder = TimingRecorder(clock=FakeClock([]))
        span = TimingSpan(
            span_id="span_batch",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=0,
            end_ns=100_000_000,
            run_id="run",
            round_id=0,
            batch_id="batch0",
            shared=True,
        )
        system_event = event_from_span(
            span,
            event_id_factory=recorder.next_event_id,
            span_kind="leaf",
            attribution="batch",
        )
        proposals = [
            CandidateProposal(
                proposal_id=f"proposal{index}",
                request_id=f"req{index}",
                worker_id="draft0",
                shape="linear",
                tokens=[index],
            )
            for index in range(4)
        ]
        attributed_events = TimingAttributor().attribute_batch_average(
            parent_span=span,
            proposals=proposals,
            event_id_factory=recorder.next_event_id,
        )

        rows = summarize_timing_events([system_event, *attributed_events])
        by_view_phase = {(row.summary_view, row.phase): row for row in rows}

        self.assertEqual(
            by_view_phase[("system_leaf_summary", "verify.batch_total")].total_measured_duration_ms,
            100.0,
        )
        self.assertEqual(
            by_view_phase[
                ("request_attributed_summary", "verify.request_attributed")
            ].total_attributed_duration_ms,
            100.0,
        )
        debug_rows = [row for row in rows if row.summary_view == "debug_summary"]
        self.assertEqual(sum(row.count for row in debug_rows), 5)
        for row in rows:
            self.assertNotEqual(row.total_attributed_duration_ms, 200.0)

    def test_artifacts_include_timing_columns(self) -> None:
        """CSV artifact 应包含 timing/attribution 所需列。"""
        recorder = TimingRecorder(clock=FakeClock([]))
        span = TimingSpan(
            span_id="span_batch",
            phase="verify.batch_total",
            method="fake_linear",
            plan_id="plan0",
            start_ns=0,
            end_ns=100_000_000,
            run_id="run",
            round_id=0,
            batch_id="batch0",
            shared=True,
        )
        event = event_from_span(
            span,
            event_id_factory=recorder.next_event_id,
            span_kind="leaf",
            attribution="batch",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_phase_events_csv([event], root / "phase_events.csv")
            write_phase_summary_csv([event], root / "phase_summary.csv")

            with (root / "phase_events.csv").open(encoding="utf-8") as handle:
                event_rows = list(csv.DictReader(handle))
            self.assertIn("event_id", event_rows[0])
            self.assertIn("span_id", event_rows[0])
            self.assertIn("measured_duration_ms", event_rows[0])
            self.assertIn("attributed_duration_ms", event_rows[0])

            with (root / "phase_summary.csv").open(encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertIn("summary_view", summary_rows[0])
            self.assertIn("total_measured_duration_ms", summary_rows[0])
            self.assertIn("mean_attributed_duration_ms", summary_rows[0])


if __name__ == "__main__":
    unittest.main()
