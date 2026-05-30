"""core schema 和 target placement 的最小测试。"""

import unittest

from specplatform.core import CandidateNode, CandidateTree, PhaseEvent, RuntimeContext, TargetPlacementConfig


class MetricsSchemaTest(unittest.TestCase):
    """验证 PhaseEvent、CandidateTree 和 target placement 的基础契约。"""

    def test_phase_event_serializes_required_fields(self) -> None:
        """PhaseEvent 应补齐 duration/timestamp 等序列化字段。"""
        event = PhaseEvent(
            run_id="r",
            request_id="q",
            method="fake_linear",
            phase="verify_total",
            duration_ms=1.2,
        )
        payload = event.to_dict()

        self.assertEqual(payload["phase"], "verify_total")
        self.assertEqual(payload["duration_ms"], 1.2)
        self.assertEqual(payload["attributed_duration_ms"], 1.2)
        self.assertEqual(payload["measured_duration_ms"], 1.2)
        self.assertEqual(payload["event_scope"], "system")
        self.assertEqual(payload["span_kind"], "leaf")
        self.assertIn("timestamp_ns", payload)

    def test_candidate_tree_validation(self) -> None:
        """CandidateTree 应接受合法的父子拓扑。"""
        tree = CandidateTree(
            root_prefix_len=2,
            nodes=[
                CandidateNode(
                    node_id=1,
                    parent_id=None,
                    token_id=10,
                    depth=1,
                    draft_logprob=None,
                    draft_worker_id="draft0",
                )
            ],
        )

        tree.validate()

    def test_target_placement_defaults_to_a100(self) -> None:
        """target/verifier 默认放在 A100。"""
        context = RuntimeContext()

        self.assertEqual(context.target_placement.placement, "a100")
        self.assertEqual(context.target_placement.to_backend_info(), {"target_placement": "a100"})

    def test_target_placement_can_be_3090(self) -> None:
        """target/verifier 也允许配置到 3090。"""
        context = RuntimeContext(
            backend_info={
                "target_placement": "3090",
                "target_backend": "fake_proposal",
                "target_host": "server-rtx3090-8c",
                "target_device": "cuda:0",
            }
        )

        placement = context.target_placement
        self.assertEqual(placement.placement, "3090")
        self.assertEqual(placement.backend, "fake_proposal")
        self.assertEqual(placement.host, "server-rtx3090-8c")
        self.assertEqual(placement.device, "cuda:0")

    def test_target_placement_rejects_unknown_location(self) -> None:
        """未知 target placement 应被拒绝。"""
        with self.assertRaises(ValueError):
            TargetPlacementConfig.from_backend_info({"target_placement": "tpu"})


if __name__ == "__main__":
    unittest.main()
