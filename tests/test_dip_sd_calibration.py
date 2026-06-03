"""DiP-SD latency calibration artifact tests."""

import json
import tempfile
import unittest
from pathlib import Path

from specplatform.core import PhaseEvent
from specplatform.methods.dip_sd.calibration import calibration_from_events, recommended_method_config_from_profile
from specplatform.methods.dip_sd.model import DiPSDModelConfig


class DiPSDCalibrationTest(unittest.TestCase):
    def test_calibration_extracts_draft_and_verify_observations(self) -> None:
        config = DiPSDModelConfig()
        events = [
            PhaseEvent(
                run_id="run",
                request_id="r1",
                method="dip_sd",
                phase="draft.generate",
                duration_ms=20.0,
                measured_duration_ms=20.0,
                worker_id="w0",
                round=0,
                metadata={
                    "runner_id": "w0",
                    "request_id": "r1",
                    "prefix_ids": [1, 2, 3],
                    "draft_token_forward_events": [
                        {"prefix_len": 3, "duration_ms": 7.0},
                        {"prefix_len": 4, "duration_ms": 8.0},
                    ],
                    "backend": "qwen3_graph",
                    "device": "cuda:0",
                },
            ),
            PhaseEvent(
                run_id="run",
                request_id="batch0",
                method="dip_sd",
                phase="verify.http_total",
                duration_ms=12.0,
                measured_duration_ms=12.0,
                batch_id="batch0",
                round=0,
                metadata={
                    "backend_name": "linear_http",
                    "response_timing": {
                        "target_forward_events": [
                            {
                                "kind": "linear_verify",
                                "shared_batch_event_id": "verify0",
                                "duration_ms": 11.0,
                                "batch_size": 2,
                                "draft_token_count": 3,
                                "prefix_len": 5,
                            }
                        ]
                    },
                },
            ),
            PhaseEvent(
                run_id="run",
                request_id="batch0",
                method="dip_sd",
                phase="verify.batch_total",
                duration_ms=13.0,
                measured_duration_ms=13.0,
                batch_id="batch0",
                round=0,
                metadata={
                    "request_ids": ["r1", "r2"],
                    "max_draft_len": 3,
                    "max_prefix_len": 5,
                },
            ),
        ]

        calibration = calibration_from_events(events, model_config=config)

        observations = calibration["observations"]
        self.assertEqual(len(observations), 4)
        self.assertEqual({row["kind"] for row in observations}, {"draft", "verify"})
        fits = calibration["fits"]
        self.assertTrue(any(row["kind"] == "draft" and row["group"] == "w0" for row in fits))
        self.assertTrue(any(row["kind"] == "draft" and row["group"] == "global" for row in fits))
        self.assertTrue(any(row["kind"] == "verify" and row["group"] == "verify.target_forward" for row in fits))
        self.assertIn("recommended_method_config", calibration)
        self.assertIn("dip_sd_draft_beta", calibration["recommended_method_config"])
        self.assertIn("dip_sd_verify_beta", calibration["recommended_method_config"])

    def test_recommended_method_config_loads_profile_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dip_sd_latency_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "recommended_method_config": {
                            "dip_sd_draft_c": 1.5,
                            "dip_sd_draft_beta": 2.5,
                            "dip_sd_verify_c": 3.5,
                            "dip_sd_verify_beta": 4.5,
                            "ignored": "value",
                        }
                    }
                ),
                encoding="utf-8",
            )

            values = recommended_method_config_from_profile(path)

        self.assertEqual(values["dip_sd_draft_c"], 1.5)
        self.assertEqual(values["dip_sd_draft_beta"], 2.5)
        self.assertEqual(values["dip_sd_verify_c"], 3.5)
        self.assertEqual(values["dip_sd_verify_beta"], 4.5)
        self.assertNotIn("ignored", values)

    def test_calibration_recommendations_do_not_emit_negative_latency_beta(self) -> None:
        config = DiPSDModelConfig()
        events = [
            PhaseEvent(
                run_id="run",
                request_id="r1",
                method="dip_sd",
                phase="draft.generate",
                duration_ms=2.0,
                measured_duration_ms=2.0,
                worker_id="w0",
                round=0,
                metadata={
                    "runner_id": "w0",
                    "request_id": "r1",
                    "prefix_ids": [1, 2, 3],
                    "draft_token_forward_events": [
                        {"prefix_len": 1, "duration_ms": 10.0},
                        {"prefix_len": 16, "duration_ms": 100.0},
                    ],
                },
            )
        ]

        calibration = calibration_from_events(events, model_config=config)
        recommended = calibration["recommended_method_config"]
        global_fit = next(
            row
            for row in calibration["fits"]
            if row["kind"] == "draft" and row["group"] == "global"
        )

        self.assertEqual(global_fit["fit_status"], "least_squares_negative_intercept_zeroed")
        self.assertGreaterEqual(recommended["dip_sd_draft_beta"], 0.0)


if __name__ == "__main__":
    unittest.main()
