"""Method tuning runner helper tests."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _load_tuning_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_method_tuning.py"
    spec = importlib.util.spec_from_file_location("method_tuning_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_method_tuning.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


class MethodTuningRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_tuning_module()

    def test_candidate_specs_generate_method_specific_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.runner.build_parser().parse_args(
                [
                    "--methods",
                    "specedge_pipeline,sled_async,dip_sd",
                    "--max-candidates-per-method",
                    "1",
                    "--request-count",
                    "4",
                    "--draft-worker-count",
                    "4",
                    "--max-new-tokens",
                    "16",
                ]
            )

            candidates = self.runner._candidate_specs(
                args,
                _base_config(),
                config_root=root / "configs",
                run_root=root / "runs",
            )

        self.assertEqual([item["method"] for item in candidates], ["specedge_pipeline", "sled_async", "dip_sd"])

        specedge = candidates[0]["config"]
        self.assertEqual(specedge["run"]["methods"], ["target_only", "specedge_pipeline"])
        self.assertEqual(specedge["data"]["sample_count"], 4)
        self.assertEqual(specedge["generation"]["max_new_tokens"], 16)
        self.assertEqual(specedge["tree"]["max_depth"], 4)
        self.assertEqual(specedge["tree"]["branch_width"], 4)
        self.assertEqual(specedge["pipeline"]["proactive_depth"], 4)

        sled = candidates[1]["config"]
        self.assertEqual(sled["run"]["methods"], ["target_only", "sled_async"])
        self.assertTrue(sled["sled"]["strict"])
        self.assertEqual(sled["sled"]["max_speculation_tokens"], 4)
        self.assertEqual(sled["sled"]["batch_size"], 4)
        self.assertEqual(sled["sled"]["confidence_threshold"], 0.4)
        self.assertTrue(sled["sled"]["static_queue"]["enabled"])

        dip_sd = candidates[2]["config"]
        self.assertEqual(dip_sd["run"]["methods"], ["target_only", "dip_sd"])
        self.assertEqual(dip_sd["dip_sd"]["solver"], "paper_milp_or_dinkelbach")
        self.assertEqual(dip_sd["dip_sd"]["max_draft_length"], 4)
        self.assertTrue(dip_sd["dip_sd"]["plan_cache_enabled"])
        self.assertFalse(dip_sd["dip_sd"]["calibration_enabled"])

    def test_best_rows_require_correctness_and_lowest_effective_time(self) -> None:
        rows = [
            _row("specedge_pipeline", "bad-fast", 10.0, matches=False),
            _row("specedge_pipeline", "good-slow", 20.0, matches=True),
            _row("specedge_pipeline", "good-fast", 12.0, matches=True),
            _row("sled_async", "sled", 30.0, matches=True),
            _row("dip_sd", "dip", 40.0, matches=True),
        ]

        best = self.runner._best_rows(rows)

        self.assertEqual(best["specedge_pipeline"]["candidate_id"], "good-fast")
        self.assertEqual(best["sled_async"]["candidate_id"], "sled")
        self.assertEqual(best["dip_sd"]["candidate_id"], "dip")

    def test_locked_config_merges_best_method_params(self) -> None:
        best = {
            "specedge_pipeline": _row(
                "specedge_pipeline",
                "spec",
                12.0,
                params={
                    "depth": 8,
                    "branch_width": 4,
                    "max_budget": 20,
                    "proactive_depth": 8,
                    "official": True,
                },
            ),
            "sled_async": _row(
                "sled_async",
                "sled",
                30.0,
                params={
                    "max_speculation_tokens": 8,
                    "confidence_threshold": 0.6,
                    "batch_size": 4,
                    "proactive_tokens": 8,
                    "queue_max_wait_ms": None,
                },
            ),
            "dip_sd": _row(
                "dip_sd",
                "dip",
                40.0,
                params={
                    "max_draft_length": 8,
                    "solver": "paper_milp_or_dinkelbach",
                    "batch_count": [2, None],
                    "calibration_profile": None,
                },
            ),
        }

        locked = self.runner._locked_config(
            _base_config(),
            best,
            prompt_split={
                "tuning_prompts_file": Path("data/sample_prompts_mixed.jsonl"),
                "heldout_prompts_file": Path("data/sample_prompts_heldout.jsonl"),
                "heldout_request_count": 8,
            },
        )

        self.assertIsNotNone(locked)
        assert locked is not None
        self.assertEqual(
            locked["run"]["methods"],
            ["target_only", "specedge_pipeline", "sled_async", "dip_sd"],
        )
        self.assertEqual(locked["tree"]["max_depth"], 8)
        self.assertEqual(locked["pipeline"]["proactive_depth"], 8)
        self.assertEqual(locked["sled"]["confidence_threshold"], 0.6)
        self.assertEqual(locked["sled"]["max_speculation_tokens"], 8)
        self.assertEqual(locked["dip_sd"]["max_draft_length"], 8)
        self.assertEqual(locked["dip_sd"]["max_batch_count"], 0)
        self.assertEqual(locked["data"]["sample_prompts"], "data/sample_prompts_heldout.jsonl")
        self.assertEqual(locked["data"]["split"]["tuning_sample_prompts"], "data/sample_prompts_mixed.jsonl")

    def test_locked_config_requires_heldout_prompts(self) -> None:
        best = {
            method: _row(method, method, 10.0, params=_minimal_params(method))
            for method in ("specedge_pipeline", "sled_async", "dip_sd")
        }

        locked = self.runner._locked_config(
            _base_config(),
            best,
            prompt_split={
                "tuning_prompts_file": Path("data/sample_prompts_mixed.jsonl"),
                "heldout_prompts_file": None,
                "heldout_request_count": 8,
            },
        )

        self.assertIsNone(locked)

    def test_dip_sd_calibration_profile_is_written_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.runner.build_parser().parse_args(
                [
                    "--methods",
                    "dip_sd",
                    "--dip-sd-calibration-profiles",
                    "experiments/calibration/profile.json",
                    "--max-candidates-per-method",
                    "1",
                ]
            )

            candidates = self.runner._candidate_specs(
                args,
                _base_config(),
                config_root=root / "configs",
                run_root=root / "runs",
            )

        profile = candidates[0]["config"]["dip_sd"]["calibration_profile"]
        self.assertTrue(Path(profile).is_absolute())
        self.assertTrue(profile.endswith("experiments/calibration/profile.json"))


def _base_config() -> dict:
    return {
        "run": {"id": "base", "methods": ["target_only"], "output_dir": "experiments/base"},
        "data": {
            "sample_prompts": "data/sample_prompts_mixed.jsonl",
            "sample_count": 8,
            "use_sample_prompts": True,
        },
        "generation": {"max_new_tokens": 8},
        "draft": {"worker_count": 8},
        "target": {"url": "http://127.0.0.1:8011"},
        "tree": {"max_depth": 8, "branch_width": 8, "max_budget": 20},
        "pipeline": {"min_depth": 1, "max_depth": 8, "proactive_depth": 8},
        "specedge": {"official": True},
        "sled": {
            "strict": True,
            "batch_size": 8,
            "max_speculation_tokens": 8,
            "confidence_threshold": 0.5,
            "async": {"proactive_tokens": 8},
            "static_queue": {"enabled": True, "pad_to_max_length": True},
        },
        "dip_sd": {
            "solver": "paper_milp_or_dinkelbach",
            "min_batch_count": 2,
            "max_batch_count": 0,
            "initial_draft_length": 7,
            "max_draft_length": 8,
        },
    }


def _row(
    method: str,
    candidate_id: str,
    effective_total_ms: float,
    *,
    matches: bool = True,
    params: dict | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "method": method,
        "status": "completed",
        "matches_target_only": matches,
        "effective_total_ms": effective_total_ms,
        "params_json": __import__("json").dumps(params or {}, sort_keys=True),
    }


def _minimal_params(method: str) -> dict:
    if method == "specedge_pipeline":
        return {
            "depth": 4,
            "branch_width": 4,
            "max_budget": 20,
            "proactive_depth": 4,
            "official": True,
        }
    if method == "sled_async":
        return {
            "max_speculation_tokens": 4,
            "confidence_threshold": 0.5,
            "batch_size": 4,
            "proactive_tokens": 4,
            "queue_max_wait_ms": None,
        }
    return {
        "max_draft_length": 4,
        "solver": "paper_milp_or_dinkelbach",
        "batch_count": [2, None],
        "calibration_profile": None,
    }
