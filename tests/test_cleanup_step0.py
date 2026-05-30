"""Step 0 清理测试：active skeleton 不再导出 fake/baseline 组件。"""

import unittest
from pathlib import Path


class CleanupStep0Test(unittest.TestCase):
    """确认 fake/baseline 已从活动代码中移除。"""

    def test_removed_fake_and_baseline_files_are_absent(self) -> None:
        """旧 fake/baseline 模块不应继续存在于 active src/tests。"""
        root = Path(__file__).resolve().parents[1]
        removed = [
            "src/specplatform/model/fake.py",
            "src/specplatform/methods/fake_linear.py",
            "src/specplatform/verification/fake_backend.py",
            "src/specplatform/runtime/autoregressive.py",
            "tests/test_autoregressive_baseline.py",
            "tests/test_draft_runner.py",
            "tests/test_unified_runtime_phase1.py",
        ]

        for relative_path in removed:
            self.assertFalse((root / relative_path).exists(), relative_path)

    def test_active_source_does_not_reference_fake_or_baseline_exports(self) -> None:
        """active src 不应再出现 fake/baseline 的公开组件名。"""
        root = Path(__file__).resolve().parents[1]
        forbidden = [
            "FakeDraftRunner",
            "FakeDraftGeneration",
            "FakeProposalVerifier",
            "FakeLinearCandidateStrategy",
            "FakeDeterministicModelRunner",
            "run_autoregressive_baseline",
            "AutoRegressiveBaselineResult",
            "fake_linear",
            "fake_proposal",
        ]
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (root / "src/specplatform").rglob("*.py")
        )

        for marker in forbidden:
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
