from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ArchiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = json.loads((ROOT / "results/full_mmad_analysis.json").read_text())
        cls.clean = json.loads((ROOT / "results/clean_panel_analysis.json").read_text())

    def test_full_exact_cover(self) -> None:
        self.assertEqual(
            self.full["coverage"],
            {
                "disjoint_exact_cover": True,
                "full_manifest_ids": 39670,
                "joined_result_ids": 39670,
                "shards": 8,
            },
        )

    def test_full_overall_recomputes(self) -> None:
        metrics = self.full["metrics"]
        expected = sum(metrics["per_task_macro"].values()) / 7
        self.assertAlmostEqual(expected, metrics["mmad_overall_macro"], places=15)
        self.assertAlmostEqual(0.8241684704909915, expected, places=15)

    def test_clean_delta_and_uncertainty(self) -> None:
        self.assertAlmostEqual(
            self.clean["paired_delta_treatment_minus_control"]["mmad_overall_macro"],
            0.0026977664813337965,
            places=15,
        )
        interval = self.clean["cluster_bootstrap"]["metrics"]["mmad_overall_macro"]["delta_ci95"]
        self.assertLess(interval[0], 0)
        self.assertGreater(interval[1], 0)

    def test_checkpoint_identity(self) -> None:
        path = ROOT / "checkpoints/echo-caver-v1/adapter_model.safetensors"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            actual,
            "21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9",
        )

    def test_title_contract(self) -> None:
        metadata = json.loads((ROOT / "metadata/archive.json").read_text())
        self.assertTrue(metadata["title"].startswith("[Research]"))


if __name__ == "__main__":
    unittest.main()

