import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from sca.core import select, paired_interval, evaluate_rows
from sca.export import export_tables
from sca.infer import semantic_dedup
from sca.nbest import (
    exact_unique_hypotheses,
    normalize_decoding_weights,
    asr_mass_cover,
)
from sca.prompts import build_sca_user_prompt, build_h1only_user_prompt

ROOT = Path(__file__).resolve().parents[1]


class DecisionTests(unittest.TestCase):
    def test_symmetric_selection_and_tie(self):
        a, b = ["music", "activate"], ["volume", "increase"]
        self.assertEqual(select(a, b, [10, -9]), 1)
        self.assertEqual(select(a, b, [9, -10]), 0)
        self.assertEqual(select(a, b, [10, -10]), 0)
        self.assertEqual(select(a, b, [10, -9], 1), 0)

    def test_agreement_and_parse_fallback(self):
        a = ["music", "activate"]
        self.assertEqual(select(a, a), 0)
        self.assertEqual(select(a, None), 0)
        self.assertEqual(select(None, a), 1)
        self.assertEqual(select(None, None), 0)

    def test_missing_and_nonfinite_margins_fail(self):
        for margins in [None, [1], [float("nan"), 1], [1, float("inf")]]:
            with self.assertRaises(ValueError):
                select(["a"], ["b"], margins)

    def test_softmax_and_unique_preserve_decoder_order(self):
        h, s, _ = exact_unique_hypotheses([" Play ", "play", "louder"], [-3, -1, -2], 5)
        self.assertEqual(h, [" Play ", "louder"])
        self.assertEqual(s, [-3.0, -2.0])
        w = normalize_decoding_weights([10000, 9999])
        self.assertAlmostEqual(sum(w), 1)
        self.assertGreater(w[0], w[1])

    def test_mass_baseline_sorts_weights(self):
        h, _, _, mass = asr_mass_cover(["a", "b", "c"], [0.1, 0.6, 0.3], 0.9)
        self.assertEqual(h, ["b", "c"])
        self.assertAlmostEqual(mass, 0.9)

    def test_decision_dedup_groups_mass(self):
        self.assertEqual(
            semantic_dedup(["a", "b", "c"], [0.3, 0.4, 0.3], ["x", "y", "x"], 0.55),
            ["a"],
        )

    def test_h1_prompt_removes_scores_and_tail(self):
        text = build_h1only_user_prompt(["play", "louder"], [0.9, 0.1], "a", "b")
        self.assertNotIn("louder", text)
        self.assertNotIn("[p=", text)
        self.assertIn("H1: play", text)

    def test_order_swap_changes_candidates_only(self):
        a = build_sca_user_prompt(["play"], [1.0], "first", "second")
        b = build_sca_user_prompt(["play"], [1.0], "second", "first")
        self.assertEqual(a.split("Candidate A:")[0], b.split("Candidate A:")[0])

    def test_paired_bootstrap_reference(self):
        a = np.array([1, 0, 1, 1, 0])
        b = np.array([0, 1, 0, 1, 0])
        rng = np.random.default_rng(42)
        values = []
        for _ in range(131):
            i = rng.integers(0, len(a), size=len(a))
            values.append((a[i].mean() - b[i].mean()) * 100)
        result = paired_interval(a, b, 42, 131)
        np.testing.assert_allclose(result[1:3], np.quantile(values, [0.025, 0.975]))
        self.assertAlmostEqual(paired_interval(a, a, 42, 131)[3], 1.0)


class ArchiveTests(unittest.TestCase):
    def test_sampling_parse_policy_difference(self):
        for seed, expected_changes in [(0, 0), (1, 1), (2, 1)]:
            path = ROOT / f"data/predictions/slurp_sampling_seed{seed}.jsonl.gz"
            rows = [
                json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()
            ]
            methods = list(rows[0]["predictions"])
            historical, _, _ = evaluate_rows(
                rows, methods, ["sca", "sa_sca"], "retain_first"
            )
            corrected, _, _ = evaluate_rows(rows, methods, ["sca", "sa_sca"])
            for method in ["sca", "sa_sca"]:
                self.assertEqual(
                    int((historical[method] != corrected[method]).sum()),
                    expected_changes,
                )

    def test_all_archives_and_paper_values(self):
        expected = json.loads((ROOT / "data/expected.json").read_text())
        for cfg in json.loads((ROOT / "data/experiments.json").read_text()):
            with self.subTest(experiment=cfg["name"]):
                data = (ROOT / cfg["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), cfg["sha256"])
                rows = [json.loads(x) for x in gzip.decompress(data).splitlines()]
                arrays, _, _ = evaluate_rows(
                    rows,
                    cfg["methods"],
                    cfg["judges"],
                    cfg.get("parse_policy", "valid_candidate"),
                )
                self.assertEqual(len(rows), cfg["n"])
                for method, value in expected.get(cfg["name"], {}).items():
                    self.assertAlmostEqual(
                        float(arrays[method].mean() * 100), value, delta=0.0051
                    )

    def test_duplicate_id_rejected(self):
        r = {"sample_id": "one", "gold": ["a"], "predictions": {}}
        with self.assertRaises(ValueError):
            evaluate_rows([r, r], [], [])

    def test_excel_numeric_cells_and_escaping(self):
        with tempfile.TemporaryDirectory() as d:
            export_tables(
                d,
                {
                    "Results": [
                        ["Method", "Accuracy", "Missing"],
                        ["A&B", 76.04, None],
                        ["=literal", 1.0, None],
                    ]
                },
            )
            with ZipFile(Path(d) / "reproduction_results.xlsx") as z:
                for name in z.namelist():
                    ElementTree.fromstring(z.read(name))
                xml = z.read("xl/worksheets/sheet1.xml").decode()
                self.assertIn("<v>76.04</v>", xml)
                self.assertNotIn("<f>", xml)
                self.assertIn("A&amp;B", xml)


if __name__ == "__main__":
    unittest.main()
