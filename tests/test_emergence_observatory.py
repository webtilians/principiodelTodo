"""Independent tests: synthetic signals are not evidence of consciousness."""
import json
import os
import random
import stat
import tempfile
import unittest
from pathlib import Path

from observatory.adapters import parse_legacy_log, read_scalar_csv, export_jsonl
from observatory.blinding import blind_runs
from observatory.core import analyze, load_jsonl, parse_points
from observatory.__main__ import generate_demo


def fixture(planted=True, seed=2, run_id="run_1", n=60):
    rng = random.Random(seed)
    return [
        {"run_id": run_id, "step": index, "metrics": {
            "signal": rng.gauss(0, 0.2) + (4.0 if planted and index >= 30 else 0.0),
        }}
        for index in range(n)
    ]


class TestObservatory(unittest.TestCase):
    def test_validation_rejects_duplicate_steps(self):
        rows = fixture(n=4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_points(rows + [rows[0]])

    def test_validation_rejects_nonfinite_and_bools(self):
        for invalid in [float("nan"), float("inf"), True, "1.2"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_points([{"run_id": "a", "step": 0, "metrics": {"m": invalid}}])

    def test_constant_signal_is_not_discovery(self):
        run = parse_points([
            {"run_id": "a", "step": i, "metrics": {"m": 1.0}} for i in range(30)
        ])
        report = analyze(run, permutations=19)
        self.assertFalse(report["candidates"][0]["flagged"])

    def test_planted_changes_repeat_across_independent_seeds(self):
        rows = []
        for index in range(3):
            rows += fixture(seed=10 + index, run_id="repeat_%s" % index)
        report = analyze(parse_points(rows), window=6, permutations=199, seed=23)
        self.assertEqual(sum(item["flagged"] for item in report["candidates"]), 3)
        self.assertTrue(all(abs(item["step"] - 30) <= 1 for item in report["candidates"]))
        self.assertTrue(any(item["n_runs"] == 3 for item in report["recurrences"]))
        self.assertTrue(all(item["label"] == "unclassified_transition" for item in report["candidates"]))

    def test_seeded_reanalysis_is_reproducible(self):
        runs = parse_points(fixture())
        first = analyze(runs, permutations=49, seed=17)
        second = analyze(runs, permutations=49, seed=17)
        self.assertEqual(first, second)

    def test_short_run_is_excluded(self):
        report = analyze(parse_points(fixture(n=10)), window=6, permutations=19)
        self.assertEqual(report["excluded"][0]["reason"], "too_short")
        self.assertFalse(report["candidates"])

    def test_blinding_removes_conditions_and_prevents_overwrites(self):
        runs = parse_points([{**row, "condition": "planted", "seed": 123} for row in fixture(n=20)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "blind.jsonl"
            mapping = root / "PRIVATE_mapping.json"
            info = blind_runs(runs, output_path=out, key_file=mapping)
            self.assertEqual(info["cases"], 1)
            public = out.read_text()
            self.assertNotIn("planted", public)
            self.assertNotIn("run_1", public)
            self.assertNotIn('"seed"', public)
            self.assertEqual(len(load_jsonl(out)), 1)
            with self.assertRaises(FileExistsError):
                blind_runs(runs, output_path=out, key_file=mapping)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(mapping.stat().st_mode), 0o600)

    def test_historical_log_extraction(self):
        source = ("ignored header\n"
                  "Recursion 2: Clusters 4, Entropy 5.300, Loss 0.376\n"
                  "R   10: C 7 | E 1.23 | L 0.042 | more\n")
        rows = parse_legacy_log(source, run_id="legacy")
        self.assertEqual([row["step"] for row in rows], [2, 10])
        self.assertEqual(rows[1]["metrics"]["clusters_reported"], 7.0)
        self.assertEqual(rows[0]["metrics"]["entropy_reported"], 5.3)

    def test_csv_import_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.csv"
            p.write_text("recursion,entropy,loss,comment\n0,0.1,1.0,a\n1,0.2,0.8,b\n")
            rows = read_scalar_csv(p, run_id="a")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["metrics"], {"entropy": 0.2, "loss": 0.8})
            dest = Path(tmp) / "data.jsonl"
            export_jsonl(dest, rows)
            self.assertEqual(len(load_jsonl(dest)["a"]), 2)
            with self.assertRaises(FileExistsError):
                export_jsonl(dest, rows)

    def test_blinded_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = generate_demo(Path(tmp), seed=42)
            self.assertEqual(info["blind"]["cases"], 6)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["run_count"], 6)
            self.assertGreaterEqual(info["flagged"], 3)
            self.assertNotIn("planted_shift", (Path(tmp) / "blinded.jsonl").read_text())
            self.assertFalse(report.get("consciousness_proven", False))


if __name__ == "__main__":
    unittest.main()
