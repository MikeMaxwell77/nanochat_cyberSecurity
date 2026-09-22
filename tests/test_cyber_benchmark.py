import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.cyber_benchmark import PROTOCOL, compare_runs, digest, evaluate, read_suite, report, score_answer, summarize


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "crypto:1", "task": "crypto", "prompt": "Which? A: one B: two",
                      "answer": "B", "letters": ["A", "B"]}]

    def run_data(self, answer):
        records = [{"id": row["id"], "task": row["task"], **score_answer(answer, row)} for row in self.rows]
        return {"suite_sha256": digest(self.rows), "protocol": PROTOCOL,
                "records": records, "model": {"source": "test"}}

    def test_strict_scoring(self):
        row = self.rows[0]
        self.assertTrue(score_answer(" B\n", row)["correct"])
        for answer in ("B because", "A or B", "", "b"):
            self.assertFalse(score_answer(answer, row)["valid"])
        self.assertTrue(score_answer("A", row)["valid"])
        self.assertFalse(score_answer("A", row)["correct"])

    def test_intervals_and_invalid_denominator(self):
        records = self.run_data("B")["records"] + self.run_data("explanation")["records"]
        result = summarize(records)["crypto"]
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid"], 1)
        self.assertLess(result["accuracy_wilson95"][0], 0.5)
        self.assertGreater(result["accuracy_wilson95"][1], 0.5)

    def test_comparison_rejects_protocol_suite_and_missing_rows(self):
        base = self.run_data("A")
        sft = self.run_data("B")
        scores = compare_runs([base, sft])
        self.assertEqual(scores[1]["crypto"]["accuracy"] - scores[0]["crypto"]["accuracy"], 1)
        for key, value in [("suite_sha256", "other"), ("protocol", {}), ("records", [])]:
            bad = copy.deepcopy(sft)
            bad[key] = value
            with self.assertRaises(ValueError):
                compare_runs([base, bad])

    def test_duplicate_suite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            path.write_text((json.dumps(self.rows[0]) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_suite(path)

    def test_api_request_excludes_answers_and_keeps_identity(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.jsonl"
            suite.write_text(json.dumps(self.rows[0]) + "\n", encoding="utf-8")
            output = Path(directory) / "gpt4.json"
            response = {"model": "gpt-4-test-snapshot", "choices": [{"message": {"content": "B"}}]}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen") as request:
                request.return_value = io.StringIO(json.dumps(response))
                evaluate(SimpleNamespace(command="api", suite=suite, output=output, model="gpt-4"))
                body = json.loads(request.call_args.args[0].data)
                self.assertEqual(body["messages"], [{"role": "user", "content": self.rows[0]["prompt"] + PROTOCOL["instruction"]}])
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["records"][0]["details"]["model"], "gpt-4-test-snapshot")
            self.assertEqual(result["results"]["crypto"]["accuracy"], 1)
            with self.assertRaises(FileExistsError):
                evaluate(SimpleNamespace(command="api", suite=suite, output=output, model="gpt-4"))

    def test_report_handles_negative_gain_and_pending_reference(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib is required for report rendering")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, answer in [("base", "B"), ("sft", "A")]:
                (root / f"{name}.json").write_text(json.dumps(self.run_data(answer)), encoding="utf-8")
            report(SimpleNamespace(base=root / "base.json", sft=root / "sft.json", gpt4=None, output_dir=root))
            text = (root / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("-100.00 | pending", text)
            self.assertGreater((root / "waterfall.png").stat().st_size, 1000)
            report(SimpleNamespace(base=root / "base.json", sft=root / "sft.json",
                                   gpt4=root / "base.json", output_dir=root))
            self.assertNotIn("pending", (root / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
