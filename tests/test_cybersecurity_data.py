import json
import tempfile
import unittest
from pathlib import Path
from scripts.clean_cyber_data import clean, parse_record
from tasks.cybersecurity import CybersecurityMCQ

class CybersecurityTests(unittest.TestCase):
    def test_preservation_dedup_and_rejection(self):
        messages = [{"role": "user", "content": "localhost 127.0.0.1\n    code"}, {"role": "assistant", "content": "Keep these."}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text(json.dumps({"messages": messages}) + "\n" + json.dumps(messages) + '\n{"messages":[\n', encoding="utf-8")
            output = Path(directory) / "output"
            counts = clean(source, output)
            self.assertEqual(counts, dict(kept=1, repaired=0, duplicates=1, rejected=1))
            self.assertEqual(json.loads((output / "training_candidates.jsonl").read_text(encoding="utf-8")), messages)

    def test_quote_repair(self):
        raw = '{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Call it "Office Preview"."}]}'
        messages, repaired = parse_record(raw)
        self.assertTrue(repaired)
        self.assertEqual(messages[-1]["content"], 'Call it "Office Preview".')

    def test_scores_are_booleans(self):
        task = CybersecurityMCQ.__new__(CybersecurityMCQ)
        task.benchmark = "CTI-MCQ"
        task.ds = [{"Question":"Q", "Option A":"a", "Option B":"b", "Option C":"c", "Option D":"d", "GT":"b"}]
        conversation = task.get_example(0)
        self.assertIs(task.evaluate(conversation, "B"), True)
        self.assertIs(task.evaluate(conversation, "A"), False)

if __name__ == "__main__":
    unittest.main()
