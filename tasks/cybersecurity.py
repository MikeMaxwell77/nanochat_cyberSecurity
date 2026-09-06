"""Evaluation-only cybersecurity tasks; never add these test splits to SFT."""
from tasks.common import Task, render_mc

class CybersecurityMCQ(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, benchmark, **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        self.benchmark = benchmark
        if benchmark == "CTI-MCQ":
            self.ds = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
        elif benchmark == "MMLU-ComputerSecurity":
            self.ds = load_dataset("cais/mmlu", "computer_security", split="test")
        else:
            raise ValueError(f"Unknown cybersecurity benchmark: {benchmark}")
        self.ds = self.ds.shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        if self.benchmark == "CTI-MCQ":
            question = row["Question"]
            choices = [row[f"Option {letter}"] for letter in self.letters]
            answer = str(row["GT"]).strip().upper()
        else:
            question, choices = row["question"], row["choices"]
            answer = self.letters[int(row["answer"])]
        if len(choices) != 4 or answer not in self.letters:
            raise ValueError(f"Invalid MCQ at index {index}")
        return {"messages": [
            {"role": "user", "content": render_mc(question, self.letters, choices)},
            {"role": "assistant", "content": answer},
        ], "letters": self.letters}

    def evaluate(self, conversation, assistant_response):
        return assistant_response == conversation["messages"][-1]["content"]
