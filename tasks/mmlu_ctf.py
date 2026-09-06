from datasets import load_dataset
from tasks.common import Task

class CyberSecMMLU(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"]
        assert split in ["auxiliary_train", "validation", "dev", "test"]
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'open_ended'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        subject = row["subject"]

        correct_answer = choices[answer_idx]

        # Convert MC ? open-ended prompt
        user_message = f"""
You are a cybersecurity assistant.

Question:
{question}

Provide your best answer in plain text.
Then give a confidence score from 0 to 1.

Format:
ANSWER: ...
CONFIDENCE: ...
"""

        conversation = {
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "answer": correct_answer,
            "subject": subject
        }

        return conversation

    def evaluate(self, conversation, assistant_response):
        # Extract answer + confidence
        answer = self.extract_answer(assistant_response)
        confidence = self.extract_confidence(assistant_response)

        correct = self.is_correct(answer, conversation["answer"])

        return {
            "correct": correct,
            "confidence": confidence
        }

    # --- helpers ---
    def extract_answer(self, text):
        if "ANSWER:" in text:
            return text.split("ANSWER:")[1].split("CONFIDENCE:")[0].strip()
        return text.strip()

    def extract_confidence(self, text):
        if "CONFIDENCE:" in text:
            try:
                return float(text.split("CONFIDENCE:")[1].strip())
            except:
                return 0.0
        return 0.0

    def is_correct(self, pred, gold):
        return pred.lower().strip() == gold.lower().strip()