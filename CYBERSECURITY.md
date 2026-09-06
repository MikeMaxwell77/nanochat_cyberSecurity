# Cybersecurity evaluation and data preparation

Run from the repository root in the existing environment:

```bash
python -m scripts.clean_cyber_data data/nanochatdataCMAPRILFOUR.jsonl --output-dir data/cleaned_v1
python -m unittest discover -s tests -p test_cybersecurity_data.py -v
python -m scripts.chat_eval -i sft -a 'CTI-MCQ|MMLU-ComputerSecurity'
```

Review `audit.jsonl` before using `training_candidates.jsonl`. The latter contains message arrays compatible with CustomJSON. Originals are never overwritten. Only exact conversations are deduplicated; near-duplicates and repeated templates need review. The cleaner preserves localhost, IP addresses, code, indentation and Markdown. Quote repair is heuristic and specific to role/content records. Unsupported roles and malformed records are audited as rejected. New output directories are required.

Copy the reviewed candidate file into the cluster's `ctf_training_set.jsonl` location and remove or replace the speedrun curl command that otherwise overwrites it with APRILTHREE. The cleaner does not modify the training mixture or select a validation split.

Run `bash runs/speedrun.sh` inside your scheduled cluster allocation. SFT runs first, followed automatically by one evaluation process covering all general and cybersecurity tasks, followed by report generation. No separate submission or CYBER_EVAL flag is needed. Training and evaluation failures stop the launcher. Neither cybersecurity benchmark is added to training or to ChatCORE. An explicit `-x 50` can be used for a smoke test; label this as a subset result. Use the same checkpoint-selection policy, prompts and limits for before/after comparisons.

CTI-MCQ uses AI4Sec/cti-bench, cti-mcq test split, with nanochat's categorical prompt and choice-logit scoring. This is an adapted protocol, not a directly comparable reproduction of the original paper's generative scores. Source: https://huggingface.co/datasets/AI4Sec/cti-bench (CC-BY-NC-SA-4.0). MMLU-ComputerSecurity uses cais/mmlu, computer_security test split. Both shuffle with seed 42. Record resolved dataset revisions and model checkpoint IDs alongside results for reproducibility. The Hugging Face datasets dependency and downloads are required on the cluster.

MMLU test material is already used for validation in the existing SFT script. Consequently, the computer-security subset is a regression metric, not an untouched final holdout. Keep CTIBench out of training and avoid repeatedly selecting checkpoints against its final scores.

The old mmlu_ctf.py remains untouched; it returns an unsupported evaluation type and a dictionary instead of the boolean expected by the evaluator. Use the new named tasks instead.
