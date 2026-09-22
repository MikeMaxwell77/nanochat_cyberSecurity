# Cybersecurity evaluation and data preparation

Run from the repository root in the existing environment:

```bash
python -m scripts.clean_cyber_data data/nanochatdataCMAPRILFOUR.jsonl --output-dir data/cleaned_v1
python -m unittest discover -s tests -p test_cybersecurity_data.py -v
python -m scripts.chat_eval -i sft -a 'CTI-MCQ|MMLU-ComputerSecurity'
```

Review `audit.jsonl` before using `training_candidates.jsonl`. The latter contains message arrays compatible with CustomJSON. Originals are never overwritten. Only exact conversations are deduplicated; near-duplicates and repeated templates need review. The cleaner preserves localhost, IP addresses, code, indentation and Markdown. Quote repair is heuristic and specific to role/content records. Unsupported roles and malformed records are audited as rejected. New output directories are required.

Copy the reviewed candidate file into the cluster's `ctf_training_set.jsonl` location and remove or replace the speedrun curl command that otherwise overwrites it with APRILTHREE. The cleaner does not modify the training mixture or select a validation split.

Run `bash runs/speedrun.sh` inside your scheduled cluster allocation. The launcher resolves the base checkpoint once, evaluates it, runs SFT, and evaluates the new SFT checkpoint on the same tasks. It also freezes a shared MCQ suite and runs generated-answer scoring on both models for the waterfall. Failures stop the launcher. Neither cybersecurity benchmark is added to training or to ChatCORE. An explicit `-x 50` on `chat_eval` can be used for a smoke test; label this as a subset result. The full before/after battery adds runtime beyond training; size your allocation accordingly.

CTI-MCQ uses AI4Sec/cti-bench, cti-mcq test split, with nanochat's categorical prompt and choice-logit scoring. This is an adapted protocol, not a directly comparable reproduction of the original paper's generative scores. Source: https://huggingface.co/datasets/AI4Sec/cti-bench (CC-BY-NC-SA-4.0). MMLU-ComputerSecurity uses cais/mmlu, computer_security test split. Both shuffle with seed 42. Record resolved dataset revisions and model checkpoint IDs alongside results for reproducibility. The Hugging Face datasets dependency and downloads are required on the cluster.

MMLU test material is already used for validation in the existing SFT script. Consequently, the computer-security subset is a regression metric, not an untouched final holdout. Keep CTIBench out of training and avoid repeatedly selecting checkpoints against its final scores.

The old mmlu_ctf.py remains untouched; it returns an unsupported evaluation type and a dictionary instead of the boolean expected by the evaluator. Use the new named tasks instead.

## Checkpoints and reproducible comparison

The launcher expects an existing base checkpoint and its original tokenizer. Tokenizer preparation is now opt-in (`PREPARE_BASE_DATA=1`); use that only when preparing a matching new base model. Base training remains commented out in this cluster-specific launcher.

Optional environment variables: `BASE_TAG`, `BASE_STEP`, `SFT_TAG`, `SAVE_EVERY` (default 200), `NPROC` (default 2), and `NANOCHAT_BASE_DIR`. By default SFT gets a timestamped tag. Existing SFT output tags are rejected. For example:

```bash
BASE_TAG=d24 BASE_STEP=10000 SFT_TAG=d24-crypto-net-v2 bash runs/speedrun.sh
```

Replace the example base step with a real checkpoint step. SFT saves periodic and final files under `$NANOCHAT_BASE_DIR/chatsft_checkpoints/$SFT_TAG/`:

- `model_XXXXXX.pt`: fine-tuned weights.
- `meta_XXXXXX.json`: architecture, configuration, source base checkpoint, and final/periodic status.
- `optim_XXXXXX_rankN.pt`: optimizer state for each rank.

Keep the metadata and original tokenizer with the weights. These artifacts support loading for inference; this change does not add exact interrupted-SFT resumption (data cursor/RNG/scaler state are not restored).

Results go under `$NANOCHAT_BASE_DIR/comparisons/$SFT_TAG/`. `base_general.json` and `sft_general.json` contain the original eight-task battery. `base.json` and `sft.json` contain the matched generated-letter evaluation, per-question responses, dataset-content hash, checkpoint identities, invalid-answer counts, and Wilson 95% accuracy intervals. `comparison.md`, `waterfall.png`, and `waterfall.svg` show **base accuracy → SFT gain/loss → SFT accuracy**, with GPT-4 as an optional independent reference bar. Intervals describe individual accuracies; they are not a significance test of the paired improvement.

## Add the GPT-4 reference

The new runner uses the same frozen questions, user instruction, temperature zero, 32-token answer budget, strict single-letter scoring, and no tools for every backend. Explanations, refusals, and malformed answers count as incorrect and are also counted as invalid. Chat templates/tokenization remain model-specific. A raw pretrained base may not understand the chat template, so this measures end-to-end chat behavior as well as knowledge; keep the separate choice-logit regression results alongside it.

The launcher does not automatically call a paid API. After the local run, set `OPENAI_API_KEY` in your environment and run:

```bash
RESULT_DIR="$NANOCHAT_BASE_DIR/comparisons/$SFT_TAG"
python -m scripts.cyber_benchmark api --model gpt-4 \
  --suite "$RESULT_DIR/suite.jsonl" --output "$RESULT_DIR/gpt4.json"
uv run --with matplotlib python -m scripts.cyber_benchmark report \
  --base "$RESULT_DIR/base.json" --sft "$RESULT_DIR/sft.json" \
  --gpt4 "$RESULT_DIR/gpt4.json" --output-dir "$RESULT_DIR"
```

[Official GPT-4 documentation](https://developers.openai.com/api/docs/models/gpt-4) lists Chat Completions support and marks older snapshots deprecated. The runner defaults to the requested `gpt-4` family and records the actual returned model ID for every response; pass a specific snapshot if your account supports it. It never substitutes GPT-4o or another model on failure. API errors stop the run; completed answers remain in `.partial.jsonl`, but partial runs cannot be plotted as complete results. Use a new output filename to retry; automatic resume is not implemented. API access and billing are required.

For a small smoke test, export a separate frozen suite using `export --max-problems 10 --output smoke.jsonl`, then evaluate **all** models against that same file. Do not compare a smoke subset to full-dataset results. A report refuses different content hashes, protocols, or question IDs.

## Recommended battery

| Area | Benchmark / test | Role and metric |
|---|---|---|
| Cryptography | [AICrypto](https://github.com/wangyu-ovo/aicrypto-agent) | Cryptography-specific evaluation; use the authors' task-specific grading and distinguish no-tools from agent/tool tracks. |
| Network engineering | [NetoAI NetBench](https://huggingface.co/datasets/NetoAISolutions/NetBench) | Network SME question-answer evaluation; grade open answers with a fixed expert rubric. This is the QA dataset, not the similarly named packet-traffic benchmark. |
| Broad cybersecurity | [CyberMetric](https://github.com/cybermetric/CyberMetric) | Start with 500 questions, then run the larger set for the final evaluation; report accuracy and invalid-answer rate. |
| Threat intelligence | [CTI-MCQ / CTIBench](https://huggingface.co/datasets/AI4Sec/cti-bench) | Already integrated. Useful supplementary coverage, but not a replacement for crypto/networking tasks. |
| Security regression | MMLU computer_security | Already integrated; exposed to validation, so not a clean holdout. |
| General regression | ARC, MMLU, GSM8K, HumanEval, SpellingBee | Already run before/after; detect general-capability loss. Existing HumanEval/GSM8K protocols may involve code/calculator tools; keep these separate from the no-tool MCQ chart. |
| Objective practical skills | Fresh subnetting, route selection, packet/TLS interpretation, crypto test-vector exercises | Build hidden, independently checked cases; score exact computed answers or executable tests. |
| Teaching quality | 100 fresh crypto/networking explanations and troubleshooting scenarios | Blind expert grading for correctness, explanation quality, unsupported claims, and actionable troubleshooting; report each rubric dimension. |

Only CTI-MCQ and MMLU-ComputerSecurity are currently exported automatically by `cyber_benchmark`. Other rows above are recommendations, not implemented benchmark adapters. You can add reviewed MCQs using the JSONL schema below, then freeze the combined file before running any model:

```json
{"id":"networking:001","task":"Networking-Holdout","prompt":"Your question and labeled choices","letters":["A","B","C","D"],"answer":"C"}
```

Suggested custom holdout: 200 crypto questions, 200 networking questions, and 100 teaching scenarios. Split by source document, scenario family, and template before training; exact deduplication alone is insufficient. Keep a separate development set for checkpoint selection, and run the final holdout only after selecting the checkpoint. For final claims, repeat training with multiple seeds and use paired bootstrap intervals on per-question SFT-minus-base outcomes. Avoid collapsing unrelated metrics into one overall waterfall; use one chart per benchmark.

## Additional training material

| Source | Best use | Preparation |
|---|---|---|
| [NetSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/NetSpec-LLM) | Networking standards/domain text | Select topics relevant to your curriculum; derive reviewed instruction-answer pairs for SFT and reserve source documents for evaluation. |
| [Tele-Data](https://huggingface.co/datasets/AliMaatouk/Tele-Data) | Telecom continual pretraining | A large text corpus, not ready-made chat SFT; sample relevant material and convert it if using the current conversation loader. |
| [Wycheproof](https://github.com/C2SP/wycheproof) | Cryptographic edge cases and validation exercises | JSON test vectors for cryptographic libraries; derive tasks with verified answers and split by test family. Not a ready-made LLM benchmark or SFT dataset. |
| Your generated networking labs | IPv4/IPv6 subnetting, longest-prefix matching, DNS/TCP/TLS troubleshooting | Generate varied scenarios with deterministic validators; hold out address ranges, topologies, and templates. |

Keep AICrypto, CyberMetric, CTIBench, and the chosen NetBench evaluation material out of training. Check each source's current license and provenance before mixing it into a distributable dataset; CTIBench is marked CC-BY-NC-SA-4.0. Raw standards corpora and test vectors need adaptation rather than direct insertion into CustomJSON.

Validation for the comparison utilities:

```bash
python -m unittest discover -s tests -p 'test_cyber*.py' -v
```
