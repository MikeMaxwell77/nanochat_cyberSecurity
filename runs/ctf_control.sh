#!/bin/bash
set -euo pipefail

: "${TREATMENT_TAG:?Set TREATMENT_TAG to the completed CTF fine-tuning tag}"
if [[ ! "$TREATMENT_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TREATMENT_TAG must be a checkpoint directory name." >&2
  exit 1
fi
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
TREATMENT_DIR="$NANOCHAT_BASE_DIR/comparisons/$TREATMENT_TAG"
CONTROL_TAG="${TREATMENT_TAG}-no-ctf"
CONTROL_DIR="$NANOCHAT_BASE_DIR/comparisons/$CONTROL_TAG"
test -f "$TREATMENT_DIR/suite.jsonl"
test -f "$TREATMENT_DIR/sft.json"
test ! -e "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$CONTROL_TAG"

metadata=$(python - "$TREATMENT_TAG" <<'PY'
import glob, json, os, sys
from nanochat.common import get_base_dir
root = os.path.join(get_base_dir(), "chatsft_checkpoints", sys.argv[1])
final = []
for path in glob.glob(os.path.join(root, "meta_*.json")):
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("checkpoint_kind") == "final":
        final.append(meta)
if len(final) != 1:
    raise SystemExit(f"Expected one final CTF checkpoint in {root}; found {len(final)}")
meta = final[0]
base = meta["base_checkpoint"]
config = meta["user_config"]
if config.get("exclude_ctf"):
    raise SystemExit("Selected treatment already excludes CTF data")
print(base["model_tag"], base["step"], meta["step"],
      config.get("device_batch_size") or 16,
      config.get("mmlu_epochs", 3), config.get("gsm8k_epochs", 4))
PY
)
read -r BASE_TAG BASE_STEP STEPS DEVICE_BATCH MMLU_EPOCHS GSM8K_EPOCHS <<< "$metadata"
if (( STEPS < 1 )); then
  echo "Treatment checkpoint has no training steps." >&2
  exit 1
fi
mkdir -p "$CONTROL_DIR"
echo "Control: $CONTROL_TAG | base: $BASE_TAG step $BASE_STEP | SFT steps: $STEPS"

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  --model-tag "$BASE_TAG" --model-step "$BASE_STEP" --output-tag "$CONTROL_TAG" \
  --num-iterations "$STEPS" --exclude-ctf --device-batch-size "$DEVICE_BATCH" \
  --mmlu-epochs "$MMLU_EPOCHS" --gsm8k-epochs "$GSM8K_EPOCHS" \
  --save-every 200 --run dummy

CONTROL_STEP=$(python - "$CONTROL_TAG" <<'PY'
import os, sys
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_last_step
print(find_last_step(os.path.join(get_base_dir(), "chatsft_checkpoints", sys.argv[1])))
PY
)
if [[ "$CONTROL_STEP" != "$STEPS" ]]; then
  echo "Control ended at step $CONTROL_STEP; expected $STEPS." >&2
  exit 1
fi
TASKS='ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval|SpellingBee|CTI-MCQ|MMLU-ComputerSecurity'
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- \
  -i sft -g "$CONTROL_TAG" -s "$CONTROL_STEP" -a "$TASKS" \
  --output "$CONTROL_DIR/sft_general.json"
python -m scripts.cyber_benchmark local --source sft --model-tag "$CONTROL_TAG" \
  --step "$CONTROL_STEP" --suite "$TREATMENT_DIR/suite.jsonl" \
  --output "$CONTROL_DIR/sft.json"
python - "$TREATMENT_DIR/sft.json" "$CONTROL_DIR/sft.json" "$CONTROL_DIR/ctf_ablation.md" "$TREATMENT_DIR/sft_general.json" "$CONTROL_DIR/sft_general.json" <<'PY'
import json, sys
from pathlib import Path
from scripts.cyber_benchmark import compare_runs
runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:3]]
without, with_ctf = compare_runs([runs[1], runs[0]])
lines = ["# CTF data ablation", "", "Same base checkpoint, training steps, and evaluation suite.",
         "", "| Task | N | Without CTF % | With CTF % | CTF difference (pp) |",
         "|---|---:|---:|---:|---:|"]
for task in without:
    n = without[task]["n"]
    no_ctf = without[task]["accuracy"] * 100
    ctf = with_ctf[task]["accuracy"] * 100
    lines.append(f"| {task} | {n} | {no_ctf:.2f} | {ctf:.2f} | {ctf-no_ctf:+.2f} |")
original = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))["results"]
control = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))["results"]
if original.keys() != control.keys():
    raise ValueError("General evaluations used different task sets")
lines += ["", "## General tasks", "", "| Task | Without CTF % | With CTF % | CTF difference (pp) |",
          "|---|---:|---:|---:|"]
for task in original:
    no_ctf = control[task] * 100
    ctf = original[task] * 100
    lines.append(f"| {task} | {no_ctf:.2f} | {ctf:.2f} | {ctf-no_ctf:+.2f} |")
Path(sys.argv[3]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY