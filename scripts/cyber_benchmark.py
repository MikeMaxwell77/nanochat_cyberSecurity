"""Frozen MCQ battery: export, evaluate local/GPT-4 models, and plot paired results.

Use `python -m scripts.cyber_benchmark --help`. API calls only occur with `api`.
"""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROTOCOL = {"name": "generated-letter-v1", "temperature": 0.0, "max_tokens": 32,
            "instruction": "\nReply with only the letter of the correct answer."}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_suite(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty suite")
    ids = set()
    for row in rows:
        if row["id"] in ids or not row["task"] or not row["prompt"]:
            raise ValueError("Suite requires unique IDs and nonempty task/prompt")
        letters = row["letters"]
        if len(letters) < 2 or len(set(letters)) != len(letters) or any(len(x) != 1 for x in letters):
            raise ValueError("Choices must have distinct single-letter labels")
        if row["answer"] not in letters:
            raise ValueError("Answer must be a choice label")
        ids.add(row["id"])
    return rows


def score_answer(text, row):
    # Intentionally strict and identical for every backend. Explanations count as invalid.
    answer = text.strip()
    return {"prediction": answer, "valid": answer in row["letters"],
            "correct": answer == row["answer"]}


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["task"]].append(row)
    results = {}
    for task, rows in groups.items():
        n = len(rows)
        accuracy = sum(r["correct"] for r in rows) / n
        z = 1.959963984540054
        center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
        margin = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
        results[task] = {"n": n, "accuracy": accuracy,
                         "accuracy_wilson95": [max(0, center-margin), min(1, center+margin)],
                         "invalid": sum(not r["valid"] for r in rows)}
    return results


def export_suite(args):
    from tasks.cybersecurity import CybersecurityMCQ
    rows = []
    for name in args.tasks.split("|"):
        task = CybersecurityMCQ(benchmark=name)
        count = min(len(task), args.max_problems) if args.max_problems else len(task)
        for i in range(count):
            conversation = task[i]
            rows.append({"id": f"{name}:{i}", "task": name,
                         "prompt": conversation["messages"][0]["content"],
                         "answer": conversation["messages"][-1]["content"],
                         "letters": list(conversation["letters"]),
                         "dataset_fingerprint": task.ds._fingerprint})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} questions; suite SHA256: {digest(rows)}")


def evaluate(args):
    rows = read_suite(args.suite)
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial.jsonl")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    if args.command == "local":
        from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine
        ddp, _, _, _, device = compute_init(args.device_type or autodetect_device_type())
        if ddp:
            raise ValueError("Run this battery with python, not torchrun")
        model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                            model_tag=args.model_tag, step=args.step)
        engine = Engine(model, tokenizer)
        identity = {"source": args.source, **meta["checkpoint"]}

        def generate(prompt):
            conversation = {"messages": [{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": ""}]}
            tokens = tokenizer.render_for_completion(conversation)
            if len(tokens) + PROTOCOL["max_tokens"] > model.config.sequence_len:
                raise ValueError("Question exceeds local context budget; revise the shared suite")
            result, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=PROTOCOL["max_tokens"],
                                               temperature=0.0, top_k=None, allow_tools=False)
            return tokenizer.decode(result[0][len(tokens):]), {}
    else:
        from urllib.request import Request, urlopen
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY before running the API benchmark")
        identity = {"source": "openai", "model": args.model}

        def generate(prompt):
            body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": PROTOCOL["max_tokens"]}
            request = Request("https://api.openai.com/v1/chat/completions",
                              data=json.dumps(body).encode(),
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            choice = result["choices"][0]
            return choice["message"].get("content") or "", {
                "model": result["model"], "usage": result.get("usage"),
                "system_fingerprint": result.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason")}
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("x", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                response, details = generate(row["prompt"] + PROTOCOL["instruction"])
                record = {"id": row["id"], "task": row["task"], "response": response,
                          **score_answer(response, row), "details": details}
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"{i+1}/{len(rows)} {row['task']}: {record['correct']}", flush=True)
        write_json(output, {"suite_sha256": digest(rows), "protocol": PROTOCOL,
                            "model": identity, "created_at": datetime.now(timezone.utc).isoformat(),
                            "results": summarize(records), "records": records})
    finally:
        if args.command == "local":
            compute_cleanup()


def compare_runs(runs):
    reference = runs[0]
    expected = [(r["id"], r["task"]) for r in reference["records"]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Empty or duplicate evaluation records")
    for run in runs:
        if run["suite_sha256"] != reference["suite_sha256"] or run["protocol"] != reference["protocol"]:
            raise ValueError("Cannot compare different suites or scoring protocols")
        if [(r["id"], r["task"]) for r in run["records"]] != expected:
            raise ValueError("Missing, reordered, or mismatched question IDs")
    return [summarize(run["records"]) for run in runs]


def report(args):
    paths = [args.base, args.sft] + ([args.gpt4] if args.gpt4 else [])
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    summaries = compare_runs(runs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(summaries[0])
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 4 * len(tasks)), squeeze=False)
    lines = ["# Measured benchmark comparison", "", "Generated-letter accuracy; improvements are percentage points.",
             "GPT-4 is an external reference, not a training stage. No tool access for any model.", "",
             "| Task | N | Base % | SFT % | SFT gain (pp) | GPT-4 % |", "|---|---:|---:|---:|---:|---:|"]
    for ax, task in zip(axes[:, 0], tasks):
        b, s = [summary[task]["accuracy"] * 100 for summary in summaries[:2]]
        delta = s - b
        ax.bar(0, b, color="#64748b")
        ax.bar(1, abs(delta), bottom=min(b, s), color="#16a34a" if delta >= 0 else "#dc2626")
        ax.bar(2, s, color="#2563eb")
        ax.plot([0.4, 1.4], [b, b], color="gray", linestyle="--")
        ax.plot([1.4, 2.4], [s, s], color="gray", linestyle="--")
        ax.text(1, max(b, s) + 2, f"{delta:+.1f} pp", ha="center")
        labels = ["Base", "SFT change", "Fine-tuned"]
        g = None
        if len(summaries) == 3:
            g = summaries[2][task]["accuracy"] * 100
            ax.bar(3, g, color="#9333ea")
            labels.append("GPT-4 reference")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(task)
        g_text = f"{g:.2f}" if g is not None else "pending"
        lines.append(f"| {task} | {summaries[0][task]['n']} | {b:.2f} | {s:.2f} | {delta:+.2f} | {g_text} |")
    fig.tight_layout()
    fig.savefig(out / "waterfall.png", dpi=160)
    fig.savefig(out / "waterfall.svg")
    plt.close(fig)
    lines += ["", "Invalid answers (included as incorrect):", "", "```json", json.dumps(summaries, indent=2), "```",
              "", "Model identities:", "", "```json", json.dumps([r["model"] for r in runs], indent=2), "```"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report and waterfall charts: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--tasks", default="CTI-MCQ|MMLU-ComputerSecurity")
    export.add_argument("--max-problems", type=int)
    export.add_argument("--output", required=True)
    for name in ("local", "api"):
        command = commands.add_parser(name)
        command.add_argument("--suite", required=True)
        command.add_argument("--output", required=True)
        if name == "local":
            command.add_argument("--source", choices=["base", "sft"], required=True)
            command.add_argument("--model-tag", required=True)
            command.add_argument("--step", type=int, required=True)
            command.add_argument("--device-type", default="")
        else:
            command.add_argument("--model", default="gpt-4")
    plot = commands.add_parser("report")
    plot.add_argument("--base", required=True)
    plot.add_argument("--sft", required=True)
    plot.add_argument("--gpt4")
    plot.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "export" and args.max_problems is not None and args.max_problems <= 0:
        parser.error("--max-problems must be positive")
    if args.command == "export":
        export_suite(args)
    elif args.command == "report":
        report(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
