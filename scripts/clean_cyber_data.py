"""Conservative JSONL cleanup. Repaired output requires review before training."""
import argparse
import json
import re
from pathlib import Path

CONTENT = re.compile(r'("content"\s*:\s*")(.*?)("(?=\s*}\s*(?:,\s*{\s*"role"|]\s*})))')

def repair_content(match):
    text = re.sub(r'\\([:()<>])', r'\1', match[2])
    result = []
    slashes = 0
    for char in text:
        if char == '"' and slashes % 2 == 0:
            result.append('\\')
        result.append(char)
        slashes = slashes + 1 if char == '\\' else 0
    return match[1] + ''.join(result) + match[3]

def parse_record(raw):
    repaired = False
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        record = json.loads(CONTENT.sub(repair_content, raw))
        repaired = True
    messages = record.get("messages") if isinstance(record, dict) else record
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError("Expected complete user/assistant conversation")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Message must be an object")
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not isinstance(message.get("content"), str):
            raise ValueError("Expected alternating user/assistant text messages")
        if not message["content"].strip():
            raise ValueError("Empty message")
    # Keep text, addresses, Markdown and code byte-for-byte after JSON decoding.
    return messages, repaired

def clean(source, output_dir):
    source, output_dir = Path(source), Path(output_dir)
    # Refuse to overwrite previous output or source data.
    with source.open(encoding="utf-8-sig") as incoming:
        output_dir.mkdir(parents=True, exist_ok=False)
        counts = dict(kept=0, repaired=0, duplicates=0, rejected=0)
        seen = {}
        with (output_dir / "training_candidates.jsonl").open("x", encoding="utf-8") as output, (output_dir / "audit.jsonl").open("x", encoding="utf-8") as audit:
            for number, raw in enumerate(incoming, 1):
                if not raw.strip():
                    continue
                try:
                    messages, repaired = parse_record(raw)
                    key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
                    if key in seen:
                        counts["duplicates"] += 1
                        entry = dict(line=number, action="duplicate", duplicate_of=seen[key], original=raw)
                    else:
                        seen[key] = number
                        output.write(json.dumps(messages, ensure_ascii=False) + "\n")
                        counts["kept"] += 1
                        counts["repaired"] += int(repaired)
                        entry = dict(line=number, action="repaired" if repaired else "kept", original=raw)
                except (ValueError, TypeError) as error:
                    counts["rejected"] += 1
                    entry = dict(line=number, action="rejected", error=str(error), original=raw)
                audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
        return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(clean(args.input, args.output_dir), indent=2))
