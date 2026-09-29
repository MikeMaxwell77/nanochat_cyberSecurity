"""Check uploaded text files before submitting a long cluster job."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_SUFFIXES = {".sh", ".bash", ".slurm", ".sbatch"}
SKIP_DIRS = {".git", ".venv", "logs", "data", "wandb", "__pycache__", ".pytest_cache"}


def file_paths():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return [ROOT / part.decode("utf-8", "surrogateescape") for part in result.stdout.split(b"\0") if part]
    paths = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        paths.extend(Path(directory) / name for name in files if not name.startswith(".env"))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="convert CRLF text files to LF")
    args = parser.parse_args()
    paths = file_paths()
    failures = []
    shell_files = []
    checked = 0
    fixed = 0

    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        if b"\r\n" in data:
            if args.fix:
                data = data.replace(b"\r\n", b"\n")
                path.write_bytes(data)
                fixed += 1
            else:
                failures.append(f"{relative}: CRLF line endings")
        if path.suffix.lower() in SHELL_SUFFIXES or data.startswith(b"#!") and b"sh" in data.split(b"\n", 1)[0]:
            shell_files.append(path)
            if b"\r" in data.replace(b"\r\n", b""):
                failures.append(f"{relative}: bare carriage return")

    bash = shutil.which("bash")
    if bash:
        for path in shell_files:
            result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                failures.append(f"{path.relative_to(ROOT)}: bash syntax error\n{result.stderr.strip()}")
    else:
        print("Warning: bash unavailable; shell syntax was not checked. Run this again on the cluster.")

    if failures:
        print("Preflight failed:")
        print("\n".join(failures))
        return 1
    print(f"Preflight passed: {checked} text files checked, {len(shell_files)} shell scripts found, {fixed} files converted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
