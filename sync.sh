#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -f .env ]]; then
  # Strip Windows line endings before Bash reads the local configuration.
  source <(sed 's/\r$//' .env)
fi

for setting in REMOTE_USER REMOTE_HOST REMOTE_PORT REMOTE_BASE; do
  if [[ -z "${!setting:-}" ]]; then
    echo "Missing $setting. Set it in .env or the environment." >&2
    exit 1
  fi
done
if [[ ! "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
  echo "REMOTE_PORT must contain only digits (for example, REMOTE_PORT=222)." >&2
  exit 1
fi

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash sync.sh [--check | --diff | --verify]"
  echo "--check tests SSH login; --diff previews uploads; --verify compares file contents after upload."
  echo "Without an option, uploads changed files to REMOTE_BASE."
  echo "Configure REMOTE_USER, REMOTE_HOST, REMOTE_PORT, and REMOTE_BASE in .env."
  exit 0
fi

for command in ssh rsync; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing $command. Run this script in a Bash environment with git, SSH, and rsync installed (such as MobaXterm or WSL)." >&2
    exit 1
  fi
done

SSH=(ssh -p "$REMOTE_PORT" -o ConnectTimeout=15)
LOGIN="${REMOTE_USER}@${REMOTE_HOST}"

if [[ "${1:-}" == "--check" ]]; then
  "${SSH[@]}" "$LOGIN" 'hostname && command -v rsync'
  exit 0
fi

EXCLUDES=(
  --exclude '.env' --exclude '.env.*'
  --exclude '.git/' --exclude '.venv/' --exclude 'logs/' --exclude 'data/'
  --exclude 'wandb/' --exclude '__pycache__/' --exclude '*.pyc'
  --filter=':- .gitignore'
)
REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"

if [ "${1:-}" = "--diff" ]; then
  shift
  echo "Files to upload to $REMOTE_BASE:"
  rsync -ahni --out-format='%n' "${EXCLUDES[@]}" -e "${SSH[*]}" ./ "$REMOTE"
  echo "(nothing listed above = already identical)"
  exit 0
fi

if [ "${1:-}" = "--verify" ]; then
  echo "Comparing local and remote file contents in $REMOTE_BASE..."
  differences=$(rsync -ahnc --out-format='%n' "${EXCLUDES[@]}" -e "${SSH[*]}" ./ "$REMOTE")
  if [[ -n "$differences" ]]; then
    printf 'Files that differ or are missing remotely:\n%s\n' "$differences" >&2
    exit 1
  fi
  echo "Verified: uploaded files match local files byte for byte."
  exit 0
fi

echo "Uploading changes to $REMOTE_BASE"
rsync -ah --info=progress2 --partial "${EXCLUDES[@]}" \
  -e "${SSH[*]}" ./ "$REMOTE"
