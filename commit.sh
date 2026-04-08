#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] Not inside a git repository: $ROOT_DIR" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
  echo "[ERROR] Detached HEAD detected. Checkout a branch before committing." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  COMMIT_MSG="$*"
else
  COMMIT_MSG="chore: update $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "[INFO] Repository: $ROOT_DIR"
echo "[INFO] Branch: $BRANCH"
echo "[INFO] Commit message: $COMMIT_MSG"

git status --short

echo "[INFO] Pulling latest changes with rebase..."
git pull --rebase --autostash origin "$BRANCH"

echo "[INFO] Staging all changes..."
git add -A

if git diff --cached --quiet; then
  echo "[INFO] No staged changes. Nothing to commit."
  exit 0
fi

echo "[INFO] Creating commit..."
git commit -m "$COMMIT_MSG"

echo "[INFO] Pushing to origin/$BRANCH ..."
git push origin "$BRANCH"

echo "[OK] Done."
