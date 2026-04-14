#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/.git" ]]; then
  echo "[ERROR] Not inside a git repository: ${SCRIPT_DIR}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
  echo "[ERROR] Detached HEAD detected. Checkout a branch before committing." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "[ERROR] Remote 'origin' is not configured." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  COMMIT_MSG="$*"
else
  COMMIT_MSG="chore: update $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "[INFO] Repository: $REPO_ROOT"
echo "[INFO] Branch: $BRANCH"
echo "[INFO] Commit message: $COMMIT_MSG"

git status --short

echo "[INFO] Staging all changes..."
git add -A

if git diff --cached --quiet; then
  echo "[INFO] No staged changes. Nothing to commit."
else
  echo "[INFO] Creating commit..."
  git commit -m "$COMMIT_MSG"
fi

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  echo "[INFO] Pulling latest changes with rebase..."
  git pull --rebase origin "$BRANCH"
else
  echo "[INFO] Remote branch origin/$BRANCH does not exist yet. Skip pull."
fi

echo "[INFO] Pushing to origin/$BRANCH ..."
git push -u origin "$BRANCH"

echo "[OK] Done."
