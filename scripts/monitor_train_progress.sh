#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ARTIFACT_DIR="${PROJECT_ROOT}/artifacts"

TRAIN_LOG="${TRAIN_LOG:-}"
if [[ -z "${TRAIN_LOG}" ]]; then
  TRAIN_LOG="$(ls -1t "${ARTIFACT_DIR}"/train_*_screen_*.log 2>/dev/null | head -n 1 || true)"
fi

REPORT_LOG="${REPORT_LOG:-${ARTIFACT_DIR}/train_progress_5min.log}"
if [[ "${REPORT_LOG}" != /* ]]; then
  REPORT_LOG="${PROJECT_ROOT}/${REPORT_LOG}"
fi

mkdir -p "$(dirname "${REPORT_LOG}")"
touch "${REPORT_LOG}"
echo "==== monitor started at $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$REPORT_LOG"
if [[ -n "${TRAIN_LOG}" ]]; then
  echo "==== tracking train log: ${TRAIN_LOG} ====" >> "$REPORT_LOG"
else
  echo "==== train log not found yet; waiting for artifacts/train_*_screen_*.log ====" >> "$REPORT_LOG"
fi

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  proc="$(pgrep -af 'python(3)? .*train_(sasrec|lightgcn|multvae)\.py' | head -n 1 || true)"

  if [[ -z "${TRAIN_LOG}" || ! -f "${TRAIN_LOG}" ]]; then
    TRAIN_LOG="$(ls -1t "${ARTIFACT_DIR}"/train_*_screen_*.log 2>/dev/null | head -n 1 || true)"
  fi

  if [[ -n "${TRAIN_LOG}" && -f "${TRAIN_LOG}" ]]; then
    model="$(rg '^\[MODEL\]' "$TRAIN_LOG" | tail -n 1 | sed 's/^\[MODEL\] //g' || true)"
    epoch="$(rg 'Epoch ' "$TRAIN_LOG" | tail -n 1 || true)"
    milestone="$(rg '^Best epoch|^Saved artifact|^\[DONE\]' "$TRAIN_LOG" | tail -n 1 || true)"
  else
    model=""
    epoch=""
    milestone=""
  fi

  if [[ -z "$model" ]]; then
    model="unknown"
  fi
  if [[ -z "$epoch" ]]; then
    epoch="(no epoch yet)"
  fi
  if [[ -z "$milestone" ]]; then
    milestone="(no new milestone)"
  fi
  if [[ -z "$proc" ]]; then
    proc="(no active trainer process)"
  fi
  if [[ -z "${TRAIN_LOG}" || ! -f "${TRAIN_LOG}" ]]; then
    milestone="(train log not found yet)"
  fi

  echo "[$ts] model=$model | epoch=$epoch | milestone=$milestone | proc=$proc | train_log=${TRAIN_LOG:-N/A}" >> "$REPORT_LOG"
  sleep 300
done
