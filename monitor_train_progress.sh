#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/chongzhang/Web_sim"
TRAIN_LOG="$ROOT_DIR/artifacts/train_amazon_mi_all_screen_20260317_162910.log"
REPORT_LOG="$ROOT_DIR/artifacts/train_progress_5min.log"

touch "$REPORT_LOG"
echo "==== monitor started at $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$REPORT_LOG"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  proc="$(pgrep -af '/opt/anaconda3/bin/python3 .*train_(sasrec|lightgcn|multvae)\.py' | head -n 1 || true)"
  model="$(rg '^\[MODEL\]' "$TRAIN_LOG" | tail -n 1 | sed 's/^\[MODEL\] //g' || true)"
  epoch="$(rg 'Epoch ' "$TRAIN_LOG" | tail -n 1 || true)"
  milestone="$(rg '^Best epoch|^Saved artifact|^\[DONE\]' "$TRAIN_LOG" | tail -n 1 || true)"

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

  echo "[$ts] model=$model | epoch=$epoch | milestone=$milestone | proc=$proc" >> "$REPORT_LOG"
  sleep 300
done
