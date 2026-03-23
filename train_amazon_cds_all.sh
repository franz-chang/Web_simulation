#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${1:-$(cd "$ROOT_DIR/.." && pwd)/WebSim_Dataset/amazon_v2/Musical_Instruments}"

echo "Training on dataset: $DATASET_DIR"

python3 "$ROOT_DIR/train_sasrec.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "$ROOT_DIR/artifacts/sasrec_amazon_mi.pt" \
  --epochs 10 \
  --eval-ks 10,20

python3 "$ROOT_DIR/train_lightgcn.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "$ROOT_DIR/artifacts/lightgcn_amazon_mi.pt" \
  --epochs 30 \
  --eval-ks 10,20

python3 "$ROOT_DIR/train_multvae.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "$ROOT_DIR/artifacts/multvae_amazon_mi.pt" \
  --epochs 30 \
  --eval-ks 10,20

echo "All three models finished."
