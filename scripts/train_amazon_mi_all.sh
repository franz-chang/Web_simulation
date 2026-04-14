#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DATASET_DIR="${1:-$(cd "${PROJECT_ROOT}/.." && pwd -P)/WebSim_Dataset/amazon_v2/Musical_Instruments}"

if [ -x "/opt/anaconda3/bin/python3" ]; then
  PYTHON_BIN="/opt/anaconda3/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${PROJECT_ROOT}/artifacts"

echo "Training on dataset: $DATASET_DIR"

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_sasrec.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "${PROJECT_ROOT}/artifacts/sasrec_amazon_mi.pt" \
  --epochs 10 \
  --eval-ks 10,20

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_lightgcn.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "${PROJECT_ROOT}/artifacts/lightgcn_amazon_mi.pt" \
  --epochs 30 \
  --eval-ks 10,20

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_multvae.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-model "${PROJECT_ROOT}/artifacts/multvae_amazon_mi.pt" \
  --epochs 30 \
  --eval-ks 10,20

echo "All three models finished."
