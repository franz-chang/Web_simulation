#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DATASET_DIR="${1:-}"
MODEL_SUFFIX="${2:-}"

if [[ -z "${DATASET_DIR}" || -z "${MODEL_SUFFIX}" ]]; then
  echo "Usage: $0 <dataset_dir> <model_suffix>" >&2
  echo "Example: $0 /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/All_Beauty amazon_all_beauty" >&2
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] Dataset dir not found: ${DATASET_DIR}" >&2
  exit 1
fi

if [ -x "/opt/anaconda3/bin/python3" ]; then
  PYTHON_BIN="/opt/anaconda3/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${PROJECT_ROOT}/artifacts"

SASREC_EPOCHS="${SASREC_EPOCHS:-10}"
LIGHTGCN_EPOCHS="${LIGHTGCN_EPOCHS:-30}"
MULTVAE_EPOCHS="${MULTVAE_EPOCHS:-30}"

echo "[INFO] Training on dataset: ${DATASET_DIR}"
echo "[INFO] Model suffix: ${MODEL_SUFFIX}"
echo "[INFO] Epochs => SASRec=${SASREC_EPOCHS}, LightGCN=${LIGHTGCN_EPOCHS}, MultVAE=${MULTVAE_EPOCHS}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_sasrec.py" \
  --dataset-dir "${DATASET_DIR}" \
  --output-model "${PROJECT_ROOT}/artifacts/sasrec_${MODEL_SUFFIX}.pt" \
  --epochs "${SASREC_EPOCHS}" \
  --eval-ks 10,20

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_lightgcn.py" \
  --dataset-dir "${DATASET_DIR}" \
  --output-model "${PROJECT_ROOT}/artifacts/lightgcn_${MODEL_SUFFIX}.pt" \
  --epochs "${LIGHTGCN_EPOCHS}" \
  --eval-ks 10,20

"${PYTHON_BIN}" "${PROJECT_ROOT}/train_multvae.py" \
  --dataset-dir "${DATASET_DIR}" \
  --output-model "${PROJECT_ROOT}/artifacts/multvae_${MODEL_SUFFIX}.pt" \
  --epochs "${MULTVAE_EPOCHS}" \
  --eval-ks 10,20

echo "[DONE] Finished all 3 models for ${MODEL_SUFFIX}."
