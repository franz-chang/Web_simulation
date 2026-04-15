#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DEFAULT_DATASET_DIR="$(cd "${PROJECT_ROOT}/.." && pwd -P)/WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions"
DATASET_DIR="${1:-${DEFAULT_DATASET_DIR}}"

exec "${SCRIPT_DIR}/train_amazon_mm2018_all.sh" "${DATASET_DIR}" "amazon_magazine_subscriptions"
