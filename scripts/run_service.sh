#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

PORT="${PORT:-19002}"

if [ -x "/opt/anaconda3/bin/python3" ]; then
  PYTHON_BIN="/opt/anaconda3/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

exec env PORT="${PORT}" "${PYTHON_BIN}" "${PROJECT_ROOT}/app.py"
