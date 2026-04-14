#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19002}"
URL="http://${HOST}:${PORT}/swipe"
HEALTH_URL="http://${HOST}:${PORT}/health"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/web.log}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
OPEN_BROWSER_NORMALIZED="$(printf '%s' "${OPEN_BROWSER}" | tr '[:upper:]' '[:lower:]')"

if [[ "${LOG_FILE}" != /* ]]; then
  LOG_FILE="${PROJECT_ROOT}/${LOG_FILE}"
fi

if lsof -tiTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[INFO] Web_sim service already running on ${HOST}:${PORT}"
else
  if [ -x "/opt/anaconda3/bin/python3" ]; then
    PYTHON_BIN="/opt/anaconda3/bin/python3"
  else
    PYTHON_BIN="$(command -v python3)"
  fi

  echo "[INFO] Starting Web_sim service on ${HOST}:${PORT} ..."
  PORT="${PORT}" nohup "${PYTHON_BIN}" "${PROJECT_ROOT}/app.py" >"${LOG_FILE}" 2>&1 &
  APP_PID=$!
  echo "[INFO] Service PID: ${APP_PID}"

  READY=0
  for _ in $(seq 1 60); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 0.5
  done

  if [ "${READY}" -ne 1 ]; then
    echo "[ERROR] Service did not become ready in time. Check log: ${LOG_FILE}" >&2
    exit 1
  fi
fi

if [[ "${OPEN_BROWSER_NORMALIZED}" != "0" && "${OPEN_BROWSER_NORMALIZED}" != "false" && "${OPEN_BROWSER_NORMALIZED}" != "no" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${URL}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}" >/dev/null 2>&1 || true
  fi
fi

echo "[OK] Swipe page: ${URL}"
