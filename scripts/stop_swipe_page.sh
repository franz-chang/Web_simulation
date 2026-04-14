#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19002}"

get_pids() {
  lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
}

PIDS="$(get_pids)"

if [ -z "${PIDS}" ]; then
  echo "[INFO] No Web_sim service listening on ${HOST}:${PORT}"
  exit 0
fi

echo "[INFO] Stopping Web_sim service on ${HOST}:${PORT} ..."
echo "[INFO] Target PID(s): ${PIDS//$'\n'/ }"
kill ${PIDS} || true

STOPPED=0
for _ in $(seq 1 20); do
  if [ -z "$(get_pids)" ]; then
    STOPPED=1
    break
  fi
  sleep 0.25
done

if [ "${STOPPED}" -ne 1 ]; then
  FORCE_PIDS="$(get_pids)"
  if [ -n "${FORCE_PIDS}" ]; then
    echo "[WARN] Graceful shutdown timed out, force killing PID(s): ${FORCE_PIDS//$'\n'/ }"
    kill -9 ${FORCE_PIDS} || true
  fi
fi

if [ -n "$(get_pids)" ]; then
  echo "[ERROR] Failed to stop service on ${HOST}:${PORT}" >&2
  exit 1
fi

echo "[OK] Web_sim service stopped on ${HOST}:${PORT}"
