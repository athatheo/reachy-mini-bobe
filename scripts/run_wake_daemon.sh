#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/config/wake-daemon.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy config/wake-daemon.env.example and set BOBE_WAKE_TOKEN." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

cd "${ROOT}"
UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv not found; install uv or set UV_BIN" >&2
  exit 127
fi

# --extra wake-daemon keeps faster-whisper/fastapi installed even when other
# uv operations re-sync the venv without extras (this bit us once: a dependency
# change pruned the extras and the daemon lost Whisper until restart).
exec "${UV_BIN}" run --extra wake-daemon bobe-wake-daemon
