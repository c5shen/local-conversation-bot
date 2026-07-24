#!/usr/bin/env bash
# End-to-end launcher for the Local Voice Chatbot (Linux / WSL2 / Git Bash).
# Starts: whisper-server (STT, :8081), llama-server (LLM, :8000), orchestrator (:8080).
# Override the paths below via environment variables before running.
set -euo pipefail
cd "$(dirname "$0")"

# --- Auto-install dependencies if anything is missing ---
if [ ! -x .venv/bin/python ] || [ ! -x tools/llama/llama-server ] || [ ! -x tools/whisper/Release/whisper-server ]; then
  echo "Some dependencies are missing - running setup first..."
  bash scripts/setup.sh
fi

WHISPER_BIN="${WHISPER_BIN:-tools/whisper/Release/whisper-server}"
WHISPER_MODEL="${WHISPER_MODEL:-tools/whisper/models/ggml-large-v3-turbo.bin}"
LLAMA_BIN="${LLAMA_BIN:-tools/llama/llama-server}"
QWEN_GGUF="${QWEN_GGUF:-data/models/Qwen3.5-9B-Q4_K_M.gguf}"
APP_HOST="${HOST:-127.0.0.1}"
APP_PORT="${PORT:-8080}"

# Activate the project venv if present.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pids=()
cleanup() {
  echo "Stopping services..."
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "[1/3] whisper-server (STT) on http://127.0.0.1:8081"
"$WHISPER_BIN" -m "$WHISPER_MODEL" --host 127.0.0.1 --port 8081 -t 8 &
pids+=($!)

echo "[2/3] llama-server (LLM) on http://127.0.0.1:8000"
"$LLAMA_BIN" -m "$QWEN_GGUF" -ngl 99 -c "${CTX:-32768}" -fa on --host 127.0.0.1 --port 8000 &
pids+=($!)

echo "Waiting 8s for the model servers to load..."
sleep 8

echo "[3/3] orchestrator (UI + WebSocket) on http://${APP_HOST}:${APP_PORT}"
python -m uvicorn server.main:app --host "$APP_HOST" --port "$APP_PORT" &
pids+=($!)

echo "All services started. Open http://localhost:${APP_PORT}"
wait
