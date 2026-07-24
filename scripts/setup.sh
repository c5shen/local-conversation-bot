#!/usr/bin/env bash
# One-shot, idempotent setup for Linux / WSL2 (NVIDIA CUDA).
# Installs the Python venv + deps, and builds llama.cpp and whisper.cpp from source
# with CUDA (no prebuilt CUDA Linux binaries are published). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== Local Voice Chatbot setup ==  $ROOT"
# (Qwen3-TTS handles every language natively - no optional phonemizer packs needed.)

WHISPER_MODEL="ggml-large-v3-turbo.bin"
GGUF="data/models/Qwen3.5-9B-Q4_K_M.gguf"

mkdir -p tools/src tools/llama tools/whisper/Release tools/whisper/models data/models

# --- 1. Python venv + deps ---
export PATH="$HOME/.local/bin:$PATH"
echo "[1/5] Python environment + dependencies..."
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python --torch-backend cu124 -r requirements.txt
else
  [ -d .venv ] || python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
  ./.venv/bin/pip install -r requirements.txt
fi

# --- 2. llama.cpp (build with CUDA) ---
if [ ! -x tools/llama/llama-server ]; then
  echo "[2/5] Building llama.cpp (CUDA)..."
  [ -d tools/src/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/src/llama.cpp
  cmake -S tools/src/llama.cpp -B tools/src/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
  cmake --build tools/src/llama.cpp/build --config Release -j
  cp -f tools/src/llama.cpp/build/bin/* tools/llama/ 2>/dev/null || true
else
  echo "[2/5] llama.cpp already built."
fi

# --- 3. whisper.cpp (build with CUDA) ---
if [ ! -x tools/whisper/Release/whisper-server ]; then
  echo "[3/5] Building whisper.cpp (CUDA)..."
  [ -d tools/src/whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp tools/src/whisper.cpp
  cmake -S tools/src/whisper.cpp -B tools/src/whisper.cpp/build -DGGML_CUDA=1
  cmake --build tools/src/whisper.cpp/build --config Release -j
  cp -f tools/src/whisper.cpp/build/bin/* tools/whisper/Release/ 2>/dev/null || true
else
  echo "[3/5] whisper.cpp already built."
fi

# --- 4. whisper model ---
if [ ! -f "tools/whisper/models/$WHISPER_MODEL" ]; then
  echo "[4/5] Downloading whisper model $WHISPER_MODEL..."
  curl -L -o "tools/whisper/models/$WHISPER_MODEL" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL"
else
  echo "[4/5] whisper model present."
fi

# --- 5. Qwen GGUF + .env ---
if [ ! -f "$GGUF" ]; then
  if [ -n "${QWEN_GGUF_URL:-}" ]; then
    echo "[5/5] Downloading Qwen GGUF..."
    curl -L -o "$GGUF" "$QWEN_GGUF_URL"
  else
    echo "[5/5] WARNING: Qwen GGUF not found at $GGUF."
    echo "      Place a Qwen3.5-9B Q4_K_M GGUF there or set QWEN_GGUF_URL and re-run."
  fi
else
  echo "[5/5] Qwen GGUF present."
fi
[ -f .env ] || cp .env.example .env

echo ""
echo "== Setup complete. Run ./start_server.sh to launch. =="
