#!/usr/bin/env bash
# Fallback LLM backend: serve Qwen3.5-9B with vLLM inside WSL2 (Ubuntu) when a
# working llama.cpp GGUF is unavailable. Exposes the same OpenAI API on :8000,
# so the orchestrator needs no changes (LLM_BASE_URL stays http://127.0.0.1:8000/v1).
#
# Setup (one time, inside WSL2 with NVIDIA CUDA on WSL configured):
#   uv venv && source .venv/bin/activate
#   uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
#
# Run:
#   bash scripts/run_llm_wsl_vllm.sh
set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-9B}"

# --language-model-only skips the vision encoder (we are text-only) to free VRAM/KV.
# --enable-prefix-caching speeds up multi-turn TTFT. --reasoning-parser qwen3 keeps
# thinking parseable; we disable thinking per-request from the orchestrator.
vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --language-model-only \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
