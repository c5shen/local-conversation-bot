"""GPU usage (NVML) and LLM context-token metrics for the status display."""
from __future__ import annotations

import httpx

from .config import settings

# llama-server root (strip the trailing /v1 from the OpenAI base URL).
_LLM_ROOT = settings.llm_base_url.rsplit("/v1", 1)[0]

_nvml = None  # None=untried, False=unavailable, else (pynvml, handle)
_ctx_limit: int | None = None
_current_tokens: int = 0


def _nvml_handle():
    global _nvml
    if _nvml is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:  # noqa: BLE001 - no NVIDIA GPU / driver
            _nvml = False
    return _nvml or None


def gpu_stats() -> dict | None:
    """Whole-device memory + utilization (covers llama, whisper, Qwen3-TTS together)."""
    handle = _nvml_handle()
    if not handle:
        return None
    pynvml, dev = handle
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
        util = pynvml.nvmlDeviceGetUtilizationRates(dev)
        name = pynvml.nvmlDeviceGetName(dev)
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        return {
            "name": name,
            "mem_used_mb": round(mem.used / 1024 / 1024),
            "mem_total_mb": round(mem.total / 1024 / 1024),
            "util_pct": util.gpu,
        }
    except Exception:  # noqa: BLE001
        return None


async def context_limit() -> int:
    """Model context window (n_ctx) reported by llama-server; cached."""
    global _ctx_limit
    if _ctx_limit is None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                data = (await client.get(f"{_LLM_ROOT}/props")).json()
            _ctx_limit = int(data.get("default_generation_settings", {}).get("n_ctx") or 8192)
        except Exception:  # noqa: BLE001
            _ctx_limit = 8192
    return _ctx_limit


async def count_tokens(text: str) -> int | None:
    """Token count of text via llama-server /tokenize (None on failure)."""
    if not text:
        return 0
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            data = (await client.post(f"{_LLM_ROOT}/tokenize", json={"content": text})).json()
        tokens = data.get("tokens", data if isinstance(data, list) else [])
        return len(tokens)
    except Exception:  # noqa: BLE001
        return None


def set_current_tokens(count: int) -> None:
    global _current_tokens
    _current_tokens = count


async def snapshot() -> dict:
    return {
        "gpu": gpu_stats(),
        "tokens": {"current": _current_tokens, "max": await context_limit()},
    }
