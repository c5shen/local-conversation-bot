# Downloads the model weights needed by the voice chatbot.
# Edit the variables below to point at your local checkouts / desired files.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\setup_models.ps1

$ErrorActionPreference = "Stop"

# --- whisper.cpp model ---
# Path to your whisper.cpp checkout (built with CUDA, see README).
$WhisperRepo = $env:WHISPER_REPO; if (-not $WhisperRepo) { $WhisperRepo = "..\whisper.cpp" }
$WhisperModel = "large-v3-turbo"

if (Test-Path $WhisperRepo) {
    Write-Host "Downloading whisper model '$WhisperModel' into $WhisperRepo\models ..."
    Push-Location $WhisperRepo
    & .\models\download-ggml-model.cmd $WhisperModel
    Pop-Location
} else {
    Write-Warning "whisper.cpp not found at $WhisperRepo. Clone & build it first (see README), then re-run."
}

# --- Qwen3.5-9B GGUF for llama.cpp ---
# Set QWEN_GGUF_REPO / QWEN_GGUF_FILE to a quantization you have verified loads
# in your llama.cpp build (Q4_K_M recommended). Requires: pip install huggingface_hub
$ModelsDir = "data\models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$QwenRepo = $env:QWEN_GGUF_REPO   # e.g. "<org>/Qwen3.5-9B-GGUF"
$QwenFile = $env:QWEN_GGUF_FILE   # e.g. "Qwen3.5-9B-Q4_K_M.gguf"

if ($QwenRepo -and $QwenFile) {
    Write-Host "Downloading $QwenFile from $QwenRepo ..."
    huggingface-cli download $QwenRepo $QwenFile --local-dir $ModelsDir
} else {
    Write-Warning "Set QWEN_GGUF_REPO and QWEN_GGUF_FILE env vars to download the Qwen3.5-9B GGUF, then re-run."
    Write-Host "Browse working GGUF quantizations from the model card: https://huggingface.co/Qwen/Qwen3.5-9B"
}

# --- Qwen3-TTS voices ---
Write-Host "Qwen3-TTS downloads its weights automatically on first run (via the 'qwen-tts' pip package)."
Write-Host "Done."
