# One-shot, idempotent setup for the Local Voice Chatbot (Windows + NVIDIA CUDA).
# Installs: uv venv + Python deps, prebuilt CUDA llama.cpp, prebuilt CUDA whisper.cpp,
# and the whisper model. Safe to re-run; existing pieces are skipped.
# (Qwen3-TTS handles every language natively - no optional phonemizer packs needed.)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --- Pinned, verified versions (bump as newer builds are validated) ---
$LLAMA_TAG     = "b9430"
$WHISPER_TAG   = "v1.8.5"
$CUDA          = "12.4"
$PYTHON_VER    = "3.11"
$TORCH_BACKEND = "cu124"
$LLAMA_ZIP     = "llama-$LLAMA_TAG-bin-win-cuda-$CUDA-x64.zip"
$CUDART_ZIP    = "cudart-llama-bin-win-cuda-$CUDA-x64.zip"
$WHISPER_ZIP   = "whisper-cublas-$CUDA.0-bin-x64.zip"
$WHISPER_MODEL = "ggml-large-v3-turbo.bin"
$GGUF          = "data\models\Qwen3.5-9B-Q4_K_M.gguf"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "== Local Voice Chatbot setup ==  $Root"

# --- locate uv (PATH or default install location) ---
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $cand = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $cand) { $uv = $cand }
}
if (-not $uv) {
    throw "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/ and re-run."
}
Write-Host "uv: $uv"

function Get-File($url, $out) {
    if (Test-Path $out) { Write-Host "  exists: $out"; return }
    Write-Host "  downloading: $url"
    Invoke-WebRequest -Uri $url -OutFile $out
}

# --- 1. Python venv + dependencies ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/5] Creating virtual environment (Python $PYTHON_VER)..."
    & $uv venv --python $PYTHON_VER .venv
} else {
    Write-Host "[1/5] Virtual environment already exists."
}
Write-Host "[1/5] Installing Python dependencies (torch backend: $TORCH_BACKEND)..."
& $uv pip install --python ".venv\Scripts\python.exe" --torch-backend $TORCH_BACKEND -r requirements.txt

New-Item -ItemType Directory -Force -Path "tools\dl","tools\llama","tools\whisper\models","data\models" | Out-Null

# --- 2. llama.cpp (CUDA) ---
if (-not (Test-Path "tools\llama\llama-server.exe")) {
    Write-Host "[2/5] Installing llama.cpp $LLAMA_TAG (CUDA $CUDA)..."
    $base = "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"
    Get-File "$base/$LLAMA_ZIP"  "tools\dl\llama.zip"
    Get-File "$base/$CUDART_ZIP" "tools\dl\cudart.zip"
    Expand-Archive "tools\dl\llama.zip"  -DestinationPath "tools\llama" -Force
    Expand-Archive "tools\dl\cudart.zip" -DestinationPath "tools\llama" -Force
} else {
    Write-Host "[2/5] llama.cpp already installed."
}

# --- 3. whisper.cpp (CUDA) ---
if (-not (Test-Path "tools\whisper\Release\whisper-server.exe")) {
    Write-Host "[3/5] Installing whisper.cpp $WHISPER_TAG (CUDA $CUDA)..."
    $wbase = "https://github.com/ggml-org/whisper.cpp/releases/download/$WHISPER_TAG"
    Get-File "$wbase/$WHISPER_ZIP" "tools\dl\whisper.zip"
    Expand-Archive "tools\dl\whisper.zip" -DestinationPath "tools\whisper" -Force
} else {
    Write-Host "[3/5] whisper.cpp already installed."
}

# --- 4. whisper model ---
if (-not (Test-Path "tools\whisper\models\$WHISPER_MODEL")) {
    Write-Host "[4/5] Downloading whisper model $WHISPER_MODEL..."
    Get-File "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL" "tools\whisper\models\$WHISPER_MODEL"
} else {
    Write-Host "[4/5] whisper model already present."
}

# --- 5. Qwen GGUF + .env ---
if (-not (Test-Path $GGUF)) {
    if ($env:QWEN_GGUF_URL) {
        Write-Host "[5/5] Downloading Qwen GGUF from QWEN_GGUF_URL..."
        Get-File $env:QWEN_GGUF_URL $GGUF
    } else {
        Write-Warning "[5/5] Qwen GGUF not found at $GGUF."
        Write-Warning "      Place a Qwen3.5-9B Q4_K_M GGUF there, or set QWEN_GGUF_URL to a direct"
        Write-Warning "      download link and re-run. Browse: https://huggingface.co/Qwen/Qwen3.5-9B"
    }
} else {
    Write-Host "[5/5] Qwen GGUF present."
}
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

# --- cleanup ---
Remove-Item -Recurse -Force "tools\dl" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "== Setup complete. Run start_server.bat (or scripts\run_*.ps1) to launch. =="
