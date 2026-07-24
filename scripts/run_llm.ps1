# Launches llama.cpp's OpenAI-compatible server (LLM) on port 8000.
# Use a prebuilt CUDA llama.cpp release or build with -DGGML_CUDA=1 (see README).
#
# NOTE: Qwen-Agent injects tool definitions as text and parses tool calls client-side,
# so we do NOT pass --jinja (which has known Qwen3 server-side tool-parsing bugs).
# If you find Qwen3.5 "thinking" is not suppressed, add --jinja so the model's chat
# template (and enable_thinking=false) is applied; the orchestrator strips <think> blocks
# defensively in any case.

$LlamaDir = $env:LLAMA_DIR; if (-not $LlamaDir) { $LlamaDir = "tools\llama" }
$Model = $env:QWEN_GGUF; if (-not $Model) { $Model = "data\models\Qwen3.5-9B-Q4_K_M.gguf" }
# Context window. 32768 ~= 4 GiB f16 KV cache; fits comfortably on a 24 GB 3090
# alongside whisper + Qwen3-TTS (see README "Context window & VRAM"). Override via CTX.
$Ctx = $env:CTX; if (-not $Ctx) { $Ctx = "32768" }
$Exe = "$LlamaDir\llama-server.exe"

if (-not (Test-Path $Exe)) { throw "llama-server.exe not found at $Exe. Install/build llama.cpp (see README)." }
if (-not (Test-Path $Model)) { throw "GGUF not found at $Model. Run scripts\setup_models.ps1 first." }

& $Exe -m $Model -ngl 99 -c $Ctx -fa on --host 127.0.0.1 --port 8000
