# Launches the FastAPI orchestrator (UI + WebSocket) on port 8080.
# Run after whisper-server (8081) and llama-server (8000) are up.
# Requires the Python venv with requirements.txt installed.

$ErrorActionPreference = "Stop"

# Ensure uv/uvx (used to spawn the web-search MCP server) is on PATH.
$UvBin = Join-Path $env:USERPROFILE ".local\bin"
if ((Test-Path $UvBin) -and ($env:Path -notlike "*$UvBin*")) { $env:Path = "$UvBin;$env:Path" }

# Activate venv if present.
if (Test-Path ".venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }

$AppHost = $env:HOST; if (-not $AppHost) { $AppHost = "127.0.0.1" }
$Port = $env:PORT; if (-not $Port) { $Port = "8080" }

python -m uvicorn server.main:app --host $AppHost --port $Port
