# Launches the whisper.cpp HTTP server (STT) on port 8081.
# Defaults to the prebuilt CUDA binaries downloaded into tools\whisper.
# Override WHISPER_BIN / WHISPER_MODEL to use a different build or model.

$Exe = $env:WHISPER_BIN
if (-not $Exe) { $Exe = "tools\whisper\Release\whisper-server.exe" }

$Model = $env:WHISPER_MODEL
if (-not $Model) { $Model = "tools\whisper\models\ggml-large-v3-turbo.bin" }

if (-not (Test-Path $Exe))   { throw "whisper-server.exe not found at $Exe (see README / tools\whisper)." }
if (-not (Test-Path $Model)) { throw "whisper model not found at $Model." }

& $Exe -m $Model --host 127.0.0.1 --port 8081 -t 8
