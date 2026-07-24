@echo off
REM End-to-end launcher for the Local Voice Chatbot (Windows).
REM Starts: whisper-server (STT, :8081), llama-server (LLM, :8000), orchestrator (:8080).
REM Each runs in its own window. Override paths via environment variables before running.

setlocal
cd /d "%~dp0"

REM --- Auto-install dependencies if anything is missing ---
set "NEED_SETUP="
if not exist ".venv\Scripts\python.exe"                set "NEED_SETUP=1"
if not exist "tools\llama\llama-server.exe"            set "NEED_SETUP=1"
if not exist "tools\whisper\Release\whisper-server.exe" set "NEED_SETUP=1"
if defined NEED_SETUP (
    echo Some dependencies are missing - running setup first...
    powershell -ExecutionPolicy Bypass -File "scripts\setup.ps1"
    if errorlevel 1 ( echo Setup failed. & exit /b 1 )
)

if "%WHISPER_BIN%"==""   set "WHISPER_BIN=tools\whisper\Release\whisper-server.exe"
if "%WHISPER_MODEL%"=="" set "WHISPER_MODEL=tools\whisper\models\ggml-large-v3-turbo.bin"
if "%LLAMA_DIR%"==""     set "LLAMA_DIR=tools\llama"
if "%QWEN_GGUF%"==""     set "QWEN_GGUF=data\models\Qwen3.5-9B-Q4_K_M.gguf"

echo ============================================================
echo  Local Voice Chatbot - starting services
echo  whisper bin : %WHISPER_BIN%
echo  llama.cpp   : %LLAMA_DIR%
echo  GGUF        : %QWEN_GGUF%
echo ============================================================

echo [1/3] whisper-server (STT) on http://127.0.0.1:8081
start "whisper-server" powershell -NoExit -ExecutionPolicy Bypass -Command "$env:WHISPER_BIN='%WHISPER_BIN%'; $env:WHISPER_MODEL='%WHISPER_MODEL%'; .\scripts\run_whisper.ps1"

echo [2/3] llama-server (LLM) on http://127.0.0.1:8000
start "llama-server" powershell -NoExit -ExecutionPolicy Bypass -Command "$env:LLAMA_DIR='%LLAMA_DIR%'; $env:QWEN_GGUF='%QWEN_GGUF%'; .\scripts\run_llm.ps1"

echo Waiting 8s for the model servers to load...
timeout /t 8 /nobreak >nul

echo [3/3] orchestrator (UI + WebSocket) on http://localhost:8080
start "voice-orchestrator" powershell -NoExit -ExecutionPolicy Bypass -File scripts\run_app.ps1

echo.
echo All services launching in separate windows.
echo Open http://localhost:8080 in your browser.
endlocal
