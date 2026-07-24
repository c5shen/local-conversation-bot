# Local Real-Time Voice Chatbot

A fully local, low-latency voice assistant for Windows 10 + NVIDIA RTX 3090 (24 GB). Hold a button, talk, and get a spoken reply with optional live web search - everything runs on your machine.

- **STT**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `whisper-server` (CUDA)
- **LLM**: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) as a Q4_K_M GGUF via [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`
- **Agent + tools**: [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) in-process, with [Tavily](https://tavily.com) web-search and page-extract tools
- **TTS**: switchable live in the UI - [Kokoro](https://github.com/hexgrad/kokoro) `82M` (default, small/fast) or [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) `12Hz-1.7B-CustomVoice` (higher quality, cross-lingual named speakers)
- **Orchestrator**: FastAPI + WebSocket browser UI (hold-to-talk or hands-free voice threshold)
- **Language tutor mode**: pick the STT language, reply language, and voice in the UI (English, Chinese, Japanese, Korean, German, French, Russian, Spanish, Italian, Portuguese)

```
Browser (mic, hold-to-talk) --PCM16 16k--> FastAPI orchestrator
                                            |-- whisper-server (/inference) -> transcript
                                            |-- Qwen-Agent -> llama-server (OpenAI API)
                                            |        \--Tavily API--> web search / extract
                                            |-- Qwen3-TTS (sentence streaming)
Browser (speaker)           <--PCM16 24k--  /
```

## Prerequisites

- Windows 10 + an NVIDIA GPU (tested: RTX 3090, 24 GB) with a recent driver
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Python/venv manager). After install it lives at `%USERPROFILE%\.local\bin\uv.exe`.
- A **Qwen3.5-9B Q4_K_M GGUF** placed at `data\models\Qwen3.5-9B-Q4_K_M.gguf` (browse quantizations from the [model card](https://huggingface.co/Qwen/Qwen3.5-9B)), or set `QWEN_GGUF_URL` to a direct download link and setup will fetch it.
Everything else (Python deps, PyTorch CUDA, llama.cpp, whisper.cpp, the whisper model) is installed automatically by the setup script. The Qwen3-TTS voice weights download from Hugging Face on first use - no phonemizer packs or espeak-ng needed; the model handles every supported language natively.

## Quick start

1. Put your `Qwen3.5-9B-Q4_K_M.gguf` in `data\models\`.
2. Double-click **`start_server.bat`** (or run it from a terminal).
   - On first run it auto-runs `scripts\setup.ps1` to install all dependencies, then launches the three services in separate windows.
3. Open <http://localhost:8080>, pick a talk mode, speak, and listen. Ask something current (e.g. "What's the latest news about ...") to trigger a web search.

That's it. Subsequent runs skip setup and start immediately.

## Talk modes

The footer has a toggle for how you trigger the mic:

- **Hold to talk** (default) - hold the on-screen button or press and hold the **spacebar** while speaking, then release to send the utterance.
- **Hands-free (threshold)** - press **Start listening** once, then just speak. A live mic-level meter and a draggable **threshold (dBFS)** slider control capture: recording starts automatically when your voice rises past the threshold marker and ends after a short pause, so you never touch the button. Drop the threshold to catch a quieter voice, raise it to ignore background noise. While the assistant is replying, listening pauses until its audio finishes so its own voice isn't picked up.

## Language tutor mode

The web UI has three selectors so you can practice a foreign language:

- **I speak** - the Whisper STT language. Leave it on *Auto-detect* (the default) to transcribe in whatever language you speak, or pin it to one language.
- **Reply in** - controls the language the assistant **writes and speaks** in (it is authoritative: a native-language instruction is added to the prompt so the model never drifts back to your spoken language).
  - *Match my speech* (the default) - the reply follows the language you just spoke: speak Japanese, get a Japanese reply.
  - A specific language - the assistant always answers in that language. Combined with a different **I speak** language this is tutor mode: e.g. speak English, hear Japanese, with gentle corrections.
- **Voice** - the Qwen3-TTS speaker for the reply (auto-selected in *Match my speech* mode). Qwen3-TTS speakers are cross-lingual, so every language exposes the full roster (Ryan, Aiden, Vivian, Serena, Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee); each language defaults to a fitting voice.

Selections apply instantly (no reload) and persist for the session. Each assistant bubble has a **▶ replay** button that re-synthesizes that reply in the language/voice it was spoken in.

**Extra dependencies per language:** none. Qwen3-TTS supports all ten languages (English, Chinese, Japanese, Korean, German, French, Russian, Spanish, Italian, Portuguese) out of the box - no espeak-ng or phonemizer packs. The model weights download from Hugging Face on first use, so the very first turn needs internet.

## What setup installs

`scripts\setup.ps1` (idempotent - safe to re-run) does:

1. Creates a `uv` virtual environment at `.venv` (Python 3.11) and installs `requirements.txt` with the CUDA 12.4 PyTorch wheel (`uv pip install --torch-backend cu124`).
2. Downloads the prebuilt CUDA `llama.cpp` release into `tools\llama\` (pinned to a build that supports the Qwen3.5 `qwen35` architecture).
3. Downloads the prebuilt CUDA `whisper.cpp` release into `tools\whisper\`.
4. Downloads the `ggml-large-v3-turbo` whisper model into `tools\whisper\models\`.
5. Copies `.env.example` to `.env` and verifies the Qwen GGUF is present.

Run it manually any time with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

## Manual run (separate terminals)

`start_server.bat` is the easy path, but you can launch services individually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_whisper.ps1   # STT  :8081
powershell -ExecutionPolicy Bypass -File scripts\run_llm.ps1       # LLM  :8000
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1       # UI   :8080
```

## Linux / WSL2

`start_server.sh` mirrors the Windows flow and auto-runs `scripts/setup.sh`, which builds `llama.cpp` and `whisper.cpp` from source with CUDA (no prebuilt CUDA Linux binaries are published) and sets up the venv. Requires `git`, `cmake`, a C++ compiler, and the CUDA toolkit.

```bash
./start_server.sh
```

## Configuration

Settings live in `server/config.py` and can be overridden via `.env` (copied from `.env.example`): service URLs, sampling, TTS engine/voice/device, and web search. Search uses [Tavily](https://tavily.com) - set `TAVILY_API_KEY` and toggle it with `ENABLE_SEARCH` (tune `TAVILY_SEARCH_DEPTH`, `TAVILY_MAX_RESULTS`, `TAVILY_INCLUDE_ANSWER`).

## Context window & VRAM (24 GB RTX 3090)

The UI shows a live **GPU** bar (whole-device memory + utilization via NVML) and a
**Context** bar (`current / max` tokens, polled from `/api/metrics`).

Qwen3.5-9B (Q4_K_M) facts measured from the GGUF on this machine:

- Weights on disk/VRAM: **5.3 GB**; +CUDA/compute buffers ≈ **~6.3 GB** base (no KV).
- Architecture: 32 layers, 4 KV heads, head_dim 256 → **f16 KV cache = 128 KiB/token**.
- Trained context length: **262144 (256K)**.

So the f16 KV cache cost is linear:

| Context (`-c`) | f16 KV cache | llama total (~6.3 GB + KV) |
| --- | --- | --- |
| 8192 | 1.0 GiB | ~7.3 GB |
| 16384 | 2.0 GiB | ~8.3 GB |
| 32768 (default) | 4.0 GiB | ~10.3 GB |
| 65536 | 8.0 GiB | ~14.3 GB |
| 131072 | 16.0 GiB | ~22.3 GB |
| 262144 | 32 GiB | won't fit |

Co-resident budget (also running whisper ~1.6 GB + Qwen3-TTS 1.7B ~4-4.5 GB in
bf16 + ~1.5 GB OS/CUDA overhead) leaves roughly **10 GB for KV → ~80K tokens (f16)
max** with all three models loaded. (Qwen3-TTS is heavier than the old Kokoro TTS,
which cost ~1 GB - swap `TTS_MODEL` to `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, ~2 GB,
to reclaim headroom for a larger context.)

**Recommendations**

- **Default is now `-c 32768`** (4 GiB KV) - a safe, comfortable balance on the 3090
  (~18 GB total with the 1.7B TTS model loaded).
- Push to **`-c 65536`** (8 GiB KV) and it still fits with whisper + Qwen3-TTS (~22 GB
  total) - tight but workable on 24 GB; use the 0.6B TTS model for more margin.
- For more, halve KV memory with quantized cache (requires `-fa`):
  `-ctk q8_0 -ctv q8_0` doubles the reachable context (e.g. **131072 at q8 ≈ 8 GiB**).
- Practical ceiling: **~80K (f16) co-resident**, ~130K llama-only (f16), or near the
  full 256K with `q8_0` KV when llama runs alone.

Override at launch: set `CTX` before `run_llm.ps1` / `start_server.*`
(e.g. `$env:CTX="65536"`). Note the KV cache is preallocated for the full `-c`, so
larger windows reserve that VRAM even for short chats.

## Verified working

- llama.cpp build `b9430` loads the `qwen35` GGUF on the 3090 at ~104 tok/s; thinking-disabled replies return in ~0.3 s.
- Full pipeline test: a synthesized question went mic -> whisper -> Qwen-Agent -> Qwen3-TTS and produced a correct spoken answer.

## Troubleshooting

- **No audio / mic blocked**: the page must be `http://localhost` (a secure context). Allow mic access when prompted.
- **`uv` not found by a script**: open a new terminal so `%USERPROFILE%\.local\bin` is on `PATH`, or pass the full path.
- **Search never triggers**: confirm `TAVILY_API_KEY` is set in `.env` (search is skipped with a log warning when it's empty); set `ENABLE_SEARCH=false` to disable tools.
- **Thinking text spoken**: the orchestrator strips `<think>...</think>`; `enable_thinking=false` is also sent to the LLM.
- **GGUF won't load (`unknown architecture: qwen35`)**: bump `$LLAMA_TAG` in `scripts\setup.ps1` to a newer llama.cpp release, or use the WSL2 + vLLM fallback (`scripts/run_llm_wsl_vllm.sh`).
- **Latency**: lower the whisper model, keep `MAX_TOKENS` modest, and keep utterances short.

## Project layout

```
server/   FastAPI orchestrator (config, stt, agent, tts, audio, pipeline, main)
web/      Browser UI (index.html, app.js, worklet.js, styles.css)
scripts/  setup.ps1/.sh, run_*.ps1 launchers, WSL2 vLLM fallback
tools/    Auto-installed llama.cpp + whisper.cpp binaries/models (gitignored)
data/     Qwen GGUF lives in data/models/ (gitignored)
```
