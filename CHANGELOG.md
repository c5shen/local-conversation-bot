# Changelog

## 2026-07-24

### Added
- **Switchable TTS engines** with **Kokoro** as the new default (small/fast) alongside **Qwen3-TTS** (higher quality, heavier). `server/tts.py` is now a pluggable multi-engine facade that loads one engine at a time and unloads the previous one to free VRAM. The engine can be changed live from the UI; the interface locks until the new model is deployed (`set_engine` → `engine_ready`/`engine_error`). New `TTS_ENGINE` setting.
- **Tavily web tools** replacing the DuckDuckGo MCP search: new `server/tools_tavily.py` registers `tavily_search` and `tavily_extract` as Qwen-Agent tools. Configured via `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH`, `TAVILY_MAX_RESULTS`, `TAVILY_INCLUDE_ANSWER`.
- **Session log**: conversations are persisted as JSON under `./.cache/` (new `server/sessions.py`), named by a timestamp id. A new sidebar lists saved sessions with **+ New** and click-to-switch; sessions are stored/updated only when they have content. Refreshing the page starts a fresh session. New endpoints: `GET /api/sessions`, `GET /api/sessions/{id}`, `DELETE /api/sessions/{id}`, plus `new_session`/`load_session` WebSocket messages.

### Changed
- **Language steering**: the system prompt now names both the input ("I speak…") and output ("Reply in…") languages explicitly, and a short reply-language nudge is appended to the latest user message sent to the LLM (Qwen otherwise mirrors the spoken language). The stored transcript stays clean.
- The `config` WebSocket handshake now advertises available engines, the active engine, and engine-specific reply languages/voices (STT languages remain engine-agnostic); `/api/tts` is engine-aware.
- Dependencies: added `kokoro`, `misaki[en,ja,zh]`, `tavily-python`, and pinned `torchaudio` in lockstep with `torch` (avoids a C++ ABI mismatch on import). Dropped the DuckDuckGo MCP dependency (`qwen-agent[mcp]` → `qwen-agent`).
- README and `.env.example` updated for the new TTS engines and Tavily search; `.gitignore` now excludes `.cache/`.

### Fixed
- Chat log auto-scroll now reliably reaches the bottom: `scrollLog()` defers measurement to the next animation frame, and `addMessage()` re-scrolls after its content is inserted (previously it measured a stale height / scrolled while the bubble was empty).
