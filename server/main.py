"""FastAPI entrypoint: serves the browser UI and the /ws audio WebSocket."""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from fastapi import Body, FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import agent, metrics, tts
from .audio import pcm16_to_wav
from .config import LANGUAGES, settings
from .pipeline import Session

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Local Voice Chatbot")


@app.on_event("startup")
async def _startup() -> None:
    # Load the TTS model and build the agent (spawns the MCP search server) up front.
    await asyncio.to_thread(tts.warmup)
    await agent.init()


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/metrics")
async def api_metrics() -> dict:
    """GPU usage + current context-token usage for the status display."""
    return await metrics.snapshot()


@app.post("/api/tts")
async def api_tts(payload: dict = Body(...)) -> Response:
    """Synthesize text to a WAV clip (used by the per-message replay button)."""
    text = (payload.get("text") or "").strip()
    if not text:
        return Response(status_code=204)
    # The active engine resolves language/voice (and falls back if unsupported).
    pcm = await asyncio.to_thread(
        tts.synth_pcm, text, payload.get("language"), payload.get("voice")
    )
    return Response(content=pcm16_to_wav(pcm, tts.sample_rate()), media_type="audio/wav")


def _config_message() -> dict:
    """The initial handshake: engines, languages (per active engine) and defaults."""
    return {
        "type": "config",
        "tts_sample_rate": tts.sample_rate(),
        "engines": tts.list_engines(),
        "engine": tts.active_engine_id(),
        # STT is engine-agnostic (whisper), so offer every configured language.
        "stt_languages": [{"key": k, "name": v["name"]} for k, v in LANGUAGES.items()],
        # Reply languages + voices depend on the active TTS engine.
        "tts_languages": tts.tts_languages(),
        "default_stt": settings.default_stt,
        "default_response": "match",
    }


async def _switch_engine(websocket: WebSocket, session: Session, engine_id: str) -> None:
    """Deploy a different TTS engine, then tell the client to unlock and refresh.

    The client keeps its controls locked from the moment it sends `set_engine`
    until it receives `engine_ready` (or `engine_error`).
    """
    try:
        await asyncio.to_thread(tts.load_engine, engine_id)
    except Exception as exc:  # noqa: BLE001 - report and leave the old engine active
        await websocket.send_json(
            {"type": "engine_error", "engine": engine_id, "text": str(exc)}
        )
        return
    # Re-resolve the session's language against the new engine's capabilities.
    session.set_response_language(session.response_key, session.tts_voice)
    await websocket.send_json(
        {
            "type": "engine_ready",
            "engine": tts.active_engine_id(),
            "tts_sample_rate": tts.sample_rate(),
            "tts_languages": tts.tts_languages(),
        }
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(_config_message())
    session = Session(websocket)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            audio = message.get("bytes")
            if audio:
                # One binary frame == one complete hold-to-talk utterance.
                # A single failed turn must not tear down the connection.
                try:
                    await session.handle_utterance(audio)
                except WebSocketDisconnect:
                    raise
                except Exception as exc:  # noqa: BLE001 - report and keep the session alive
                    with contextlib.suppress(Exception):
                        await websocket.send_json({"type": "error", "text": str(exc)})
                        await websocket.send_json({"type": "done"})
                continue
            text = message.get("text")
            if text:
                with contextlib.suppress(Exception):
                    data = json.loads(text)
                    kind = data.get("type")
                    if kind == "set_language":
                        session.configure(data)
                    elif kind == "set_engine":
                        await _switch_engine(websocket, session, data.get("engine"))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - report then close cleanly
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "text": str(exc)})
