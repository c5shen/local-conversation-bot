"""Speech-to-text client for the whisper.cpp whisper-server HTTP API."""
from __future__ import annotations

import httpx

from .config import settings


async def transcribe(wav_bytes: bytes, language: str | None = None) -> tuple[str, str | None]:
    """POST a 16kHz mono WAV to whisper-server /inference.

    language: ISO code (e.g. "ja", "es") to force decoding in that language, or
    None/"auto" to let whisper detect it.

    Returns (transcript, detected_language) where detected_language is the whisper
    language code (e.g. "ja") when available, else None.
    """
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    # verbose_json so the response includes the auto-detected language. transcribe
    # (not translate) keeps the text in the spoken language.
    data = {"response_format": "verbose_json", "temperature": "0",
            "language": language or "auto"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{settings.whisper_url}/inference", files=files, data=data)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            return resp.text.strip(), None
        if not isinstance(payload, dict):
            return "", None
        return (payload.get("text") or "").strip(), payload.get("language")
