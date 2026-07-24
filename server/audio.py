"""Audio byte helpers and a streaming sentence segmenter."""
from __future__ import annotations

import io
import re
import wave

import numpy as np

# Matches a sentence terminator: CJK full-width punctuation (。！？) breaks immediately
# since CJK text has no spaces; ASCII terminators break on trailing space / end-of-buffer.
_BOUNDARY = re.compile(r"(?:[\u3002\uff01\uff1f]+|[.!?\u2026]+[\"')\]]*(?:\s|$)|\n+)")


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian mono PCM16 bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 waveform in [-1, 1] to little-endian PCM16 bytes."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


class SentenceAccumulator:
    """Accumulates streamed text deltas and yields complete sentences.

    Short fragments are held back until a sentence terminator arrives so that
    the TTS model receives natural, well-punctuated chunks.
    """

    def __init__(self, min_chars: int = 2) -> None:
        self._buf = ""
        self._min_chars = min_chars

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            match = _BOUNDARY.search(self._buf)
            if not match:
                break
            end = match.end()
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if len(sentence) >= self._min_chars:
                out.append(sentence)
        return out

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        return rest
