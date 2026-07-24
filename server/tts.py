"""Pluggable text-to-speech producing PCM16 bytes.

Two engines are available and can be swapped at runtime (see load_engine):

* "kokoro" - Kokoro-82M, small and fast; the default. Supports a subset of the
  configured languages (English, Spanish, French, Italian, Portuguese, Chinese,
  Japanese).
* "qwen"   - Qwen3-TTS CustomVoice, higher quality but much heavier/slower.
  Cross-lingual speakers cover every configured language.

Only one engine is resident at a time; switching unloads the previous model and
frees its VRAM. All calls that touch a model are serialised by a lock so a synth
request and an engine switch can't race (this is a single-user local app).
"""
from __future__ import annotations

import threading

import numpy as np

from .audio import float_to_pcm16
from .config import LANGUAGES, settings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_device(preferred: str) -> str:
    try:
        import torch

        if preferred.startswith("cuda") and torch.cuda.is_available():
            return preferred
    except Exception:  # noqa: BLE001 - torch missing or no CUDA -> CPU
        pass
    return "cpu"


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _to_float_wave(wavs) -> np.ndarray:
    """Normalize engine output (tensor/array, possibly batched) to 1-D float32."""
    if hasattr(wavs, "detach"):
        wavs = wavs.detach().cpu().numpy()
    audio = np.asarray(wavs, dtype=np.float32)
    while audio.ndim > 1:
        audio = audio[0]
    return audio


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-resample a 1-D waveform; no-op when the rates already match."""
    if src_sr == dst_sr or audio.size == 0:
        return audio
    dst_len = int(round(audio.shape[0] * dst_sr / float(src_sr)))
    if dst_len <= 0:
        return audio
    src_idx = np.linspace(0.0, audio.shape[0] - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(audio.shape[0]), audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


class _Engine:
    """Base class. Subclasses fill in `id`, `name`, `langs` and the model hooks.

    `langs` maps a LANGUAGES key -> {"voices": [...], ...engine-specific...}; the
    first voice is that language's default.
    """

    id: str = ""
    name: str = ""
    sample_rate: int = 24000
    langs: dict[str, dict] = {}

    # --- lifecycle (override) ---
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:  # noqa: D401 - optional override
        pass

    def _synth_float(self, text: str, lang_key: str, voice: str):
        """Return (waveform, sample_rate). Called with a validated lang_key/voice."""
        raise NotImplementedError

    # --- shared ---
    def voices_for(self, lang_key: str) -> list[str]:
        entry = self.langs.get(lang_key)
        return list(entry["voices"]) if entry else []

    def default_voice(self, lang_key: str) -> str:
        voices = self.voices_for(lang_key)
        return voices[0] if voices else ""

    def resolve(self, lang_key: str | None, voice: str | None) -> tuple[str, str]:
        """Coerce (lang_key, voice) to a pair this engine actually supports."""
        if lang_key not in self.langs:
            lang_key = (
                settings.default_language
                if settings.default_language in self.langs
                else next(iter(self.langs))
            )
        voices = self.langs[lang_key]["voices"]
        if voice not in voices:
            voice = voices[0]
        return lang_key, voice

    def synth_pcm(self, text: str, lang_key: str, voice: str) -> bytes:
        lang_key, voice = self.resolve(lang_key, voice)
        audio, sr = self._synth_float(text, lang_key, voice)
        audio = _resample(_to_float_wave(audio), int(sr), self.sample_rate)
        return float_to_pcm16(audio)


class QwenEngine(_Engine):
    """Qwen3-TTS CustomVoice: cross-lingual speakers, every configured language."""

    id = "qwen"
    name = "Qwen3-TTS"

    def __init__(self) -> None:
        self.sample_rate = settings.tts_sample_rate
        self.langs = {
            k: {"qwen": v["qwen"], "voices": list(v["voices"])}
            for k, v in LANGUAGES.items()
        }
        self._model = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        device = _resolve_device(settings.tts_device)
        kwargs = {
            "device_map": device,
            "dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        }
        if settings.tts_attn:
            kwargs["attn_implementation"] = settings.tts_attn
        self._model = Qwen3TTSModel.from_pretrained(settings.tts_model, **kwargs)

    def unload(self) -> None:
        self._model = None
        _free_cuda()

    def _synth_float(self, text: str, lang_key: str, voice: str):
        wavs, sr = self._model.generate_custom_voice(
            text=text, language=self.langs[lang_key]["qwen"], speaker=voice
        )
        return wavs, sr


# Kokoro lang_code + a curated voice roster per supported language. Kokoro cannot
# do Korean/German/Russian, so those keys are simply absent here (the UI only
# offers the languages the active engine reports).
_KOKORO_LANGS: dict[str, dict] = {
    "en": {"code": "a", "voices": ["af_heart", "af_bella", "af_nicole",
                                    "am_michael", "am_adam", "am_fenrir", "am_puck"]},
    "es": {"code": "e", "voices": ["ef_dora", "em_alex"]},
    "fr": {"code": "f", "voices": ["ff_siwis"]},
    "it": {"code": "i", "voices": ["if_sara", "im_nicola"]},
    "pt": {"code": "p", "voices": ["pf_dora", "pm_alex"]},
    "zh": {"code": "z", "voices": ["zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi"]},
    "ja": {"code": "j", "voices": ["jf_alpha", "jm_kumo"]},
}
_KOKORO_REPO = "hexgrad/Kokoro-82M"


class KokoroEngine(_Engine):
    """Kokoro-82M: lightweight, fast. One shared model, one pipeline per language."""

    id = "kokoro"
    name = "Kokoro"
    sample_rate = 24000  # Kokoro emits 24 kHz mono

    def __init__(self) -> None:
        self.langs = {
            k: {"code": v["code"], "voices": list(v["voices"])}
            for k, v in _KOKORO_LANGS.items()
        }
        self._model = None
        self._pipelines: dict[str, object] = {}

    def load(self) -> None:
        from kokoro import KModel

        device = _resolve_device(settings.tts_device)
        self._model = KModel(repo_id=_KOKORO_REPO).to(device).eval()
        self._pipelines = {}
        # Warm the default-language pipeline so the first turn is fast.
        self._pipeline_for(settings.default_language if settings.default_language in self.langs else "en")

    def unload(self) -> None:
        self._model = None
        self._pipelines = {}
        _free_cuda()

    def _pipeline_for(self, lang_key: str):
        code = self.langs[lang_key]["code"]
        pipe = self._pipelines.get(code)
        if pipe is None:
            from kokoro import KPipeline

            pipe = KPipeline(lang_code=code, repo_id=_KOKORO_REPO, model=self._model)
            self._pipelines[code] = pipe
        return pipe

    def _synth_float(self, text: str, lang_key: str, voice: str):
        pipe = self._pipeline_for(lang_key)
        chunks = [r.audio for r in pipe(text, voice=voice, speed=1) if r.audio is not None]
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        audio = np.concatenate([_to_float_wave(c) for c in chunks])
        return audio, self.sample_rate


# ---------------------------------------------------------------------------
# Registry + active-engine state
# ---------------------------------------------------------------------------

_ENGINES: dict[str, _Engine] = {e.id: e for e in (KokoroEngine(), QwenEngine())}
_ENGINE_ORDER = ["kokoro", "qwen"]

_lock = threading.RLock()
_active_id: str | None = None
_filler_cache: dict[tuple[str, str, str], bytes] = {}


def _active() -> _Engine:
    if _active_id is None:
        # Lazily load the configured default engine on first use.
        load_engine(settings.tts_engine if settings.tts_engine in _ENGINES else "kokoro")
    return _ENGINES[_active_id]


def list_engines() -> list[dict]:
    """Metadata for the engine picker, in display order."""
    return [{"id": e.id, "name": e.name} for e in
            (_ENGINES[i] for i in _ENGINE_ORDER if i in _ENGINES)]


def active_engine_id() -> str:
    return _active().id


def sample_rate() -> int:
    return _active().sample_rate


def tts_languages() -> list[dict]:
    """[{key, name, voices}] for the active engine, in the configured order."""
    eng = _active()
    out = []
    for key, meta in LANGUAGES.items():
        if key in eng.langs:
            out.append({"key": key, "name": meta["name"], "voices": eng.voices_for(key)})
    return out


def voices_for(lang_key: str) -> list[str]:
    return _active().voices_for(lang_key)


def default_voice(lang_key: str) -> str:
    return _active().default_voice(lang_key)


def supports(lang_key: str) -> bool:
    return lang_key in _active().langs


def load_engine(engine_id: str) -> None:
    """Deploy `engine_id`, unloading whatever was active. Blocking; raises on failure.

    Safe to call with the already-active id (no-op). Serialised against synth calls.
    """
    global _active_id
    if engine_id not in _ENGINES:
        raise ValueError(f"Unknown TTS engine: {engine_id!r}")
    with _lock:
        if _active_id == engine_id and _ENGINES[engine_id]._model is not None:
            return
        # Unload the previous engine first so both models never sit in VRAM.
        if _active_id is not None and _active_id != engine_id:
            try:
                _ENGINES[_active_id].unload()
            except Exception:  # noqa: BLE001 - unload failure shouldn't block switching
                pass
        _ENGINES[engine_id].load()
        _active_id = engine_id
        _filler_cache.clear()


def synth_pcm(text: str, language: str | None = None, voice: str | None = None) -> bytes:
    """Synthesize `text` to little-endian mono PCM16 at the active engine's rate.

    `language` is a LANGUAGES key (e.g. "en"); the active engine maps it to its own
    language id. Blocking (runs torch); call via asyncio.to_thread from async code.
    """
    text = text.strip()
    if not text:
        return b""
    language = language or settings.default_language
    with _lock:
        return _active().synth_pcm(text, language, voice)


def warmup() -> None:
    """Load the default engine and pre-render its default filler clip."""
    load_engine(settings.tts_engine if settings.tts_engine in _ENGINES else "kokoro")
    synth_pcm("Hello.", settings.default_language)
    filler_pcm(settings.default_language)


def filler_pcm(language: str | None = None, voice: str | None = None,
               text: str | None = None) -> bytes:
    """Return (and cache) the synthesized 'let me look that up' clip for a voice.

    Cached per (engine, language, voice); the cache is cleared on engine switch.
    """
    language = language or settings.default_language
    eng = _active()
    lang_key, voice = eng.resolve(language, voice)
    key = (eng.id, lang_key, voice)
    pcm = _filler_cache.get(key)
    if pcm is None:
        pcm = synth_pcm(text or settings.filler_text, lang_key, voice)
        _filler_cache[key] = pcm
    return pcm
