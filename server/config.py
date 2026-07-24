"""Centralized settings, loaded from environment / .env (see .env.example)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The base system prompt lives in prompts/sys_prompt.md so it can be edited without
# touching code. It is read at use time (see load_system_prompt) and language-specific
# instructions are appended per session in pipeline.py.
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sys_prompt.md"
_PROMPT_FALLBACK = (
    "You are a friendly voice assistant. Keep your answers short, spoken, and "
    "conversational; avoid lists, markdown, and code unless explicitly asked."
)


def load_system_prompt() -> str:
    """Read the base system prompt from prompts/sys_prompt.md (fresh each call)."""
    try:
        text = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or _PROMPT_FALLBACK
    except OSError:
        return _PROMPT_FALLBACK

# Qwen3-TTS CustomVoice speakers. Unlike Kokoro voices, these are cross-lingual:
# any speaker can render any of the model's supported languages (the speaker only
# sets the voice timbre), so we surface the whole roster for every language and just
# default to a fitting voice. See https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu",
                 "Dylan", "Eric", "Ono_Anna", "Sohee"]


def _voices(*preferred: str) -> list[str]:
    """Full speaker roster with `preferred` pulled to the front (first = default)."""
    rest = [s for s in QWEN_SPEAKERS if s not in preferred]
    return list(preferred) + rest


# Supported conversation languages. Each entry maps a UI key to:
#   name   - human-readable label shown in the UI
#   stt    - whisper language code ("auto" lets whisper detect)
#   qwen   - Qwen3-TTS language name passed to generate_custom_voice(language=...)
#   voices - selectable Qwen3-TTS speakers for that language (first is the default)
#   filler - short "let me look that up" clip spoken while a tool runs
#   reply_directive - a native-language instruction that reliably forces the LLM to
#                     answer in this language (Qwen tends to mirror the user otherwise)
#
# The set matches the languages Qwen3-TTS supports. No extra phonemizer packs or
# espeak-ng are needed - the model handles all of these natively.
LANGUAGES: dict[str, dict] = {
    "en": {"name": "English", "stt": "en", "qwen": "English",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Let me look that up.",
           "reply_directive": "Always answer in English."},
    "zh": {"name": "Chinese (中文)", "stt": "zh", "qwen": "Chinese",
           "voices": _voices("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"),
           "filler": "让我查一下。",
           "reply_directive": "请务必只用中文回答。"},
    "ja": {"name": "Japanese (日本語)", "stt": "ja", "qwen": "Japanese",
           "voices": _voices("Ono_Anna"),
           "filler": "ちょっと調べますね。",
           "reply_directive": "必ず日本語だけで答えてください。"},
    "ko": {"name": "Korean (한국어)", "stt": "ko", "qwen": "Korean",
           "voices": _voices("Sohee"),
           "filler": "잠깐 찾아볼게요.",
           "reply_directive": "반드시 한국어로만 대답하세요."},
    "de": {"name": "German (Deutsch)", "stt": "de", "qwen": "German",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Lass mich das kurz nachschauen.",
           "reply_directive": "Antworte ausschließlich auf Deutsch."},
    "fr": {"name": "French (Français)", "stt": "fr", "qwen": "French",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Laissez-moi vérifier.",
           "reply_directive": "Réponds uniquement en français."},
    "ru": {"name": "Russian (Русский)", "stt": "ru", "qwen": "Russian",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Дайте я посмотрю.",
           "reply_directive": "Отвечай только на русском языке."},
    "es": {"name": "Spanish (Español)", "stt": "es", "qwen": "Spanish",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Déjame buscar eso.",
           "reply_directive": "Responde únicamente en español."},
    "it": {"name": "Italian (Italiano)", "stt": "it", "qwen": "Italian",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Fammi controllare.",
           "reply_directive": "Rispondi esclusivamente in italiano."},
    "pt": {"name": "Portuguese (Português)", "stt": "pt", "qwen": "Portuguese",
           "voices": _voices("Ryan", "Aiden"),
           "filler": "Deixa eu verificar.",
           "reply_directive": "Responda apenas em português."},
}


# Map whisper's detected language (code like "ja" or name like "japanese") to a key.
_WHISPER_ALIASES = {
    "en": "en", "english": "en",
    "zh": "zh", "chinese": "zh", "mandarin": "zh",
    "ja": "ja", "japanese": "ja",
    "ko": "ko", "korean": "ko",
    "de": "de", "german": "de",
    "fr": "fr", "french": "fr",
    "ru": "ru", "russian": "ru",
    "es": "es", "spanish": "es",
    "it": "it", "italian": "it",
    "pt": "pt", "portuguese": "pt",
}


def language_key_from_whisper(value: str | None) -> str | None:
    """Resolve a whisper language code/name to a supported LANGUAGES key, or None."""
    if not value:
        return None
    return _WHISPER_ALIASES.get(value.strip().lower())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Orchestrator HTTP server
    host: str = "127.0.0.1"
    port: int = 8080

    # Speech-to-text (whisper.cpp whisper-server)
    whisper_url: str = "http://127.0.0.1:8081"
    stt_sample_rate: int = 16000

    # LLM (llama.cpp llama-server, OpenAI-compatible API)
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "Qwen/Qwen3.5-9B"
    llm_model_type: str = "oai"
    llm_api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 32768
    # Max prompt tokens Qwen-Agent keeps before truncating history/tool output.
    # 0 = auto: derive from the llama-server context window (n_ctx - max_tokens - margin)
    # so the agent uses the full context instead of a small fixed budget.
    max_input_tokens: int = 0

    # Conversation language (key into LANGUAGES). Drives default STT + TTS + reply language.
    default_language: str = "en"
    # Default STT language: a LANGUAGES key, or "auto" to let whisper detect.
    default_stt: str = "auto"

    # Text-to-speech. Two engines are available and can be switched live from the
    # UI (see server/tts.py): "kokoro" (small/fast, the default) and "qwen"
    # (Qwen3-TTS CustomVoice, higher quality but heavier/slower).
    tts_engine: str = "kokoro"  # "kokoro" | "qwen" - the engine loaded at startup
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"  # Qwen engine weights
    tts_language: str = "English"  # a Qwen3-TTS language name (LANGUAGES[*]["qwen"])
    tts_voice: str = "Ryan"  # a Qwen3-TTS CustomVoice speaker (see QWEN_SPEAKERS)
    tts_sample_rate: int = 24000  # both engines emit 24 kHz mono
    tts_device: str = "cuda"
    # Attention impl: "sdpa" is built into torch>=2 and needs nothing extra;
    # "flash_attention_2" is faster/leaner but requires the flash-attn package.
    tts_attn: str = "sdpa"
    filler_text: str = "Let me look that up."

    # Agent / tools (Tavily web search + extract; https://tavily.com)
    enable_search: bool = True
    tavily_api_key: str = ""  # from TAVILY_API_KEY in .env
    # "basic" is fast/cheap; "advanced" digs deeper (higher latency + cost).
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5
    # Ask Tavily for a short LLM-generated answer alongside the raw results.
    tavily_include_answer: bool = True


settings = Settings()
