"""Per-connection orchestration: STT -> Qwen-Agent -> sentence-streamed TTS."""
from __future__ import annotations

import asyncio

from . import agent, metrics, stt, tts
from .audio import SentenceAccumulator, pcm16_to_wav
from .config import LANGUAGES, language_key_from_whisper, load_system_prompt, settings


class Session:
    """Holds conversation history + language choices and drives one websocket client."""

    def __init__(self, websocket) -> None:
        self.ws = websocket
        self.history: list[dict] = []
        self.stt_language: str | None = None  # whisper code, or None for auto-detect
        # When True, the reply language follows the detected spoken language.
        # When False, the user's explicit "Reply in" choice is authoritative.
        self.match_speech: bool = True
        self.set_response_language(settings.default_language)
        self.set_stt_language(settings.default_stt)

    def set_stt_language(self, key: str) -> None:
        """key is a LANGUAGES key or 'auto'."""
        lang = LANGUAGES.get(key)
        self.stt_language = lang["stt"] if lang else None

    def set_response_language(self, key: str, voice: str | None = None) -> None:
        # Fall back to a language the active TTS engine can actually speak (engines
        # differ: e.g. Kokoro has no Korean/German/Russian).
        if key not in LANGUAGES or not tts.supports(key):
            supported = tts.tts_languages()
            key = (
                settings.default_language
                if tts.supports(settings.default_language)
                else (supported[0]["key"] if supported else settings.default_language)
            )
        lang = LANGUAGES.get(key) or LANGUAGES[settings.default_language]
        self.response_key = key
        self.lang_name = lang["name"]
        self.filler_text = lang["filler"]
        voices = tts.voices_for(key)
        self.tts_voice = voice if (voice and voice in voices) else (voices[0] if voices else "")

    def configure(self, msg: dict) -> None:
        """Apply a {type:'set_language', stt, response, voice} control message."""
        if "stt" in msg:
            self.set_stt_language(msg["stt"])
        if "response" in msg:
            if msg["response"] == "match":
                self.match_speech = True  # follow the spoken language
            else:
                self.match_speech = False  # explicit choice wins
                self.set_response_language(msg["response"], msg.get("voice"))
        elif "voice" in msg:
            self.set_response_language(self.response_key, msg["voice"])

    def _system_message(self) -> str:
        lang = LANGUAGES[self.response_key]
        # A native-language directive reliably forces the reply language; Qwen
        # otherwise tends to mirror the language the user spoke.
        instr = (
            f"\n\nIMPORTANT: Write your entire reply in {self.lang_name} only. "
            f"{lang.get('reply_directive', '')} "
            "Never use any other language, even if the user speaks one."
        )
        # Tutor framing only when the user deliberately practices a different language.
        if not self.match_speech and self.stt_language and self.stt_language != lang["stt"]:
            instr += (
                f" Act as a friendly {self.lang_name} tutor and gently correct the "
                "user's mistakes when it helps them learn."
            )
        return load_system_prompt() + instr

    async def _send_audio(self, pcm: bytes) -> None:
        if pcm:
            await self.ws.send_bytes(pcm)

    async def _speak(self, sentence: str) -> None:
        # Carry the reply language/voice so the client can replay it later.
        await self.ws.send_json(
            {"type": "assistant", "text": sentence,
             "language": self.response_key, "voice": self.tts_voice}
        )
        pcm = await asyncio.to_thread(tts.synth_pcm, sentence, self.response_key, self.tts_voice)
        await self._send_audio(pcm)

    async def handle_utterance(self, pcm16: bytes) -> None:
        """Transcribe one captured utterance, then reason and speak the reply."""
        transcript, detected = await stt.transcribe(
            pcm16_to_wav(pcm16, settings.stt_sample_rate), self.stt_language
        )
        if not transcript:
            await self.ws.send_json({"type": "status", "text": "No speech detected."})
            return

        # "Match my speech" mode: make the whole conversation follow the spoken
        # language, but only if the active TTS engine can actually voice it.
        if self.match_speech:
            key = language_key_from_whisper(detected)
            if key and key != self.response_key and tts.supports(key):
                self.set_response_language(key)
                await self.ws.send_json(
                    {"type": "language_update", "response": key, "voice": self.tts_voice}
                )

        # Tag the user message with the spoken language/voice so the UI can
        # replay it (synthesize the user's own text) in a matching voice.
        spoken_key = language_key_from_whisper(detected) or self.response_key
        if not tts.supports(spoken_key):
            spoken_key = self.response_key
        spoken_voice = tts.default_voice(spoken_key) or self.tts_voice
        await self.ws.send_json(
            {"type": "user", "text": transcript,
             "language": spoken_key, "voice": spoken_voice}
        )
        self.history.append({"role": "user", "content": transcript})

        accumulator = SentenceAccumulator()
        final_answer = ""

        async for event, payload in agent.stream_reply(self.history, self._system_message()):
            if event == "tool_call":
                await self.ws.send_json({"type": "tool", "name": payload})
                await self._send_audio(
                    tts.filler_pcm(self.response_key, self.tts_voice, self.filler_text)
                )
            elif event == "tool_result":
                await self.ws.send_json({"type": "tool_result", **payload})
            elif event == "answer":
                for sentence in accumulator.push(payload):
                    await self._speak(sentence)
            elif event == "final":
                final_answer = payload
            elif event == "error":
                await self.ws.send_json({"type": "error", "text": payload})

        remainder = accumulator.flush()
        if remainder:
            await self._speak(remainder)

        self.history.append({"role": "assistant", "content": final_answer})
        await self._update_token_usage()
        await self.ws.send_json({"type": "done"})

    async def _update_token_usage(self) -> None:
        """Estimate current context-token usage (system prompt + conversation)."""
        parts = [self._system_message()]
        parts += [m["content"] for m in self.history if isinstance(m.get("content"), str)]
        count = await metrics.count_tokens("\n".join(parts))
        if count is not None:
            metrics.set_current_tokens(count)
