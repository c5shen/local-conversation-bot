"""Per-connection orchestration: STT -> Qwen-Agent -> sentence-streamed TTS."""
from __future__ import annotations

import asyncio
from datetime import datetime

from . import agent, metrics, sessions, stt, tts
from .audio import SentenceAccumulator, pcm16_to_wav
from .config import LANGUAGES, language_key_from_whisper, load_system_prompt, settings


class Session:
    """Holds conversation history + language choices and drives one websocket client."""

    def __init__(self, websocket) -> None:
        self.ws = websocket
        self.history: list[dict] = []
        # Rich transcript persisted to the session log (role/text/language/voice).
        self.messages: list[dict] = []
        self.persist_id: str | None = None   # set once the session first gets content
        self.created_at: str | None = None
        self.stt_language: str | None = None  # whisper code, or None for auto-detect
        # When True, the reply language follows the detected spoken language.
        # When False, the user's explicit "Reply in" choice is authoritative.
        self.match_speech: bool = True
        self.set_response_language(settings.default_language)
        self.set_stt_language(settings.default_stt)

    # ---- Session log (persisted under ./.cache) ----
    def reset_session(self) -> None:
        """Start a fresh, empty conversation (the '+ New' action)."""
        self.history = []
        self.messages = []
        self.persist_id = None
        self.created_at = None

    def load_saved(self, session_id: str) -> dict | None:
        """Make a stored session the active one so the conversation can continue."""
        data = sessions.load(session_id)
        if not data:
            return None
        self.messages = list(data.get("messages") or [])
        self.history = [
            {"role": m["role"], "content": m["text"]}
            for m in self.messages
            if m.get("text") and m.get("role") in ("user", "assistant")
        ]
        self.persist_id = data.get("id")
        self.created_at = data.get("created_at")
        return data

    def _record(self, role: str, text: str, language: str, voice: str) -> None:
        if (text or "").strip():
            self.messages.append(
                {"role": role, "text": text, "language": language, "voice": voice}
            )

    def _persist(self) -> dict | None:
        """Write the session file (assigning an id on first content). Blocking IO."""
        if not self.messages:
            return None
        if self.persist_id is None:
            self.persist_id = sessions.new_id()
            self.created_at = datetime.now().isoformat(timespec="seconds")
        return sessions.save(self.persist_id, self.created_at, self.messages)

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

    def _input_language_name(self, spoken_key: str | None) -> str | None:
        """Human name of the language the user is speaking, or None if unknown.

        An explicit "I speak X" choice wins; otherwise fall back to the language
        whisper detected for this utterance (spoken_key).
        """
        if self.stt_language:
            key = language_key_from_whisper(self.stt_language)
            if key:
                return LANGUAGES[key]["name"]
        if spoken_key and spoken_key in LANGUAGES:
            return LANGUAGES[spoken_key]["name"]
        return None

    def _system_message(self, spoken_key: str | None = None) -> str:
        """Build the system prompt, naming BOTH the input and the output language.

        The input language ("I speak ...", or the auto-detected one) and the output
        language ("Reply in ...") are stated separately so the model doesn't just
        mirror the transcript's language back to the TTS engine.
        """
        lang = LANGUAGES[self.response_key]
        out_name = self.lang_name
        in_name = self._input_language_name(spoken_key)
        directive = lang.get("reply_directive", "").strip()

        lines = ["\n\nLANGUAGE INSTRUCTIONS:"]
        if in_name and in_name != out_name:
            # Cross-lingual: the user speaks one language, the reply must be another.
            lines.append(
                f"- The user is speaking to you in {in_name}, so the transcribed "
                f"message you receive is written in {in_name}."
            )
            lines.append(
                f"- No matter what language the user speaks, you MUST write your "
                f"entire reply in {out_name} only. {directive}".rstrip()
            )
            lines.append(
                f"- Do NOT reply in {in_name} and do NOT mirror the user's language; "
                f"translate your response into {out_name}."
            )
        else:
            # Same language in and out (explicit match, or the input is unknown).
            spoken = f"in {in_name}" if in_name else "to you"
            lines.append(f"- The user is speaking {spoken}.")
            lines.append(
                f"- Write your entire reply in {out_name} only. {directive}".rstrip()
            )
            lines.append(
                f"- Never switch to another language, even if the user does."
            )
        instr = "\n".join(lines)

        # Tutor framing only when the user deliberately practices a different language.
        if not self.match_speech and self.stt_language and self.stt_language != lang["stt"]:
            instr += (
                f"\n- Also act as a friendly {out_name} tutor and gently correct the "
                "user's mistakes when it helps them learn."
            )
        return load_system_prompt() + instr

    def _reply_nudge(self) -> str:
        """A short, per-turn instruction appended to the latest user message.

        Qwen tends to mirror the language of the most recent user turn, so a system
        prompt directive alone is unreliable. Repeating the requirement right on the
        user message (in English + the target language) makes it stick.
        """
        directive = LANGUAGES[self.response_key].get("reply_directive", "").strip()
        return f"\n\n[Reply only in {self.lang_name}. {directive}]"

    def _agent_messages(self) -> list[dict]:
        """History copy with the reply-language nudge on the last user message.

        The stored history stays clean (plain transcripts); only the copy handed to
        the LLM carries the nudge.
        """
        msgs = [dict(m) for m in self.history]
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i]["content"] = (msgs[i].get("content") or "") + self._reply_nudge()
                break
        return msgs

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

        # The language whisper actually detected for this utterance (used both to
        # name the input language in the prompt and to pick a replay voice).
        detected_key = language_key_from_whisper(detected)

        # Tag the user message with the spoken language/voice so the UI can
        # replay it (synthesize the user's own text) in a matching voice.
        spoken_key = detected_key or self.response_key
        if not tts.supports(spoken_key):
            spoken_key = self.response_key
        spoken_voice = tts.default_voice(spoken_key) or self.tts_voice
        await self.ws.send_json(
            {"type": "user", "text": transcript,
             "language": spoken_key, "voice": spoken_voice}
        )
        self.history.append({"role": "user", "content": transcript})
        self._record("user", transcript, spoken_key, spoken_voice)

        accumulator = SentenceAccumulator()
        final_answer = ""

        async for event, payload in agent.stream_reply(
            self._agent_messages(), self._system_message(detected_key)
        ):
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
        self._record("assistant", final_answer, self.response_key, self.tts_voice)
        # Persist the turn (creates the file on first content) and let the UI know.
        meta = await asyncio.to_thread(self._persist)
        await self._update_token_usage()
        if meta:
            await self.ws.send_json({"type": "session_saved", "session": meta})
        await self.ws.send_json({"type": "done"})

    async def _update_token_usage(self) -> None:
        """Estimate current context-token usage (system prompt + conversation)."""
        parts = [self._system_message()]
        parts += [m["content"] for m in self.history if isinstance(m.get("content"), str)]
        count = await metrics.count_tokens("\n".join(parts))
        if count is not None:
            metrics.set_current_tokens(count)
