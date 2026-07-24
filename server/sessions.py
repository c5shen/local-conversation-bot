"""Local session-log storage: one JSON file per conversation under ./.cache.

A session is persisted only once it has content (at least one user transcript or
assistant reply). Files are named by the session's creation timestamp, which also
serves as the session id.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{3}$")


def _cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def new_id() -> str:
    """A filename-safe, sortable id from the current local time (ms precision)."""
    now = time.time()
    stamp = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{int((now % 1) * 1000):03d}"


def _valid_id(session_id: str) -> bool:
    return bool(session_id) and bool(_ID_RE.match(session_id))


def _path(session_id: str) -> Path | None:
    if not _valid_id(session_id):
        return None
    return _cache_dir() / f"{session_id}.json"


def _title(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and (msg.get("text") or "").strip():
            text = " ".join(msg["text"].split())
            return text[:60] + ("..." if len(text) > 60 else "")
    for msg in messages:
        if (msg.get("text") or "").strip():
            text = " ".join(msg["text"].split())
            return text[:60] + ("..." if len(text) > 60 else "")
    return "Untitled session"


def _meta(data: dict) -> dict:
    messages = data.get("messages") or []
    return {
        "id": data.get("id"),
        "title": data.get("title") or _title(messages),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "turns": sum(1 for m in messages if m.get("role") == "user"),
    }


def save(session_id: str, created_at: str, messages: list[dict]) -> dict | None:
    """Write (or overwrite) a session file. No-op for empty sessions or bad ids."""
    path = _path(session_id)
    if path is None or not messages:
        return None
    data = {
        "id": session_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "title": _title(messages),
        "messages": messages,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem
    return _meta(data)


def load(session_id: str) -> dict | None:
    """Return the full session dict (with messages), or None if missing/invalid."""
    path = _path(session_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_all() -> list[dict]:
    """Session metadata for every stored session, newest-updated first."""
    out: list[dict] = []
    for path in _cache_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("messages"):
            out.append(_meta(data))
    out.sort(key=lambda m: m.get("updated_at") or "", reverse=True)
    return out


def delete(session_id: str) -> bool:
    path = _path(session_id)
    if path is None or not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
