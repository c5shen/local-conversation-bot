"""Qwen-Agent reasoning loop with Tavily web-search + page-extract tools.

Qwen-Agent runs as a synchronous generator. We run it in a worker thread and
bridge its output to an async generator that yields events:

    ("tool_call", tool_name)   - the agent invoked a tool (e.g. search)
    ("answer", delta)          - new final-answer text (thinking already stripped)
    ("final", full_text)       - the complete final answer (for chat history)
    ("error", message)         - something went wrong
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator

from .config import load_system_prompt, settings

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_THINK_OPEN = re.compile(r"<think>.*$", re.S)

# Fallback prompt budget when the llama-server context can't be probed yet.
_FALLBACK_INPUT_TOKENS = 32768
# Tokens reserved on top of max_tokens so input + reply stays inside the context.
_CONTEXT_MARGIN = 512

_bot = None
_input_budget_set = False


def _strip_think(text: str) -> str:
    """Remove Qwen reasoning so it is never spoken, including an unclosed block."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def _build_bot():
    from qwen_agent.agents import Assistant

    llm_cfg = {
        "model": settings.llm_model,
        "model_type": settings.llm_model_type,
        "model_server": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "generate_cfg": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            # max_input_tokens is resolved against the real context window in
            # _apply_input_budget() once llama-server is reachable.
            "max_input_tokens": _FALLBACK_INPUT_TOKENS,
            "max_tokens": settings.max_tokens,
            # Ask the server to disable Qwen3.5 thinking when it honors the flag.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    }

    function_list = []
    if settings.enable_search:
        if settings.tavily_api_key.strip():
            # Importing the module registers the tools with Qwen-Agent's registry.
            from . import tools_tavily  # noqa: F401
            function_list += ["tavily_search", "tavily_extract"]
        else:
            logger.warning(
                "Web search disabled: TAVILY_API_KEY is not set. Add it to your .env "
                "to enable Tavily search/extract (https://tavily.com)."
            )

    try:
        return Assistant(llm=llm_cfg, function_list=function_list, system_message=load_system_prompt())
    except Exception as exc:  # noqa: BLE001 - never let a bad tool block startup
        if function_list:
            logger.warning("Web search tool failed to start (%s); continuing without it.", exc)
            return Assistant(llm=llm_cfg, function_list=[], system_message=load_system_prompt())
        raise


def _fetch_context_size() -> int:
    """llama-server's n_ctx (full context window), or 0 if it can't be reached."""
    import httpx

    root = settings.llm_base_url.rsplit("/v1", 1)[0]
    try:
        data = httpx.get(f"{root}/props", timeout=5).json()
        return int(data.get("default_generation_settings", {}).get("n_ctx") or 0)
    except Exception:  # noqa: BLE001 - server not up yet / unreachable
        return 0


def _apply_input_budget() -> None:
    """Point Qwen-Agent's prompt-truncation budget at the real context window.

    Without this the agent truncates input to a small fixed value regardless of
    the llama-server `-c` setting. Resolved once (cached after a successful probe).
    """
    global _input_budget_set
    if _input_budget_set or _bot is None:
        return
    # compute maximum input tokens allowed
    if settings.max_input_tokens > 0:
        budget, _input_budget_set = settings.max_input_tokens, True
    else:
        n_ctx = _fetch_context_size()
        if n_ctx > 0:
            budget = max(_FALLBACK_INPUT_TOKENS, n_ctx - settings.max_tokens - _CONTEXT_MARGIN)
            _input_budget_set = True
        else:
            # llama-server not reachable yet; use a safe budget and retry next turn.
            budget = _FALLBACK_INPUT_TOKENS
    _bot.llm.generate_cfg["max_input_tokens"] = budget
    if _input_budget_set:
        logger.info("Qwen-Agent prompt budget (max_input_tokens) set to %d.", budget)


async def init() -> None:
    """Build the agent (and spawn the MCP server) once, off the event loop."""
    global _bot
    if _bot is None:
        _bot = await asyncio.to_thread(_build_bot)
    await asyncio.to_thread(_apply_input_budget)


def _extract_answer(responses: list) -> str:
    """Concatenate textual assistant content (ignoring tool-call messages)."""
    parts: list[str] = []
    for msg in responses:
        if msg.get("role") != "assistant" or msg.get("function_call"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            parts.append(content)
    return _strip_think("".join(parts))


def _message_text(content) -> str:
    """Flatten a message's content (string or list of content blocks) to text."""
    if isinstance(content, list):
        content = "\n".join(
            (p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return (content or "").strip()


def _tool_query(responses: list, idx: int, name: str) -> str:
    """Find the query a tool was called with, by scanning back for its call args.

    Prefers the call matching `name`, but falls back to the nearest preceding
    call so we still show a query if the tool/result names differ.
    """
    fallback = ""
    for j in range(idx - 1, -1, -1):
        call = responses[j].get("function_call")
        if not call:
            continue
        args = call.get("arguments") or ""
        try:
            parsed = json.loads(args)
            query = str(parsed.get("query") or parsed.get("q") or parsed.get("keywords") or args) \
                if isinstance(parsed, dict) else args
        except (ValueError, TypeError):
            query = args
        if not name or call.get("name") == name:
            return query
        if not fallback:
            fallback = query
    return fallback


async def stream_reply(
    history: list[dict], system_message: str | None = None
) -> AsyncIterator[tuple[str, object]]:
    """Run the agent for the current history and stream events asynchronously.

    system_message overrides the agent's system prompt for this run (used to steer
    the reply language). Runs are sequential per websocket; this app is single-user.
    """
    if _bot is None:
        await init()
    # Resolve the prompt budget if startup couldn't reach llama-server yet.
    if not _input_budget_set:
        await asyncio.to_thread(_apply_input_budget)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(item) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker() -> None:
        seen_calls: set[int] = set()    # announced function-call message indices
        seen_results: set[int] = set()  # delivered tool-result message indices
        last_answer = ""
        final_answer = ""
        if system_message:
            _bot.system_message = system_message
        try:
            for responses in _bot.run(messages=history):
                for i, msg in enumerate(responses):
                    call = msg.get("function_call")
                    if call and i not in seen_calls:
                        seen_calls.add(i)
                        emit(("tool_call", call.get("name") or "tool"))
                    # A finished tool result: pair it with its call's query.
                    if msg.get("role") == "function" and msg.get("content") and i not in seen_results:
                        seen_results.add(i)
                        name = msg.get("name") or "search"
                        emit(("tool_result", {
                            "name": name,
                            "query": _tool_query(responses, i, name),
                            "text": _message_text(msg.get("content"))[:4000],
                        }))
                answer = _extract_answer(responses)
                if len(answer) > len(last_answer):
                    emit(("answer", answer[len(last_answer):]))
                    last_answer = answer
                final_answer = answer
            emit(("final", final_answer))
        except Exception as exc:  # noqa: BLE001 - surface any agent failure to the UI
            emit(("error", str(exc)))
        finally:
            emit(None)

    loop.run_in_executor(None, worker)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
