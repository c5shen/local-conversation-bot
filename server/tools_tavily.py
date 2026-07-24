"""Tavily-backed web tools for Qwen-Agent: `tavily_search` and `tavily_extract`.

Importing this module registers the tools with Qwen-Agent's global registry; the
agent then enables them by name (see server/agent.py). A single TavilyClient is
created lazily on first use from settings.tavily_api_key (TAVILY_API_KEY in .env).
"""
from __future__ import annotations

import json
import logging

from qwen_agent.tools.base import BaseTool, register_tool

from .config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazily build (and cache) the shared TavilyClient; raise if no API key."""
    global _client
    if _client is None:
        api_key = (settings.tavily_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "Tavily web search is unavailable: set TAVILY_API_KEY in your .env."
            )
        from tavily import TavilyClient

        _client = TavilyClient(api_key=api_key)
    return _client


def _args(params) -> dict:
    """Qwen-Agent passes tool arguments as a JSON string or an already-parsed dict."""
    if isinstance(params, dict):
        return params
    try:
        parsed = json.loads(params)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


@register_tool("tavily_search")
class TavilySearch(BaseTool):
    """Search the web via Tavily and return the top results (title, URL, snippet)."""

    description = (
        "Search the web for current, factual, or fresh information using the Tavily "
        "search engine. Use this whenever you are unsure of an answer or need up-to-date "
        "facts, news, prices, or events. Returns a concise answer plus the most relevant "
        "web results with their URLs."
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query, phrased as natural-language keywords.",
            "required": True,
        }
    ]

    def call(self, params, **kwargs) -> str:
        query = str(_args(params).get("query") or "").strip()
        if not query:
            return "No search query was provided."
        try:
            resp = _get_client().search(
                query=query,
                search_depth=settings.tavily_search_depth,
                max_results=settings.tavily_max_results,
                include_answer=settings.tavily_include_answer,
            )
        except Exception as exc:  # noqa: BLE001 - surface a readable observation
            logger.warning("Tavily search failed: %s", exc)
            return f"Web search failed: {exc}"

        lines: list[str] = []
        answer = (resp.get("answer") or "").strip()
        if answer:
            lines.append(f"Answer: {answer}\n")
        results = resp.get("results") or []
        if not results:
            lines.append("No results found.")
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = " ".join((r.get("content") or "").split())[:500]
            lines.append(f"{i}. {title}\n   {url}\n   {content}")
        return "\n".join(lines).strip()


@register_tool("tavily_extract")
class TavilyExtract(BaseTool):
    """Fetch and return the readable text content of one or more web pages via Tavily."""

    description = (
        "Extract the full readable text content from one or more web page URLs using "
        "Tavily. Use this after a search to read a specific page in depth, or when the "
        "user gives you a URL to summarize."
    )
    parameters = [
        {
            "name": "urls",
            "type": "array",
            "items": {"type": "string"},
            "description": "One or more http(s) URLs to extract the page content from.",
            "required": True,
        }
    ]

    def call(self, params, **kwargs) -> str:
        raw = _args(params).get("urls")
        if isinstance(raw, str):
            urls = [raw.strip()] if raw.strip() else []
        elif isinstance(raw, list):
            urls = [str(u).strip() for u in raw if str(u).strip()]
        else:
            urls = []
        if not urls:
            return "No URLs were provided to extract."
        try:
            resp = _get_client().extract(urls=urls)
        except Exception as exc:  # noqa: BLE001 - surface a readable observation
            logger.warning("Tavily extract failed: %s", exc)
            return f"Page extraction failed: {exc}"

        lines: list[str] = []
        for r in resp.get("results") or []:
            url = (r.get("url") or "").strip()
            content = " ".join((r.get("raw_content") or "").split())[:3000]
            lines.append(f"URL: {url}\n{content}")
        for f in resp.get("failed_results") or []:
            url = (f.get("url") or "").strip() if isinstance(f, dict) else str(f)
            lines.append(f"URL: {url}\n(could not extract this page)")
        return "\n\n".join(lines).strip() or "No content could be extracted."
