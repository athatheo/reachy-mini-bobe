"""Claude integration helpers for BoBe."""

from __future__ import annotations
import os
from typing import Any, Mapping, Protocol, cast
from dataclasses import dataclass

from bobe.prompts import get_claude_system_prompt
from bobe.env_utils import parse_int, clean_optional


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024  # search tool-use blocks count toward output tokens
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
DEFAULT_WEB_SEARCH_MAX_USES = 3
# Explicit per-request timeout: the SDK default (10 minutes) would pin the
# background tool on a stuck request for the rest of the conversation.
DEFAULT_REQUEST_TIMEOUT_S = 60.0


class ClaudeNotConfiguredError(RuntimeError):
    """Raised when Claude is requested without an Anthropic API key."""


@dataclass(frozen=True)
class ClaudeSettings:
    """Runtime settings for Claude-backed answers."""

    api_key: str | None
    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    web_search: bool = True

    @property
    def is_configured(self) -> bool:
        """Return whether Claude has the credentials needed for API calls."""
        return bool(self.api_key and self.api_key.strip())


def load_claude_settings(env: Mapping[str, str] | None = None) -> ClaudeSettings:
    """Load Claude settings from environment variables."""
    source = os.environ if env is None else env

    return ClaudeSettings(
        api_key=clean_optional(source.get("ANTHROPIC_API_KEY")),
        model=source.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL,
        max_tokens=max(1, parse_int(source.get("CLAUDE_MAX_TOKENS"), DEFAULT_MAX_TOKENS)),
        web_search=(source.get("BOBE_CLAUDE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}),
    )


def extract_message_text(message: Any) -> str:
    """Extract plain text from an Anthropic message response."""
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts).strip()


class _MessagesClient(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _ClaudeClient(Protocol):
    messages: _MessagesClient


# One process-wide AsyncAnthropic client (it owns an httpx connection pool):
# creating one per ask_claude call leaks pooled sockets until GC.
_shared_client: Any = None
_shared_client_api_key: str | None = None


async def _get_shared_client(api_key: str) -> _ClaudeClient:
    """Return the shared Anthropic client, rebuilding it when the API key changes."""
    global _shared_client, _shared_client_api_key
    if _shared_client is not None and _shared_client_api_key == api_key:
        return cast(_ClaudeClient, _shared_client)

    from anthropic import AsyncAnthropic

    stale = _shared_client
    _shared_client = AsyncAnthropic(api_key=api_key, timeout=DEFAULT_REQUEST_TIMEOUT_S)
    _shared_client_api_key = api_key
    if stale is not None:
        await _close_client(stale)
    return cast(_ClaudeClient, _shared_client)


async def aclose_shared_claude_client() -> None:
    """Close the shared Anthropic client (app shutdown / tests)."""
    global _shared_client, _shared_client_api_key
    stale = _shared_client
    _shared_client = None
    _shared_client_api_key = None
    if stale is not None:
        await _close_client(stale)


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        await close()


async def ask_claude(
    question: str,
    *,
    settings: ClaudeSettings | None = None,
    client: _ClaudeClient | None = None,
) -> str:
    """Ask Claude for a spoken-friendly answer."""
    active_settings = settings or load_claude_settings()
    if not active_settings.is_configured:
        raise ClaudeNotConfiguredError("ANTHROPIC_API_KEY is required to ask Claude")

    active_client = client
    if active_client is None:
        active_client = await _get_shared_client(cast(str, active_settings.api_key))

    request: dict[str, Any] = {
        "model": active_settings.model,
        "max_tokens": active_settings.max_tokens,
        "system": get_claude_system_prompt(),
        "messages": [{"role": "user", "content": question}],
    }
    if active_settings.web_search:
        request["tools"] = [
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": DEFAULT_WEB_SEARCH_MAX_USES,
            }
        ]

    response = await active_client.messages.create(**request)
    return extract_message_text(response)
