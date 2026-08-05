"""Profile tool that sends BoBe requests to Hermes, the user's personal agent."""

import os
import logging
from typing import Any

import httpx

from bobe.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# One Hermes agent turn can include tool calls; give it real time to work.
REQUEST_TIMEOUT_S = 90.0
# Constant session id so consecutive voice asks share one Hermes conversation.
HERMES_SESSION_ID = "bobe-voice"


def _extract_answer(data: Any) -> str:
    """Pull the assistant text out of an OpenAI-format chat completion."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content.strip() if isinstance(content, str) else ""


class AskHermes(Tool):
    """Send a request to the user's Hermes agent and return its answer."""

    name = "ask_hermes"
    description = (
        "Send a request to Hermes, the user's personal agent on their Mac. "
        "Use it for anything involving the user's own tasks, messages, kanban, "
        "files, computer, or long-running jobs — or whenever the user says "
        "'Hermes'. Answer ordinary conversation and general knowledge yourself. "
        "Hermes can take up to a minute to reply."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's request or question, forwarded to Hermes verbatim.",
            },
        },
        "required": ["request"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Forward the request to Hermes and return a speakable answer."""
        request = str(kwargs.get("request", "")).strip()
        if not request:
            return {"status": "error", "error": "request is required"}

        base_url = (os.getenv("BOBE_HERMES_URL") or "").strip().rstrip("/")
        api_key = (os.getenv("BOBE_HERMES_API_KEY") or "").strip()
        if not base_url or not api_key:
            return {
                "status": "missing_config",
                "error": "Hermes is not configured on this robot.",
                "setup": "Set BOBE_HERMES_URL and BOBE_HERMES_API_KEY in the app environment.",
            }

        payload = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": request}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Hermes-Session-Id": HERMES_SESSION_ID,
        }
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
                response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return {"status": "error", "error": "Hermes did not answer in time; try again or check on it later."}
        except Exception as exc:
            logger.exception("Hermes request failed")
            return {"status": "error", "error": f"Hermes request failed: {type(exc).__name__}"}

        answer = _extract_answer(data)
        if not answer:
            return {"status": "error", "error": "Hermes returned an empty answer."}
        return {"status": "ok", "answer": answer}
