"""Profile tool that routes BoBe questions to Claude."""

import logging
from typing import Any

from bobe.claude import ClaudeNotConfiguredError, ask_claude, load_claude_settings
from bobe.emotion_policy import classify_emotion
from bobe.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class AskClaude(Tool):
    """Ask Claude for the answer BoBe should speak."""

    name = "ask_claude"
    description = (
        "Ask Claude online for a concise answer to the user's question. "
        "Claude can search the web, so use this for anything needing current information "
        "such as weather, news, or prices, as well as normal BoBe assistant answers."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question or request to send to Claude.",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Ask Claude and return a speakable answer."""
        question = str(kwargs.get("question", "")).strip()
        if not question:
            return {"status": "error", "error": "question is required"}

        settings = load_claude_settings()
        try:
            answer = await ask_claude(question, settings=settings)
        except ClaudeNotConfiguredError as exc:
            return {
                "status": "missing_api_key",
                "error": str(exc),
                "setup": "Set ANTHROPIC_API_KEY in the app environment.",
            }
        except Exception as exc:
            logger.exception("Claude request failed")
            return {"status": "error", "error": f"Claude request failed: {type(exc).__name__}"}

        emotion = classify_emotion(answer)
        return {
            "status": "ok",
            "model": settings.model,
            "answer": answer,
            "emotion": emotion.emotion,
            "should_play_emotion": emotion.should_play_emotion,
        }
