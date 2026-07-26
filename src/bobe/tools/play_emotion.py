import logging
import threading
from typing import Any, Dict

from bobe.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

EMOTION_LIBRARY_ID = "pollen-robotics/reachy-mini-emotions-library"

# Structural import failures (missing packages) are not retryable.
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from bobe.dance_emotion_moves import EmotionQueueMove

    _IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RecordedMoves = None  # type: ignore[assignment]
    EmotionQueueMove = None  # type: ignore[assignment]
    _IMPORTS_AVAILABLE = False

_RECORDED_MOVES: Any = None
_RECORDED_MOVES_LOCK = threading.Lock()


def _get_recorded_moves() -> Any:
    """Return the recorded-emotions library, loading it lazily on demand.

    Loading touches the HuggingFace Hub (network and local cache), so any
    failure - not just ImportError - is caught and logged loudly; the next
    call simply retries instead of disabling the tool for the whole run.
    """
    global _RECORDED_MOVES
    if not _IMPORTS_AVAILABLE:
        return None
    with _RECORDED_MOVES_LOCK:
        if _RECORDED_MOVES is None:
            try:
                # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
                _RECORDED_MOVES = RecordedMoves(EMOTION_LIBRARY_ID)
            except Exception as e:
                logger.error(
                    "Failed to load emotion library '%s' (will retry on next use): %s",
                    EMOTION_LIBRARY_ID,
                    e,
                )
                return None
        return _RECORDED_MOVES


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    recorded_moves = _get_recorded_moves()
    if recorded_moves is None:
        return "Emotion list unavailable right now; calling with an unknown name returns the available emotions."

    try:
        emotion_names = recorded_moves.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = recorded_moves.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        recorded_moves = _get_recorded_moves()
        if recorded_moves is None:
            return {"error": "Emotion library unavailable (still loading or unreachable); please try again"}

        # Check if emotion exists
        try:
            emotion_names = recorded_moves.list_moves()
            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, recorded_moves)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
