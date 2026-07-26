import logging
from typing import Any, Dict

from bobe.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class HeadTracking(Tool):
    """Toggle head tracking state."""

    name = "head_tracking"
    description = "Toggle head tracking state."
    parameters_schema = {
        "type": "object",
        "properties": {"start": {"type": "boolean"}},
        "required": ["start"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Enable or disable head tracking."""
        enable = bool(kwargs.get("start"))

        if deps.camera_worker is None:
            logger.warning("Tool call: head_tracking requested but camera is disabled")
            return {"error": "camera is not available, so head tracking cannot be used"}

        tracker = deps.camera_worker.head_tracker
        if tracker is None or not getattr(tracker, "available", True):
            logger.warning("Tool call: head_tracking requested but no head tracker is configured")
            return {"error": "no head tracker is configured; the app must be launched with --head-tracker yolo or mediapipe"}

        # Update camera worker head tracking state
        deps.camera_worker.set_head_tracking_enabled(enable)

        status = "started" if enable else "stopped"
        logger.info("Tool call: head_tracking %s", status)
        return {"status": f"head tracking {status}"}
