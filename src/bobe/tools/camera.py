import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

from bobe.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _encode_frame_jpeg_b64(frame: Any) -> str:
    """Encode a BGR frame to a base64 JPEG string (CPU-bound; run in a thread)."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No fresh frame available from camera worker")
                return {"error": "No recent camera frame available; the camera may be stalled or disconnected"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # Encode image directly to JPEG bytes without writing to file. Run in a
        # worker thread: encoding a full frame takes tens of milliseconds and
        # would otherwise stall the realtime audio event loop.
        b64_encoded = await asyncio.to_thread(_encode_frame_jpeg_b64, frame)
        return {"b64_im": b64_encoded}
