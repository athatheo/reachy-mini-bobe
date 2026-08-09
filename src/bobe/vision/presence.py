"""Ship periodic camera snapshots to the Mac daemon for presence detection.

The robot's bundled OpenCV is a minimal build without object detection, so
face-finding happens on the Mac: this watcher just downscales the latest
camera frame to a small JPEG every few seconds and hands it to a callback
(which relays it over the authenticated wake WebSocket). The daemon runs the
face detector and the sit-vs-pass-by dwell logic.
"""

from __future__ import annotations
import logging
import threading
from typing import Any, Callable

import cv2


logger = logging.getLogger(__name__)

# How often a snapshot is shipped. Policy (dwell, daily gates) lives daemon-side.
CHECK_INTERVAL_S = 5.0
# Downscale target width: plenty for a desk-distance face, tiny on the wire.
FRAME_WIDTH = 320
_JPEG_QUALITY = 70


class PresenceWatcher:
    """Ship small camera snapshots for daemon-side presence detection."""

    def __init__(
        self,
        camera_worker: Any,
        on_frame: Callable[[bytes], None],
        *,
        check_interval_s: float = CHECK_INTERVAL_S,
        encoder: Callable[[Any], bytes | None] | None = None,
    ) -> None:
        """Create the watcher; ``encoder`` is injectable for tests."""
        self._camera_worker = camera_worker
        self._on_frame = on_frame
        self._check_interval_s = check_interval_s
        self._encoder = encoder or self._encode_jpeg
        # Diagnostics surfaced via /status.
        self._checks = 0
        self._frames = 0
        self._shipped = 0
        self._last_error = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _encode_jpeg(self, frame: Any) -> bytes | None:
        height, width = frame.shape[:2]
        if width > FRAME_WIDTH:
            scale = FRAME_WIDTH / float(width)
            frame = cv2.resize(frame, (FRAME_WIDTH, max(1, int(height * scale))))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        return encoded.tobytes() if ok else None

    def check_once(self) -> bool:
        """Grab, downscale, and ship one snapshot; returns True when shipped."""
        self._checks += 1
        frame = self._camera_worker.get_latest_frame()
        if frame is None:
            return False
        self._frames += 1
        try:
            jpeg = self._encoder(frame)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:200]
            logger.debug("Snapshot encoding failed", exc_info=True)
            return False
        if not jpeg:
            return False
        try:
            self._on_frame(jpeg)
            self._shipped += 1
        except Exception:
            logger.exception("Snapshot ship callback failed")
            return False
        return True

    def _run(self) -> None:
        logger.info("Presence snapshot shipper started (every %.0fs)", self._check_interval_s)
        while not self._stop_event.wait(self._check_interval_s):
            self.check_once()

    def start(self) -> None:
        """Start the watcher thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="presence-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the watcher thread."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None

    def debug_state(self) -> dict[str, object]:
        """Snapshot of watcher counters for the /status diagnostics page."""
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "checks": self._checks,
            "frames": self._frames,
            "snapshots_shipped": self._shipped,
            "last_error": self._last_error or None,
        }
