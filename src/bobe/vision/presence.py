"""Lightweight person-presence detection from the camera frame buffer.

Watches the camera worker's latest frame for a frontal face (OpenCV Haar
cascade — no extra ML dependencies) and reports sightings to a callback at a
throttled rate. The Mac wake daemon turns those sightings into behavior (e.g.
the first-sighting-after-6am morning briefing); the robot only reports.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Any, Callable

import cv2


logger = logging.getLogger(__name__)

# How often a frame is checked, and the minimum spacing between reports.
CHECK_INTERVAL_S = 5.0
REPORT_INTERVAL_S = 30.0
# Dwell requirement: this many consecutive positive checks before a sighting
# is reported. With 5 s checks, 3 hits ≈ 10-15 s of continuous presence —
# someone sitting down, not walking past.
DWELL_CHECKS = 3
# Haar tuning: robust-ish defaults; a desk-distance face is well over 60 px.
_SCALE_FACTOR = 1.1
_MIN_NEIGHBORS = 5
_MIN_SIZE = (60, 60)


class PresenceWatcher:
    """Poll camera frames for a face and report sightings (rate-limited)."""

    def __init__(
        self,
        camera_worker: Any,
        on_present: Callable[[], None],
        *,
        check_interval_s: float = CHECK_INTERVAL_S,
        report_interval_s: float = REPORT_INTERVAL_S,
        dwell_checks: int = DWELL_CHECKS,
        detector: Callable[[Any], bool] | None = None,
    ) -> None:
        """Create the watcher; ``detector`` is injectable for tests."""
        self._camera_worker = camera_worker
        self._on_present = on_present
        self._check_interval_s = check_interval_s
        self._report_interval_s = report_interval_s
        self._dwell_checks = max(1, dwell_checks)
        self._detector = detector or self._haar_face_present
        self._cascade: Any = None
        self._consecutive_hits = 0
        self._last_report_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _haar_face_present(self, frame: Any) -> bool:
        if self._cascade is None:
            # cv2.data is untyped in the opencv stubs; resolve defensively.
            haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
            self._cascade = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=_SCALE_FACTOR,
            minNeighbors=_MIN_NEIGHBORS,
            minSize=_MIN_SIZE,
        )
        return len(faces) > 0

    def check_once(self) -> bool:
        """Run one detection pass; report (dwell-gated, throttled) on presence.

        A sighting is only reported after ``dwell_checks`` consecutive
        positive checks, so someone walking past the camera between two
        checks never counts as "sat down".
        """
        frame = self._camera_worker.get_latest_frame()
        if frame is None:
            self._consecutive_hits = 0
            return False
        try:
            present = bool(self._detector(frame))
        except Exception:
            logger.debug("Presence detection failed on a frame", exc_info=True)
            present = False
        if not present:
            self._consecutive_hits = 0
            return False
        self._consecutive_hits += 1
        if self._consecutive_hits < self._dwell_checks:
            return True
        now = time.monotonic()
        if now - self._last_report_at >= self._report_interval_s:
            self._last_report_at = now
            try:
                self._on_present()
            except Exception:
                logger.exception("Presence report callback failed")
        return True

    def _run(self) -> None:
        logger.info("Presence watcher started (every %.0fs)", self._check_interval_s)
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
