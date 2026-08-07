"""Pytest configuration for path setup and shared test stubs."""

import os
import sys
import json
from typing import Any, Callable
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)


@pytest.fixture(autouse=True)
def _stub_wake_detector_unless_real(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Avoid spinning up ONNX wake detectors in handler tests."""
    if request.node.get_closest_marker("wake_detector"):
        return
    monkeypatch.setattr("bobe.voice_handler.create_wake_detector", lambda *args, **kwargs: None)


class JsonResponse:
    """Context-manager urllib response stub returning a fixed JSON payload."""

    def __init__(self, payload: Any) -> None:
        """Store the payload serialized by read()."""
        self._payload = payload

    def __enter__(self) -> "JsonResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def read(self) -> bytes:
        """Return the JSON-encoded payload."""
        return json.dumps(self._payload).encode()


def make_opener(
    payload: Any,
    calls: list[tuple[Any, float]] | None = None,
    *,
    before_response: Callable[[], None] | None = None,
) -> Callable[..., JsonResponse]:
    """Build a urllib opener stub returning a fresh JsonResponse(payload) per call.

    When ``calls`` is given, each invocation records a ``(request, timeout)``
    tuple. An optional ``before_response`` hook runs (and may block) before
    the response is constructed, mirroring a slow network round-trip.
    """

    def _opener(request: Any, *, timeout: float) -> JsonResponse:
        if before_response is not None:
            before_response()
        if calls is not None:
            calls.append((request, timeout))
        return JsonResponse(payload)

    return _opener
