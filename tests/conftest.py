"""Pytest configuration for path setup."""

import os
import sys
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
    monkeypatch.setattr("bobe.openai_realtime.create_wake_detector", lambda *args, **kwargs: None)
