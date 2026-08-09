# ruff: noqa: D103
import numpy as np

from bobe.vision.presence import PresenceWatcher


class FakeCamera:
    def __init__(self, frame=None):
        self.frame = frame

    def get_latest_frame(self):
        return self.frame


def test_ships_encoded_snapshot():
    shipped = []
    watcher = PresenceWatcher(
        FakeCamera(np.zeros((480, 640, 3), dtype=np.uint8)),
        shipped.append,
    )

    assert watcher.check_once() is True
    assert len(shipped) == 1
    assert shipped[0][:2] == b"\xff\xd8"  # JPEG magic
    assert len(shipped[0]) < 50_000


def test_no_frame_ships_nothing():
    shipped = []
    watcher = PresenceWatcher(FakeCamera(None), shipped.append)
    assert watcher.check_once() is False
    assert shipped == []


def test_encoder_failure_is_contained():
    shipped = []

    def broken(frame):
        raise RuntimeError("boom")

    watcher = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)), shipped.append, encoder=broken
    )
    assert watcher.check_once() is False
    assert shipped == []
    assert "boom" in watcher.debug_state()["last_error"]
