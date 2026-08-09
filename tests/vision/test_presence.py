# ruff: noqa: D103
import numpy as np

from bobe.vision.presence import PresenceWatcher


class FakeCamera:
    def __init__(self, frame=None):
        self.frame = frame

    def get_latest_frame(self):
        return self.frame


def test_presence_reports_when_face_detected(monkeypatch):
    reports = []
    watcher = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)),
        lambda: reports.append(1),
        dwell_checks=1,
        detector=lambda frame: True,
    )

    assert watcher.check_once() is True
    assert reports == [1]


def test_presence_rate_limits_reports(monkeypatch):
    import bobe.vision.presence as presence_mod

    now = {"t": 1000.0}
    monkeypatch.setattr(presence_mod.time, "monotonic", lambda: now["t"])
    reports = []
    watcher = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)),
        lambda: reports.append(1),
        report_interval_s=30.0,
        dwell_checks=1,
        detector=lambda frame: True,
    )

    watcher.check_once()
    now["t"] += 5.0
    watcher.check_once()  # throttled
    now["t"] += 30.0
    watcher.check_once()  # allowed again

    assert len(reports) == 2


def test_presence_no_frame_or_no_face_is_quiet():
    reports = []
    watcher = PresenceWatcher(FakeCamera(None), lambda: reports.append(1), detector=lambda f: True)
    assert watcher.check_once() is False

    watcher2 = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)),
        lambda: reports.append(1),
        detector=lambda frame: False,
    )
    assert watcher2.check_once() is False
    assert reports == []


def test_presence_dwell_requires_consecutive_hits():
    reports = []
    seen = {"present": True}
    watcher = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)),
        lambda: reports.append(1),
        dwell_checks=3,
        detector=lambda frame: seen["present"],
    )

    watcher.check_once()  # hit 1
    watcher.check_once()  # hit 2 — still below dwell
    assert reports == []

    watcher.check_once()  # hit 3 — sat down
    assert reports == [1]


def test_presence_dwell_resets_on_gap():
    reports = []
    seen = {"present": True}
    watcher = PresenceWatcher(
        FakeCamera(np.zeros((10, 10, 3), dtype=np.uint8)),
        lambda: reports.append(1),
        dwell_checks=3,
        detector=lambda frame: seen["present"],
    )

    watcher.check_once()
    watcher.check_once()
    seen["present"] = False  # walked away between checks
    watcher.check_once()
    seen["present"] = True
    watcher.check_once()
    watcher.check_once()
    assert reports == []  # never 3 in a row

    watcher.check_once()
    assert reports == [1]
