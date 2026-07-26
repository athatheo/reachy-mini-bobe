# ruff: noqa: D101,D102,D103,D107

import math
from types import SimpleNamespace

import numpy as np
import pytest

from bobe.tools import play_emotion
from bobe.tools.move_head import MoveHead
from bobe.tools.stop_dance import StopDance
from bobe.tools.stop_emotion import StopEmotion
from bobe.profiles._bobe_locked_profile.sweep_look import SweepLook


class FakeMovementManager:
    def __init__(self):
        self.clear_count = 0
        self.queued = []

    def clear_move_queue(self):
        self.clear_count += 1

    def queue_move(self, move):
        self.queued.append(move)


class FakeReachyMini:
    def __init__(self, body_yaw=0.0, antennas=(0.0, 0.0)):
        self._body_yaw = body_yaw
        self._antennas = antennas

    def get_current_head_pose(self):
        return np.eye(4)

    def get_current_joint_positions(self):
        head_joints = np.array([self._body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return head_joints, np.array(self._antennas)


def _pose_yaw(pose):
    """Extract the yaw angle from a pure-yaw 4x4 head pose."""
    return math.atan2(pose[1, 0], pose[0, 0])


@pytest.mark.parametrize("tool_cls", [StopDance, StopEmotion])
def test_stop_tools_do_not_require_dummy_arguments(tool_cls):
    schema = tool_cls.parameters_schema

    assert schema["properties"] == {}
    assert schema["required"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls", [StopDance, StopEmotion])
async def test_stop_tools_clear_move_queue_without_arguments(tool_cls):
    movement_manager = FakeMovementManager()
    deps = SimpleNamespace(movement_manager=movement_manager)

    result = await tool_cls()(deps)

    assert movement_manager.clear_count == 1
    assert result["status"]


@pytest.mark.asyncio
async def test_move_head_starts_body_yaw_from_head_joints():
    """move_head must not use the antenna joint angle as body yaw (finding #30)."""
    movement_manager = FakeMovementManager()
    deps = SimpleNamespace(
        movement_manager=movement_manager,
        reachy_mini=FakeReachyMini(body_yaw=0.3, antennas=(-0.5, 0.5)),
        motion_duration_s=1.0,
    )

    result = await MoveHead()(deps, direction="left")

    assert result == {"status": "looking left"}
    assert len(movement_manager.queued) == 1
    move = movement_manager.queued[0]
    assert move.start_body_yaw == pytest.approx(0.3)
    assert move.target_body_yaw == 0
    assert move.start_antennas == (-0.5, 0.5)


@pytest.mark.asyncio
async def test_sweep_look_uses_bounded_lockstep_yaw_legs():
    """sweep_look legs must be bounded and keep head/body yaw consistent (finding #12)."""
    movement_manager = FakeMovementManager()
    deps = SimpleNamespace(
        movement_manager=movement_manager,
        reachy_mini=FakeReachyMini(body_yaw=0.4, antennas=(-0.5, 0.5)),
        motion_duration_s=1.0,
    )

    await SweepLook()(deps)

    assert movement_manager.clear_count == 1
    moves = movement_manager.queued
    assert len(moves) >= 7

    # The first move recenters from the arbitrary current pose.
    first = moves[0]
    assert first.start_body_yaw == pytest.approx(0.4)
    assert first.target_body_yaw == pytest.approx(0.0, abs=1e-9)
    assert _pose_yaw(first.target_head_pose) == pytest.approx(0.0, abs=1e-9)

    previous_target = first.target_body_yaw
    for move in moves[1:]:
        # Legs chain continuously...
        assert move.start_body_yaw == pytest.approx(previous_target, abs=1e-9)
        previous_target = move.target_body_yaw
        # ...span at most ~90 degrees so slerp can never wrap the wrong way...
        assert abs(move.target_body_yaw - move.start_body_yaw) <= math.pi / 2 + 1e-9
        # ...and head yaw stays in lockstep with the lerped body yaw.
        assert _pose_yaw(move.start_head_pose) == pytest.approx(move.start_body_yaw, abs=1e-9)
        assert _pose_yaw(move.target_head_pose) == pytest.approx(move.target_body_yaw, abs=1e-9)

    # The sweep reaches both extremes and ends back at center.
    targets = [move.target_body_yaw for move in moves]
    assert max(targets) == pytest.approx(0.9 * math.pi)
    assert min(targets) == pytest.approx(-0.9 * math.pi)
    assert moves[-1].target_body_yaw == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_play_emotion_retries_library_load_after_failure(monkeypatch):
    """A failed HF library load must not disable play_emotion permanently (finding #44)."""
    attempts = {"count": 0}
    fake_move = SimpleNamespace(description="demo", duration=1.0)

    class FakeRecordedMoves:
        def list_moves(self):
            return ["happy"]

        def get(self, name):
            return fake_move

    def flaky_loader(library_id):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("hub unreachable")
        return FakeRecordedMoves()

    monkeypatch.setattr(play_emotion, "_RECORDED_MOVES", None)
    monkeypatch.setattr(play_emotion, "_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(play_emotion, "RecordedMoves", flaky_loader)

    movement_manager = FakeMovementManager()
    deps = SimpleNamespace(movement_manager=movement_manager)
    tool = play_emotion.PlayEmotion()

    first = await tool(deps, emotion="happy")
    assert "error" in first
    assert not movement_manager.queued

    second = await tool(deps, emotion="happy")
    assert second == {"status": "queued", "emotion": "happy"}
    assert len(movement_manager.queued) == 1
    assert attempts["count"] == 2
