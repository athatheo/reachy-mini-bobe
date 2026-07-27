"""User-facing wake/sleep cues: chime synthesis and antenna posture moves.

Pure UX helpers with no realtime-session knowledge; the handler in
``bobe.openai_realtime`` delegates to them around wake/sleep transitions.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Final, Tuple

import numpy as np
from fastrtc import AdditionalOutputs
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# World-frame head translation applied when falling asleep (millimeters, vertical).
_SLEEP_HEAD_Z_OFFSET_MM: Final[float] = 30.0


def make_chime(sample_rate: int, *, ascending: bool) -> NDArray[np.int16]:
    """Generate a short two-tone chime marking a wake/sleep transition."""
    freqs = (660.0, 880.0) if ascending else (880.0, 660.0)
    fade = max(1, int(sample_rate * 0.01))
    tones = []
    for freq in freqs:
        t = np.arange(int(sample_rate * 0.12)) / sample_rate
        tone = 0.25 * np.sin(2 * np.pi * freq * t)
        tone[:fade] *= np.linspace(0.0, 1.0, fade)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade)
        tones.append(tone)
    return (np.concatenate(tones) * 32767).astype(np.int16)


async def play_chime(
    output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]",
    sample_rate: int,
    *,
    ascending: bool,
) -> None:
    """Queue a wake/sleep chime on the caller's current output queue."""
    try:
        chime = make_chime(sample_rate, ascending=ascending)
        await output_queue.put((sample_rate, chime.reshape(1, -1)))
    except Exception:
        logger.debug("Chime skipped", exc_info=True)


def queue_antenna_cue(movement_manager: Any, *, awake: bool) -> None:
    """Raise antennas while streaming; relax them and nod head down when asleep."""
    try:
        from reachy_mini.utils import create_head_pose
        from reachy_mini.utils.interpolation import compose_world_offset
        from bobe.dance_emotion_moves import GotoQueueMove

        # Snapshot the PRIMARY (offset-free) target pose, not the measured
        # pose: the measured pose already contains speech-sway and
        # face-tracking offsets, and using it as the goto target would bake
        # those offsets into the primary pose permanently (finding #33).
        head_pose, antennas, body_yaw = movement_manager.get_primary_target_pose()
        # Mirrored joints: (-, +) perks both antennas outward; (+, -) crosses them.
        target = (-0.5, 0.5) if awake else (0.0, 0.0)
        z_offset_mm = _SLEEP_HEAD_Z_OFFSET_MM if awake else -_SLEEP_HEAD_Z_OFFSET_MM
        head_offset = create_head_pose(0, 0, z_offset_mm, 0, 0, 0, degrees=True, mm=True)
        target_head_pose = compose_world_offset(head_pose, head_offset, reorthonormalize=True)
        move = GotoQueueMove(
            target_head_pose=target_head_pose,
            start_head_pose=head_pose,
            target_antennas=target,
            start_antennas=antennas,
            target_body_yaw=body_yaw,
            start_body_yaw=body_yaw,
            duration=0.6,
        )
        movement_manager.queue_move(move)
    except Exception:
        logger.debug("Antenna cue skipped", exc_info=True)
