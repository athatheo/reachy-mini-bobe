import logging
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


class SweepLook(Tool):
    """Sweep head from left to right and back to center, pausing at each position."""

    name = "sweep_look"
    description = "Sweep head from left to right while rotating the body, pausing at each extreme, then return to center"
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Sweep amplitude from center (radians).
    MAX_ANGLE_RAD = 0.9 * np.pi
    # Upper bound on the yaw covered by a single interpolation leg. Slerp
    # (linear_pose_interpolation) always takes the MINIMAL rotation, so any leg
    # spanning more than pi would wrap the wrong way and fight the linearly
    # interpolated body yaw; keeping legs <= pi/2 leaves a comfortable margin.
    MAX_LEG_ANGLE_RAD = np.pi / 2

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute sweep look: center -> left -> hold -> center -> right -> hold -> center."""
        logger.info("Tool call: sweep_look")

        # Clear any existing moves
        deps.movement_manager.clear_move_queue()

        # Get current state
        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()

        # Extract body_yaw from head joints (first element of the 7 head joint positions)
        current_body_yaw = float(head_joints[0])
        antennas = (float(antenna_joints[0]), float(antenna_joints[1]))

        # Define sweep parameters
        max_angle = self.MAX_ANGLE_RAD
        transition_duration = 3.0  # Time to move between center and an extreme
        hold_duration = 1.0  # Time to hold at each extreme
        center_duration = 1.5  # Time to recenter from the arbitrary current pose

        def yaw_pose(yaw: float) -> NDArray[np.float64]:
            return create_head_pose(0, 0, 0, 0, 0, yaw, degrees=False)

        def yaw_leg(start_yaw: float, target_yaw: float, duration: float) -> GotoQueueMove:
            """One sweep leg with head and body yaw moving in lockstep."""
            return GotoQueueMove(
                target_head_pose=yaw_pose(target_yaw),
                start_head_pose=yaw_pose(start_yaw),
                target_antennas=antennas,
                start_antennas=antennas,
                target_body_yaw=target_yaw,
                start_body_yaw=start_yaw,
                duration=duration,
            )

        def sweep_legs(start_yaw: float, target_yaw: float, total_duration: float) -> List[GotoQueueMove]:
            """Split a sweep into bounded legs so slerp can never wrap the wrong way."""
            span = target_yaw - start_yaw
            steps = max(1, int(np.ceil(abs(span) / self.MAX_LEG_ANGLE_RAD)))
            leg_duration = total_duration / steps
            return [
                yaw_leg(
                    start_yaw + span * step / steps,
                    start_yaw + span * (step + 1) / steps,
                    leg_duration,
                )
                for step in range(steps)
            ]

        moves: List[GotoQueueMove] = []

        # Move 0: recenter from the arbitrary current pose (minimal rotation,
        # always < pi) so every sweep leg below starts from a known waypoint
        # with head and body yaw both at 0.
        moves.append(
            GotoQueueMove(
                target_head_pose=yaw_pose(0.0),
                start_head_pose=current_head_pose,
                target_antennas=antennas,
                start_antennas=antennas,
                target_body_yaw=0.0,
                start_body_yaw=current_body_yaw,
                duration=center_duration,
            ),
        )

        # Sweep to the left in bounded legs, hold, then return to center.
        moves.extend(sweep_legs(0.0, max_angle, transition_duration))
        moves.append(yaw_leg(max_angle, max_angle, hold_duration))
        moves.extend(sweep_legs(max_angle, 0.0, transition_duration))

        # Sweep to the right in bounded legs, hold, then return to center.
        moves.extend(sweep_legs(0.0, -max_angle, transition_duration))
        moves.append(yaw_leg(-max_angle, -max_angle, hold_duration))
        moves.extend(sweep_legs(-max_angle, 0.0, transition_duration))

        # Queue all moves in sequence
        for move in moves:
            deps.movement_manager.queue_move(move)

        total_duration = center_duration + transition_duration * 4 + hold_duration * 2
        return {"status": f"sweeping look left-right-center, total {total_duration:.1f}s"}
