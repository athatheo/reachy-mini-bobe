"""Coordinates spoken confirmations across the Claude Code launch and command controllers.

The realtime handler forwards each completed user transcript here; this module
owns the pure coordination policy between :mod:`bobe.claude_code_launch` and
:mod:`bobe.claude_code_session` and returns the reply to surface, keeping the
policy out of the realtime audio handler.
"""

from __future__ import annotations
from typing import Any

from bobe.claude_code_client import transcript_attempts_confirmation
from bobe.claude_code_launch import (
    maybe_confirm_claude_code_launch,
    pending_launch_confirmation_instruction,
)
from bobe.claude_code_session import (
    maybe_confirm_claude_code_command,
    pending_command_confirmation_instruction,
)


async def resolve_confirmation(transcript: str) -> dict[str, Any] | None:
    """Resolve a completed transcript against any pending Claude Code confirmations.

    Exact matches on BOTH controllers run first: with a launch and a command
    pending at once, one controller's corrective mismatch reply must never
    intercept the other's exactly-spoken phrase. Only after both exact matches
    fail is a garbled attempt answered with a single correction.
    """
    result = await maybe_confirm_claude_code_launch(transcript, correct_mismatch=False)
    if result is not None:
        return result
    result = await maybe_confirm_claude_code_command(transcript, correct_mismatch=False)
    if result is not None:
        return result
    return _correct_garbled_attempt(transcript)


def _correct_garbled_attempt(transcript: str) -> dict[str, Any] | None:
    """Build a corrective reply for a garbled confirmation attempt.

    Runs only once the transcript matched NEITHER exact confirmation phrase,
    and mentions every phrase still pending so the user knows what to repeat
    even when a launch and a command are pending at once.
    """
    if not transcript_attempts_confirmation(transcript):
        return None
    instructions = [
        instruction
        for instruction in (
            pending_launch_confirmation_instruction(),
            pending_command_confirmation_instruction(),
        )
        if instruction
    ]
    if not instructions:
        return None
    message = " ".join(("That wasn't the exact confirmation phrase.", *instructions))
    return {"status": "confirmation_mismatch", "message": message}
