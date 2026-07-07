# ruff: noqa: D103

from types import SimpleNamespace

import pytest

from bobe.claude_code_launch import (
    ClaudeCodeLaunchSettings,
    ClaudeCodeLaunchController,
    reset_claude_code_launch_controller,
)
from bobe.profiles._bobe_locked_profile.launch_claude_code import LaunchClaudeCode


@pytest.fixture(autouse=True)
def reset_controller():
    yield
    reset_claude_code_launch_controller()


@pytest.mark.asyncio
async def test_launch_tool_creates_pending_request():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        )
    )
    reset_claude_code_launch_controller(controller)

    result = await LaunchClaudeCode()(SimpleNamespace(), action="request")

    assert result["status"] == "pending_confirmation"
    assert result["confirmation_phrase"] == "confirm launch claude code"
    assert controller.has_pending() is True


@pytest.mark.asyncio
async def test_launch_tool_cancels_pending_request():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        )
    )
    reset_claude_code_launch_controller(controller)
    controller.request()

    result = await LaunchClaudeCode()(SimpleNamespace(), action="cancel")

    assert result["status"] == "cancelled"
    assert controller.has_pending() is False


@pytest.mark.asyncio
async def test_launch_tool_reports_missing_config():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(launch_url=None, launch_token=None)
    )
    reset_claude_code_launch_controller(controller)

    result = await LaunchClaudeCode()(SimpleNamespace(), action="request")

    assert result["status"] == "missing_config"
    assert controller.has_pending() is False
