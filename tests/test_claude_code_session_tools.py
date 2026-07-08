# ruff: noqa: D103

from types import SimpleNamespace

import pytest

from bobe.claude_code_session import (
    ClaudeCodeSessionSettings,
    ClaudeCodeSessionController,
    reset_claude_code_session_controller,
)
from bobe.profiles._bobe_locked_profile.get_claude_code_status import GetClaudeCodeStatus
from bobe.profiles._bobe_locked_profile.send_claude_code_command import SendClaudeCodeCommand
from bobe.profiles._bobe_locked_profile.stop_claude_code_session import StopClaudeCodeSession
from bobe.profiles._bobe_locked_profile.start_claude_code_session import StartClaudeCodeSession


@pytest.fixture(autouse=True)
def reset_controller():
    yield
    reset_claude_code_session_controller()


@pytest.mark.asyncio
async def test_send_tool_stages_command_without_posting():
    calls = []
    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    reset_claude_code_session_controller(controller)

    result = await SendClaudeCodeCommand()(SimpleNamespace(), command="run tests")

    assert result["status"] == "pending_confirmation"
    assert result["confirmation_phrase"] == "confirm claude command"
    assert controller.has_pending() is True
    assert calls == []


@pytest.mark.asyncio
async def test_session_tools_call_controller(monkeypatch):
    class FakeController:
        async def start(self):
            return {"ok": True, "action": "start"}

        async def status(self):
            return {"ok": True, "action": "status"}

        async def stop(self):
            return {"ok": True, "action": "stop"}

    reset_claude_code_session_controller(FakeController())  # type: ignore[arg-type]

    assert (await StartClaudeCodeSession()(SimpleNamespace()))["action"] == "start"
    assert (await GetClaudeCodeStatus()(SimpleNamespace()))["action"] == "status"
    assert (await StopClaudeCodeSession()(SimpleNamespace()))["action"] == "stop"
