# ruff: noqa: D103

import json

import pytest

from bobe.claude_code_session import (
    ClaudeCodeSessionSettings,
    ClaudeCodeSessionController,
    derive_control_url_from_wake_url,
    command_confirmation_phrase_matches,
)


def test_command_confirmation_phrase_matches_exact_phrase():
    assert command_confirmation_phrase_matches("Confirm Claude command.")
    assert command_confirmation_phrase_matches(" confirm   claude command ")


def test_command_confirmation_phrase_rejects_near_misses():
    assert not command_confirmation_phrase_matches("please confirm Claude command")
    assert not command_confirmation_phrase_matches("confirm Claude command now")


def test_derives_control_url_from_wake_url():
    assert derive_control_url_from_wake_url("ws://Mac.local:8765/v1/stream") == "http://Mac.local:8765/v1/claude-code"
    assert (
        derive_control_url_from_wake_url("wss://Mac.local:8765/v1/stream") == "https://Mac.local:8765/v1/claude-code"
    )


def test_request_send_requires_config():
    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(base_url=None, token=None)
    )

    result = controller.request_send("run tests")

    assert result["status"] == "missing_config"
    assert controller.has_pending() is False


@pytest.mark.asyncio
async def test_start_posts_to_session_endpoint():
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True, "session_id": "session-1"}).encode()

    def fake_opener(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=fake_opener,
    )

    result = await controller.start()

    assert result["ok"] is True
    request, timeout = calls[0]
    assert request.full_url == "http://mac.local:8765/v1/claude-code/session/start"
    assert request.get_method() == "POST"
    assert request.get_header("X-bobe-launch-token") == "control-token"
    assert timeout == 10.0


@pytest.mark.asyncio
async def test_confirmed_command_posts_to_send_endpoint():
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True, "output": "done"}).encode()

    def fake_opener(request, *, timeout):
        calls.append(request)
        return FakeResponse()

    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=fake_opener,
    )
    controller.request_send("run tests")

    result = await controller.maybe_confirm_from_transcript("confirm Claude command")

    assert result is not None
    assert result["status"] == "sent"
    request = calls[0]
    assert request.full_url == "http://mac.local:8765/v1/claude-code/session/send"
    assert json.loads(request.data.decode()) == {"command": "run tests"}


@pytest.mark.asyncio
async def test_expired_command_confirmation_does_not_post():
    now = {"value": 10.0}
    calls = []
    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
            confirm_ttl_s=1.0,
        ),
        clock=lambda: now["value"],
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    controller.request_send("run tests")
    now["value"] = 12.0

    result = await controller.maybe_confirm_from_transcript("confirm Claude command")

    assert result is not None
    assert result["status"] == "expired"
    assert calls == []
