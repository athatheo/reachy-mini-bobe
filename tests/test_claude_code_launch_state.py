# ruff: noqa: D103

import json

import pytest

from bobe.claude_code_client import DEFAULT_REQUEST_TIMEOUT_S
from bobe.claude_code_launch import (
    ClaudeCodeLaunchSettings,
    ClaudeCodeLaunchController,
    confirmation_phrase_matches,
    derive_launch_url_from_wake_url,
)


def test_confirmation_phrase_matches_exact_phrase_with_punctuation():
    assert confirmation_phrase_matches("Confirm launch Claude Code.")
    assert confirmation_phrase_matches("  confirm   launch claude code  ")


def test_confirmation_phrase_rejects_near_misses():
    assert not confirmation_phrase_matches("please confirm launch Claude Code")
    assert not confirmation_phrase_matches("confirm launch Claude Code now")
    assert not confirmation_phrase_matches("confirm launch code")


def test_derives_launch_url_from_wake_url():
    assert (
        derive_launch_url_from_wake_url("ws://Mac.local:8765/v1/stream")
        == "http://Mac.local:8765/v1/launch/claude-code"
    )
    assert (
        derive_launch_url_from_wake_url("wss://Mac.local:8765/v1/stream")
        == "https://Mac.local:8765/v1/launch/claude-code"
    )


def test_request_requires_robot_endpoint_and_token():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(launch_url=None, launch_token=None)
    )

    result = controller.request()

    assert result["status"] == "missing_config"
    assert controller.has_pending() is False


@pytest.mark.asyncio
async def test_confirm_posts_to_mac_endpoint():
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True, "workdir": "/tmp/repos/bobe"}).encode()

    def fake_opener(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=fake_opener,
    )
    assert controller.request()["status"] == "pending_confirmation"

    result = await controller.maybe_confirm_from_transcript("confirm launch Claude Code")

    assert result is not None
    assert result["status"] == "launched"
    assert len(calls) == 1
    request, timeout = calls[0]
    assert timeout == DEFAULT_REQUEST_TIMEOUT_S
    assert request.full_url == "http://mac.local:8765/v1/launch/claude-code"
    assert request.get_header("X-bobe-launch-token") == "launch-token"


@pytest.mark.asyncio
async def test_expired_confirmation_does_not_post():
    now = {"value": 100.0}
    calls = []
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
            confirm_ttl_s=1.0,
        ),
        clock=lambda: now["value"],
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    controller.request()
    now["value"] = 102.0

    result = await controller.maybe_confirm_from_transcript("confirm launch Claude Code")

    assert result is not None
    assert result["status"] == "expired"
    assert calls == []
