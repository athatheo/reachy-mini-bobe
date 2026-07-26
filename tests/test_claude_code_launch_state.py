# ruff: noqa: D103

import json
import http.client

import pytest

from bobe.claude_code_client import DEFAULT_REQUEST_TIMEOUT_S, request_daemon_json
from bobe.claude_code_launch import (
    ClaudeCodeLaunchSettings,
    ClaudeCodeLaunchController,
    confirmation_phrase_matches,
    derive_launch_url_from_wake_url,
)


def test_confirmation_phrase_matches_exact_phrase_with_punctuation():
    assert confirmation_phrase_matches("Confirm launch Claude Code.")
    assert confirmation_phrase_matches("  confirm   launch claude code  ")


def test_confirmation_phrase_matches_common_asr_outputs():
    # gpt-4o-transcribe routinely inserts internal punctuation.
    assert confirmation_phrase_matches("Confirm, launch Claude Code.")
    assert confirmation_phrase_matches("Confirm launch, Claude Code.")
    assert confirmation_phrase_matches("Confirm. Launch Claude Code.")
    assert confirmation_phrase_matches("Confirm launch Claude-Code.")
    # Well-known claude/confirm mishears.
    assert confirmation_phrase_matches("Confirm launch cloud code.")
    assert confirmation_phrase_matches("confirm launch clod code")
    assert confirmation_phrase_matches("Confirmed launch Claude Code.")


def test_confirmation_phrase_rejects_near_misses():
    assert not confirmation_phrase_matches("please confirm launch Claude Code")
    assert not confirmation_phrase_matches("confirm launch Claude Code now")
    assert not confirmation_phrase_matches("confirm launch code")


class _BrokenReadResponse:
    def __init__(self, exc: Exception):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [ConnectionResetError("peer reset"), http.client.IncompleteRead(b"partial")],
)
def test_request_daemon_json_maps_body_read_failures_to_error_dict(exc):
    result = request_daemon_json(
        lambda request, *, timeout: _BrokenReadResponse(exc),
        url="http://mac.local:8765/v1/launch/claude-code",
        token="launch-token",
        method="POST",
        payload={},
        timeout_s=1.0,
        log_label="test",
    )

    assert result == {"ok": False, "error": "endpoint_error"}


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
async def test_garbled_confirmation_attempt_prompts_correction_and_keeps_pending():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=lambda *args, **kwargs: pytest.fail("must not post on a mismatch"),
    )
    controller.request()

    result = await controller.maybe_confirm_from_transcript("Confirm launching Claude Code.")

    assert result is not None
    assert result["status"] == "confirmation_mismatch"
    assert "confirm launch claude code" in result["message"]
    assert controller.has_pending() is True


@pytest.mark.asyncio
async def test_garbled_attempt_is_silent_when_mismatch_correction_disabled():
    """With correct_mismatch=False only exact phrases produce a result.

    Orchestrators coordinating several pending confirmations disable the
    corrective reply here so it can never shadow another controller's
    exactly-spoken confirmation phrase.
    """
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=lambda *args, **kwargs: pytest.fail("must not post on a mismatch"),
    )
    controller.request()

    result = await controller.maybe_confirm_from_transcript(
        "Confirm launching Claude Code.", correct_mismatch=False
    )

    assert result is None
    assert controller.has_pending() is True


@pytest.mark.asyncio
async def test_unrelated_transcript_is_not_consumed_while_pending():
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        )
    )
    controller.request()

    assert await controller.maybe_confirm_from_transcript("what's the weather like?") is None
    assert controller.has_pending() is True


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
