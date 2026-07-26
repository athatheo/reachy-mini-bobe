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


def test_command_confirmation_phrase_matches_common_asr_outputs():
    assert command_confirmation_phrase_matches("Confirm, Claude command.")
    assert command_confirmation_phrase_matches("Confirm cloud command.")
    assert command_confirmation_phrase_matches("confirm clod command")
    assert command_confirmation_phrase_matches("Confirmed Claude command.")


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


def _settings():
    return ClaudeCodeSessionSettings(
        base_url="http://mac.local:8765/v1/claude-code",
        token="control-token",
    )


class _JsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


@pytest.mark.asyncio
async def test_accepted_command_polls_status_until_result():
    """A 202-style accepted send must poll status and speak the finished result."""
    status_payloads = iter(
        [
            {"ok": True, "active": True, "session_id": "s1", "running": True, "last_result": None},
            {
                "ok": True,
                "active": True,
                "session_id": "s1",
                "running": False,
                "last_result": {"ok": True, "output": "done", "session_id": "s1"},
            },
        ]
    )

    def fake_opener(request, *, timeout):
        if request.full_url.endswith("/session/send"):
            return _JsonResponse({"ok": True, "accepted": True, "running": True, "session_id": "s1"})
        assert request.full_url.endswith("/session/status")
        return _JsonResponse(next(status_payloads))

    now = {"value": 0.0}

    async def fake_sleep(seconds):
        now["value"] += seconds

    controller = ClaudeCodeSessionController(
        settings_loader=_settings,
        clock=lambda: now["value"],
        opener=fake_opener,
        sleeper=fake_sleep,
    )
    controller.request_send("run tests")

    result = await controller.maybe_confirm_from_transcript("confirm Claude command")

    assert result is not None
    assert result["status"] == "sent"
    assert result["result"]["output"] == "done"


@pytest.mark.asyncio
async def test_accepted_command_still_running_reports_running_not_failure():
    """Long commands must be reported as running, never as a false failure."""

    def fake_opener(request, *, timeout):
        if request.full_url.endswith("/session/send"):
            return _JsonResponse({"ok": True, "accepted": True, "running": True, "session_id": "s1"})
        return _JsonResponse(
            {"ok": True, "active": True, "session_id": "s1", "running": True, "last_result": None}
        )

    now = {"value": 0.0}

    async def fake_sleep(seconds):
        now["value"] += seconds

    controller = ClaudeCodeSessionController(
        settings_loader=_settings,
        clock=lambda: now["value"],
        opener=fake_opener,
        sleeper=fake_sleep,
    )
    controller.request_send("refactor the config module")

    result = await controller.maybe_confirm_from_transcript("confirm Claude command")

    assert result is not None
    assert result["status"] == "running"
    assert "working on" in result["message"]


@pytest.mark.asyncio
async def test_accepted_command_reports_stopped_session():
    def fake_opener(request, *, timeout):
        if request.full_url.endswith("/session/send"):
            return _JsonResponse({"ok": True, "accepted": True, "running": True, "session_id": "s1"})
        return _JsonResponse({"ok": True, "active": False, "session_id": None, "running": False, "last_result": None})

    now = {"value": 0.0}

    async def fake_sleep(seconds):
        now["value"] += seconds

    controller = ClaudeCodeSessionController(
        settings_loader=_settings,
        clock=lambda: now["value"],
        opener=fake_opener,
        sleeper=fake_sleep,
    )
    controller.request_send("run tests")

    result = await controller.maybe_confirm_from_transcript("confirm Claude command")

    assert result is not None
    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_garbled_command_confirmation_prompts_correction_and_keeps_pending():
    controller = ClaudeCodeSessionController(
        settings_loader=_settings,
        opener=lambda *args, **kwargs: pytest.fail("must not post on a mismatch"),
    )
    controller.request_send("run tests")

    result = await controller.maybe_confirm_from_transcript("Confirm the Claude command.")

    assert result is not None
    assert result["status"] == "confirmation_mismatch"
    assert "confirm claude command" in result["message"]
    assert controller.has_pending() is True


@pytest.mark.asyncio
async def test_garbled_command_attempt_is_silent_when_mismatch_correction_disabled():
    """With correct_mismatch=False only the exact command phrase produces a result.

    Orchestrators coordinating several pending confirmations disable the
    corrective reply here so it can never shadow another controller's
    exactly-spoken confirmation phrase.
    """
    controller = ClaudeCodeSessionController(
        settings_loader=_settings,
        opener=lambda *args, **kwargs: pytest.fail("must not post on a mismatch"),
    )
    controller.request_send("run tests")

    result = await controller.maybe_confirm_from_transcript(
        "Confirm the Claude command.", correct_mismatch=False
    )

    assert result is None
    assert controller.has_pending() is True


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
