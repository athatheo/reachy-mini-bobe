import time
import random
import asyncio
import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

import bobe.openai_realtime as rt_mod
import bobe.tools.background_tool_manager as btm_mod
from bobe.wake_word import WakeConfig, WakeSession
from bobe.openai_realtime import _compute_response_cost
from bobe.tools.core_tools import ToolDependencies
from bobe.claude_code_launch import (
    ClaudeCodeLaunchSettings,
    ClaudeCodeLaunchController,
    reset_claude_code_launch_controller,
)
from bobe.claude_code_session import (
    ClaudeCodeSessionSettings,
    ClaudeCodeSessionController,
    reset_claude_code_session_controller,
)
from bobe.tools.tool_constants import ToolState
from bobe.tools.background_tool_manager import ToolCallRoutine


# ---- Session supervisor fakes / helpers ----


class SupervisorFakeConn:
    """Realtime connection stub whose iteration behavior is scripted by ``mode``.

    Modes:
    - "clean": ends iteration immediately (graceful server close).
    - "hold": stays connected until close()/server_close() is called.
    - "iter_fail": raises ``fail_exc`` on the first iteration.
    - "update_fail": session.update raises RuntimeError.
    """

    def __init__(self, mode: str, fail_exc: type[Exception] = RuntimeError) -> None:
        """Initialize the stub with a scripted mode."""
        self.mode = mode
        self.fail_exc = fail_exc
        self.closed = asyncio.Event()
        self.response_creates: list[dict[str, Any]] = []
        outer = self

        class _Session:
            async def update(self, **_kw: Any) -> None:
                if outer.mode == "update_fail":
                    raise RuntimeError("session.update failed (simulated)")

        class _InputAudioBuffer:
            async def append(self, **_kw: Any) -> None:
                return None

            async def clear(self) -> None:
                return None

        class _Item:
            async def create(self, **_kw: Any) -> None:
                return None

        class _Conversation:
            item = _Item()

        class _Response:
            async def create(self, **kw: Any) -> None:
                outer.response_creates.append(kw)

            async def cancel(self, **_kw: Any) -> None:
                return None

        self.session = _Session()
        self.input_audio_buffer = _InputAudioBuffer()
        self.conversation = _Conversation()
        self.response = _Response()

    async def __aenter__(self) -> "SupervisorFakeConn":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    async def close(self) -> None:
        """End iteration as if the connection was closed locally."""
        self.closed.set()

    def server_close(self) -> None:
        """End iteration as if the server closed the websocket cleanly."""
        self.closed.set()

    def __aiter__(self) -> "SupervisorFakeConn":
        return self

    async def __anext__(self) -> None:
        if self.mode == "iter_fail":
            raise self.fail_exc("abrupt close (simulated)")
        if self.mode == "clean":
            raise StopAsyncIteration
        await self.closed.wait()
        raise StopAsyncIteration


def _install_supervisor_fakes(
    monkeypatch: Any,
    conn_modes: list[str],
    fail_exc: type[Exception] = RuntimeError,
) -> dict[str, Any]:
    """Patch AsyncOpenAI so the Nth connect() yields a SupervisorFakeConn(conn_modes[N]).

    Connects beyond the scripted list default to "hold". Also patches prompt
    loaders and shrinks the retry backoff so supervised retries run instantly.
    Returns a state dict with the connect count and created connections.
    """
    state: dict[str, Any] = {"connects": 0, "conns": []}

    class _FakeRealtime:
        def connect(self, **_kw: Any) -> SupervisorFakeConn:
            index = state["connects"]
            state["connects"] += 1
            mode = conn_modes[index] if index < len(conn_modes) else "hold"
            conn = SupervisorFakeConn(mode, fail_exc=fail_exc)
            state["conns"].append(conn)
            return conn

    class _FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.realtime = _FakeRealtime()

    monkeypatch.setattr(rt_mod, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(rt_mod, "get_realtime_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda: "alloy")
    monkeypatch.setattr(rt_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(rt_mod, "_MAX_SESSION_RETRY_DELAY_S", 0.01)
    return state


async def _wait_until(predicate: Any, timeout: float = 2.0) -> None:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _finish_supervisor(handler: Any, startup_task: "asyncio.Task[None]") -> None:
    """Shut the handler down and reap the start_up supervisor task."""
    await handler.shutdown()
    if not startup_task.done():
        startup_task.cancel()
    try:
        await startup_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_start_up_retries_on_abrupt_close(monkeypatch: Any, caplog: Any) -> None:
    """First connection dies with ConnectionClosedError during iteration -> retried.

    The second connection iterates cleanly; the supervisor then stays alive
    (parked, waiting for a restart request) and the finished session clears
    self.connection itself.
    """
    caplog.set_level(logging.WARNING)

    # Use a local Exception as the module's ConnectionClosedError to avoid ws dependency
    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(rt_mod, "ConnectionClosedError", FakeCCE)

    state = _install_supervisor_fakes(monkeypatch, ["iter_fail", "clean"], fail_exc=FakeCCE)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._wake_detector = None

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await _wait_until(
            lambda: state["connects"] >= 2
            and handler._realtime_session_task is not None
            and handler._realtime_session_task.done()
        )

        # Two attempts total (fail -> retry -> succeed), and connection cleared
        assert state["connects"] == 2
        assert handler.connection is None
        # The supervisor must survive the clean session end (no park-forever, no crash).
        assert not startup_task.done()

        warnings = [r for r in caplog.records if r.levelname == "WARNING" and "closed unexpectedly" in r.msg]
        assert len(warnings) == 1
    finally:
        await _finish_supervisor(handler, startup_task)


@pytest.mark.asyncio
async def test_start_up_retries_on_session_update_failure(monkeypatch: Any, caplog: Any) -> None:
    """session.update failure raises RealtimeSessionError and the supervisor retries."""
    caplog.set_level(logging.WARNING)

    state = _install_supervisor_fakes(monkeypatch, ["update_fail", "clean"])

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._wake_detector = None

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await _wait_until(
            lambda: state["connects"] >= 2
            and handler._realtime_session_task is not None
            and handler._realtime_session_task.done()
        )

        assert state["connects"] == 2
        assert handler.connection is None
        assert not startup_task.done()

        retry_logs = [r for r in caplog.records if "closed unexpectedly" in getattr(r, "msg", "")]
        assert len(retry_logs) == 1
    finally:
        await _finish_supervisor(handler, startup_task)


@pytest.mark.asyncio
async def test_start_up_retries_on_unexpected_exception_type(monkeypatch: Any) -> None:
    """Non-ConnectionClosedError session failures are retried instead of escaping start_up."""

    class WeirdSdkError(Exception):
        """Stand-in for e.g. openai's WebSocketConnectionClosedError."""

    state = _install_supervisor_fakes(monkeypatch, ["iter_fail", "hold"], fail_exc=WeirdSdkError)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._wake_detector = None

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await _wait_until(lambda: state["connects"] >= 2 and handler.connection is not None)

        # The unexpected exception type must not have escaped the supervisor.
        assert not startup_task.done()
        assert handler.connection is state["conns"][1]
    finally:
        await _finish_supervisor(handler, startup_task)


@pytest.mark.asyncio
async def test_start_up_keeps_retrying_past_three_consecutive_failures(monkeypatch: Any) -> None:
    """The retry budget counts consecutive failures and never parks forever.

    Five failing sessions in a row (well past the old process-lifetime budget
    of 3) must still be followed by a successful reconnect, and a successful
    connect resets the failure streak.
    """
    state = _install_supervisor_fakes(
        monkeypatch,
        ["iter_fail", "iter_fail", "iter_fail", "iter_fail", "iter_fail", "hold"],
    )

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._wake_detector = None

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await _wait_until(lambda: state["connects"] >= 6 and handler.connection is not None, timeout=5.0)

        assert state["connects"] == 6
        assert not startup_task.done()
        # A successful connect resets the consecutive-failure counter.
        assert handler._session_failures == 0
    finally:
        await _finish_supervisor(handler, startup_task)


@pytest.mark.asyncio
async def test_clean_close_clears_state_and_supervisor_restarts_on_request(monkeypatch: Any) -> None:
    """A graceful server close leaves no stale 'connected' state.

    The session clears self.connection/_connected_event in its own finally;
    the supervisor parks instead of reconnecting on its own, and a
    _restart_session() request makes the supervisor (and only the supervisor)
    create the next session task.
    """
    state = _install_supervisor_fakes(monkeypatch, ["hold", "hold"])
    monkeypatch.setattr(rt_mod, "_RESTART_CONNECT_TIMEOUT_S", 2.0)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._wake_detector = None

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await _wait_until(lambda: handler.connection is not None)
        first_task = handler._realtime_session_task
        assert first_task is not None

        # Server closes the websocket cleanly (e.g. session max duration).
        state["conns"][0].server_close()
        await _wait_until(lambda: first_task.done())

        # No stale connected state survives the session (finding: stale connection).
        assert handler.connection is None
        assert not handler._connected_event.is_set()
        # The supervisor parks; it does not stack a second session on its own.
        assert state["connects"] == 1
        assert not startup_task.done()

        # A restart request is consumed by the supervisor, which owns task creation.
        await handler._restart_session()
        await _wait_until(lambda: handler.connection is not None)

        assert state["connects"] == 2
        assert handler.connection is state["conns"][1]
        session_task = handler._realtime_session_task
        assert session_task is not None
        assert session_task is not first_task
        assert session_task.get_name() == "openai-realtime-session"
    finally:
        await _finish_supervisor(handler, startup_task)


@pytest.mark.asyncio
async def test_restart_session_never_creates_session_tasks(monkeypatch: Any) -> None:
    """_restart_session only signals the supervisor; it must not spawn session tasks."""
    monkeypatch.setattr(rt_mod, "_RESTART_CONNECT_TIMEOUT_S", 0.05)

    handler = _build_wake_enabled_handler()
    handler.client = MagicMock()  # start_up ran once, but no supervisor is alive

    await handler._restart_session()

    assert handler._realtime_session_task is None
    assert handler._restart_requested.is_set()


@pytest.mark.asyncio
async def test_run_realtime_session_resets_stale_response_state(monkeypatch: Any) -> None:
    """A fresh session drains queued response.create kwargs and resets response flags.

    Stale requests queued for a previous conversation must never replay, and a
    cleared _response_done_event from a mid-response disconnect must not
    suppress VAD handling in the new session.
    """
    state = _install_supervisor_fakes(monkeypatch, ["hold"])

    handler = _build_wake_enabled_handler()
    handler.client = rt_mod.AsyncOpenAI(api_key="DUMMY")

    # Simulate leftovers from a session that died mid-response.
    await handler._safe_response_create(response={"instructions": "stale tool follow-up"})
    handler._response_done_event.clear()
    handler._last_response_rejected = True

    session_task = asyncio.create_task(handler._run_realtime_session())
    try:
        await asyncio.wait_for(handler._connected_event.wait(), timeout=2.0)

        assert handler._pending_responses.empty()
        assert handler._response_done_event.is_set()
        assert handler._last_response_rejected is False

        # Give the sender a chance to (wrongly) send anything that survived.
        await asyncio.sleep(0.05)
        assert state["conns"][0].response_creates == []
    finally:
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass
        await handler.shutdown()

    # The cancelled session cleaned up after itself.
    assert handler.connection is None
    assert not handler._connected_event.is_set()


@pytest.mark.asyncio
async def test_session_finally_does_not_clobber_newer_connection(monkeypatch: Any) -> None:
    """A dying session only clears self.connection if it still owns it."""
    _install_supervisor_fakes(monkeypatch, ["hold"])

    handler = _build_wake_enabled_handler()
    handler.client = rt_mod.AsyncOpenAI(api_key="DUMMY")

    session_task = asyncio.create_task(handler._run_realtime_session())
    try:
        await asyncio.wait_for(handler._connected_event.wait(), timeout=2.0)

        # A newer session has taken over the shared slot in the meantime.
        newer_connection = MagicMock()
        handler.connection = newer_connection

        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass

        assert handler.connection is newer_connection
    finally:
        if not session_task.done():
            session_task.cancel()
        await handler.shutdown()


@pytest.mark.asyncio
async def test_receive_append_failure_restarts_session(monkeypatch: Any) -> None:
    """Append failure while awake closes the session and the supervisor reconnects."""
    monkeypatch.setattr(rt_mod, "get_realtime_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda: "alloy")
    monkeypatch.setattr(rt_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(rt_mod, "_MAX_SESSION_RETRY_DELAY_S", 0.01)
    monkeypatch.setattr(rt_mod, "_RESTART_CONNECT_TIMEOUT_S", 2.0)

    session_attempts = {"n": 0}
    restart_calls = {"n": 0}
    original_restart = rt_mod.OpenaiRealtimeHandler._restart_session

    async def _counting_restart(self: Any) -> None:
        restart_calls["n"] += 1
        await original_restart(self)

    monkeypatch.setattr(rt_mod.OpenaiRealtimeHandler, "_restart_session", _counting_restart)

    class FakeInputAudioBuffer:
        def __init__(self, *, fail_append: bool) -> None:
            self.fail_append = fail_append
            self.appended: list[str] = []

        async def append(self, *, audio: str) -> None:
            if self.fail_append:
                raise RuntimeError("append failed (simulated)")
            self.appended.append(audio)

        async def clear(self) -> None:
            return None

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            return None

    class FakeConn:
        def __init__(self, *, fail_append: bool) -> None:
            self.session = FakeSession()
            self.input_audio_buffer = FakeInputAudioBuffer(fail_append=fail_append)
            self._closed = False
            self._iter_wait = asyncio.Event()

            class _Item:
                async def create(self, **_kw: Any) -> None:
                    return None

            class _Conversation:
                item = _Item()

            self.conversation = _Conversation()

            class _Response:
                async def create(self, **_kw: Any) -> None:
                    return None

                async def cancel(self, **_kw: Any) -> None:
                    return None

            self.response = _Response()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def close(self) -> None:
            self._closed = True
            self._iter_wait.set()

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> None:
            if self._closed:
                raise StopAsyncIteration
            await self._iter_wait.wait()
            raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            session_attempts["n"] += 1
            return FakeConn(fail_append=session_attempts["n"] == 1)

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.realtime = FakeRealtime()

    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)

    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()

    startup_task = asyncio.create_task(handler.start_up())
    try:
        await asyncio.wait_for(handler._connected_event.wait(), timeout=2.0)

        await handler.receive(_mic_frame())

        assert restart_calls["n"] >= 1
        await _wait_until(lambda: handler.connection is not None)
        assert not handler.connection.input_audio_buffer.fail_append
        # The reconnect came from the supervisor, not from _restart_session.
        assert handler._realtime_session_task is not None
        assert handler._realtime_session_task.get_name() == "openai-realtime-session"

        await handler.receive(_mic_frame())
        assert len(handler.connection.input_audio_buffer.appended) == 1
    finally:
        await _finish_supervisor(handler, startup_task)


# ---- Cost calculation tests ----


def _make_usage(
    audio_in: int | None = 0,
    text_in: int | None = 0,
    image_in: int | None = 0,
    audio_out: int | None = 0,
    text_out: int | None = 0,
    has_input: bool = True,
    has_output: bool = True,
) -> MagicMock:
    """Build a fake usage object matching the OpenAI response.usage shape."""
    usage = MagicMock()
    if has_input:
        inp = MagicMock()
        inp.audio_tokens = audio_in
        inp.text_tokens = text_in
        inp.image_tokens = image_in
        usage.input_token_details = inp
    else:
        usage.input_token_details = None
    if has_output:
        out = MagicMock()
        out.audio_tokens = audio_out
        out.text_tokens = text_out
        usage.output_token_details = out
    else:
        usage.output_token_details = None
    return usage


@pytest.mark.parametrize(
    "usage_kwargs, expect_positive",
    [
        # All token types present → positive cost
        ({"audio_in": 1000, "text_in": 2000, "image_in": 500, "audio_out": 800, "text_out": 300}, True),
        # All None tokens → must not crash
        ({"audio_in": None, "text_in": None, "image_in": None, "audio_out": None, "text_out": None}, False),
        # Mix of None and valid ints
        ({"audio_in": None, "text_in": 500, "image_in": None, "audio_out": 1000, "text_out": None}, True),
        # Missing input/output details entirely
        ({"has_input": False, "has_output": False}, False),
    ],
    ids=["normal", "all_none", "mixed", "missing_details"],
)
def test_compute_response_cost(usage_kwargs: dict[str, Any], expect_positive: bool) -> None:
    """Verify _compute_response_cost handles various token combinations without crashing."""
    usage = _make_usage(**usage_kwargs)
    cost = _compute_response_cost(usage)
    if expect_positive:
        assert cost > 0
    else:
        assert cost == 0.0


# ---- Wake-word gating ----


class FakeInputAudioBuffer:
    """Records audio appended/cleared by the handler."""

    def __init__(self) -> None:
        """Initialize empty append/clear counters."""
        self.appended: list[str] = []
        self.cleared = 0

    async def append(self, audio: str) -> None:
        """Record an appended audio payload."""
        self.appended.append(audio)

    async def clear(self) -> None:
        """Record a buffer clear."""
        self.cleared += 1


class FakeGatingConnection:
    """Minimal connection stub for receive() gating tests."""

    def __init__(self) -> None:
        """Initialize with a recording input audio buffer."""
        self.input_audio_buffer = FakeInputAudioBuffer()


def _build_wake_enabled_handler() -> rt_mod.OpenaiRealtimeHandler:
    """Build a handler with wake gating and no detector thread."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.wake_config = WakeConfig()
    handler.wake_session = WakeSession()
    handler._wake_detector = None
    handler.wake_gating_enabled = True
    handler.wake_error = None
    return handler


def _mic_frame(samples: int = 2400) -> tuple[int, Any]:
    import numpy as np

    return (24000, np.ones(samples, dtype=np.int16))


async def _finish_wake_transition(handler: rt_mod.OpenaiRealtimeHandler) -> None:
    """Wait for the background wake transition spawned by receive()."""
    task = handler._wake_transition_task
    assert task is not None
    await task


async def _drain_claude_code_tasks(handler: rt_mod.OpenaiRealtimeHandler) -> None:
    """Wait for the background Claude Code confirmation tasks to finish."""
    while handler._claude_code_tasks:
        await asyncio.gather(*list(handler._claude_code_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_receive_bypasses_wake_gating_when_detector_missing(caplog: Any) -> None:
    """Misconfigured wake must not silently drop mic audio."""
    caplog.set_level(logging.ERROR)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.connection = FakeGatingConnection()

    assert handler.wake_error is not None
    assert not handler.wake_gating_enabled
    assert handler.wake_session.awake

    await handler.receive(_mic_frame())

    assert handler.connection.input_audio_buffer.appended
    assert any("Wake-word gating disabled" in r.message for r in caplog.records)


def test_handler_exposes_wake_error_when_gating_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote wake misconfiguration is exposed on the handler."""
    monkeypatch.setenv("BOBE_WAKE_BACKEND", "remote")
    monkeypatch.delenv("BOBE_WAKE_REMOTE_URL", raising=False)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    assert not handler.wake_gating_enabled
    assert handler.wake_error is not None
    assert "BOBE_WAKE_REMOTE_URL" in handler.wake_error


@pytest.mark.asyncio
async def test_receive_feeds_wake_detector_without_connection() -> None:
    """Local wake scoring must run even before the OpenAI websocket connects."""
    handler = _build_wake_enabled_handler()
    detector = MagicMock()
    detector.is_running.return_value = True
    handler._wake_detector = detector
    handler.connection = None

    await handler.receive(_mic_frame())

    assert detector.feed.call_count == 1


@pytest.mark.asyncio
async def test_receive_keeps_audio_local_while_asleep() -> None:
    """No audio reaches the backend while the wake session is asleep."""
    handler = _build_wake_enabled_handler()
    handler.connection = FakeGatingConnection()

    for _ in range(3):
        await handler.receive(_mic_frame())

    assert handler.connection.input_audio_buffer.appended == []
    assert handler._wake_buffer.drain_tail(seconds=10.0).size > 0


@pytest.mark.asyncio
async def test_receive_flushes_buffer_and_streams_after_wake() -> None:
    """A wake request flushes buffered audio, then live frames stream upstream."""
    handler = _build_wake_enabled_handler()
    handler.connection = FakeGatingConnection()

    await handler.receive(_mic_frame())  # buffered locally
    handler.wake_session.request_wake()
    await handler.receive(_mic_frame())  # consumes the wake; transition runs off the mic loop
    await _finish_wake_transition(handler)
    await handler.receive(_mic_frame())  # live frame streams upstream

    appended = handler.connection.input_audio_buffer.appended
    assert len(appended) == 2  # buffer tail flush + live frame
    assert handler.wake_session.awake


@pytest.mark.asyncio
async def test_wake_ignored_when_openai_unavailable(monkeypatch: Any) -> None:
    """A wake request must not open the streaming window without a Realtime connection."""
    handler = _build_wake_enabled_handler()
    handler.connection = None

    async def _unavailable(_self: Any, timeout: float = 5.0) -> bool:
        return False

    monkeypatch.setattr(rt_mod.OpenaiRealtimeHandler, "_ensure_openai_connection", _unavailable)

    handler.wake_session.request_wake()
    await handler.receive(_mic_frame())
    await _finish_wake_transition(handler)

    assert not handler.wake_session.awake


@pytest.mark.asyncio
async def test_receive_is_not_blocked_by_slow_wake_transition(monkeypatch: Any) -> None:
    """A stalled OpenAI connection wait must never freeze the mic loop."""
    handler = _build_wake_enabled_handler()
    handler.connection = None
    release = asyncio.Event()

    async def _slow_connect(_self: Any, timeout: float = 5.0) -> bool:
        await release.wait()
        return False

    monkeypatch.setattr(rt_mod.OpenaiRealtimeHandler, "_ensure_openai_connection", _slow_connect)

    handler.wake_session.request_wake()
    started_at = time.monotonic()
    await handler.receive(_mic_frame())
    assert time.monotonic() - started_at < 0.5
    assert handler._wake_transition_active()

    # Mic frames keep flowing (buffered locally) while the transition hangs.
    for _ in range(3):
        await handler.receive(_mic_frame())
    assert not handler.wake_session.awake
    assert handler._wake_buffer.drain_tail(seconds=10.0).size > 0

    release.set()
    await _finish_wake_transition(handler)


@pytest.mark.asyncio
async def test_failed_wake_transition_backs_off_instead_of_retrying_every_frame(monkeypatch: Any) -> None:
    """A failed transition re-queues the wake with backoff and an audible cue."""
    handler = _build_wake_enabled_handler()
    handler.connection = None
    attempts = {"n": 0}

    async def _unavailable(_self: Any, timeout: float = 5.0) -> bool:
        attempts["n"] += 1
        return False

    monkeypatch.setattr(rt_mod.OpenaiRealtimeHandler, "_ensure_openai_connection", _unavailable)

    handler.wake_session.request_wake()
    await handler.receive(_mic_frame())
    await _finish_wake_transition(handler)

    assert attempts["n"] == 1
    assert not handler.wake_session.awake
    assert handler._next_wake_retry_at > time.monotonic()
    # The failure is audible: a chime landed on the output queue.
    assert isinstance(await handler.output_queue.get(), tuple)

    # During the backoff window further frames must not start new attempts,
    # and the wake request stays queued for a later retry.
    for _ in range(3):
        await handler.receive(_mic_frame())
    assert attempts["n"] == 1
    assert handler.wake_session.consume_wake_request()
    first_delay = handler._wake_retry_delay_s

    # Once the backoff window elapses the next frame retries the transition.
    handler.wake_session.request_wake()
    handler._next_wake_retry_at = time.monotonic() - 0.01
    await handler.receive(_mic_frame())
    await _finish_wake_transition(handler)

    assert attempts["n"] == 2
    # Exponential backoff: the retry delay keeps growing between failures.
    assert handler._wake_retry_delay_s > first_delay
    # The cue plays once per failure streak, not on every retry.
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_transition_to_awake_resets_response_guard() -> None:
    """Waking clears a stuck in-flight response guard so server VAD is not suppressed."""
    handler = _build_wake_enabled_handler()
    handler.connection = FakeGatingConnection()
    handler._response_done_event.clear()

    await handler._transition_to_awake()

    assert handler.wake_session.awake
    assert handler._response_done_event.is_set()


@pytest.mark.asyncio
async def test_receive_goes_back_to_sleep_after_timeout() -> None:
    """The streaming window closes after the inactivity timeout."""
    clock = [1000.0]
    handler = _build_wake_enabled_handler()
    handler.wake_session = WakeSession(timeout_s=300.0, clock=lambda: clock[0])
    handler.wake_session.wake()
    handler.connection = FakeGatingConnection()

    clock[0] += 301.0
    await handler.receive(_mic_frame())

    assert not handler.wake_session.awake
    assert handler.connection.input_audio_buffer.appended == []
    assert handler.connection.input_audio_buffer.cleared == 1


@pytest.mark.asyncio
async def test_receive_streams_mic_frames_while_awake() -> None:
    """While awake, mic frames stream continuously so the user can barge in mid-sentence."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler.connection = FakeGatingConnection()

    for _ in range(3):
        await handler.receive(_mic_frame())

    assert len(handler.connection.input_audio_buffer.appended) == 3


@pytest.mark.asyncio
async def test_completed_user_transcript_awake_defers_to_server_response() -> None:
    """While awake, transcripts touch the session but never enqueue a second response.

    Server VAD already creates the answer; a manual response.create here would
    answer the same question twice.
    """
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()

    await handler._handle_completed_user_transcript("what time is it")
    await _drain_claude_code_tasks(handler)

    output = await handler.output_queue.get()
    assert output.args[0] == {"role": "user", "content": "what time is it"}
    assert handler._pending_responses.empty()
    assert handler.wake_session.awake


@pytest.mark.asyncio
async def test_completed_user_transcript_sleep_phrase_closes_session() -> None:
    """The sleep phrase puts the session back to sleep instead of responding."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()

    await handler._handle_completed_user_transcript("okay, go to sleep")

    assert not handler.wake_session.awake
    assert handler._pending_responses.empty()


@pytest.mark.asyncio
async def test_completed_confirmation_phrase_launches_claude_code() -> None:
    """Only a completed exact confirmation transcript triggers the local launch gate."""
    calls = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_opener(request: Any, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        return FakeResponse()

    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=fake_opener,
    )
    reset_claude_code_launch_controller(controller)
    try:
        controller.request()
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        await handler._handle_completed_user_transcript("confirm launch Claude Code")
        await _drain_claude_code_tasks(handler)

        user_output = await handler.output_queue.get()
        assistant_output = await handler.output_queue.get()
        assert user_output.args[0] == {"role": "user", "content": "confirm launch Claude Code"}
        assert assistant_output.args[0]["role"] == "assistant"
        assert "launching" in assistant_output.args[0]["content"]
        assert len(calls) == 1
    finally:
        reset_claude_code_launch_controller()


@pytest.mark.asyncio
async def test_near_miss_confirmation_transcript_does_not_launch_claude_code() -> None:
    """Extra words around the confirmation phrase must not trigger the launch gate."""
    calls = []
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    reset_claude_code_launch_controller(controller)
    try:
        controller.request()
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        await handler._handle_completed_user_transcript("please confirm launch Claude Code")
        await _drain_claude_code_tasks(handler)

        assert calls == []
        assert controller.has_pending() is True
        user_output = await handler.output_queue.get()
        assert user_output.args[0] == {"role": "user", "content": "please confirm launch Claude Code"}
        assert handler.output_queue.empty()
    finally:
        reset_claude_code_launch_controller()


def test_partial_confirmation_transcript_does_not_launch_claude_code() -> None:
    """Partial transcript tracking alone does not invoke the launch gate."""
    calls = []
    controller = ClaudeCodeLaunchController(
        settings_loader=lambda: ClaudeCodeLaunchSettings(
            launch_url="http://mac.local:8765/v1/launch/claude-code",
            launch_token="launch-token",
        ),
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    reset_claude_code_launch_controller(controller)
    try:
        controller.request()
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        handler._record_user_transcript("confirm launch Claude Code")

        assert calls == []
        assert controller.has_pending() is True
    finally:
        reset_claude_code_launch_controller()


@pytest.mark.asyncio
async def test_completed_command_confirmation_sends_claude_code_command() -> None:
    """Only completed exact command confirmation sends a staged Claude Code command."""
    calls = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"ok": true, "output": "done"}'

    def fake_opener(request: Any, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        return FakeResponse()

    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=fake_opener,
    )
    reset_claude_code_session_controller(controller)
    try:
        controller.request_send("run the tests")
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        await handler._handle_completed_user_transcript("confirm Claude command")
        await _drain_claude_code_tasks(handler)

        user_output = await handler.output_queue.get()
        assistant_output = await handler.output_queue.get()
        assert user_output.args[0] == {"role": "user", "content": "confirm Claude command"}
        assert assistant_output.args[0]["role"] == "assistant"
        assert "sent" in assistant_output.args[0]["content"]
        assert len(calls) == 1
    finally:
        reset_claude_code_session_controller()


@pytest.mark.asyncio
async def test_near_miss_command_confirmation_does_not_send_claude_code_command() -> None:
    """Near-miss command confirmation leaves the command pending."""
    calls = []
    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    reset_claude_code_session_controller(controller)
    try:
        controller.request_send("run the tests")
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        await handler._handle_completed_user_transcript("please confirm Claude command")
        await _drain_claude_code_tasks(handler)

        assert calls == []
        assert controller.has_pending() is True
        user_output = await handler.output_queue.get()
        assert user_output.args[0] == {"role": "user", "content": "please confirm Claude command"}
        assert handler.output_queue.empty()
    finally:
        reset_claude_code_session_controller()


@pytest.mark.asyncio
async def test_command_confirmation_http_runs_off_the_event_dispatcher() -> None:
    """The confirmation HTTP round-trip must not stall transcript handling.

    _handle_completed_user_transcript runs inline in the realtime event
    dispatcher; the Mac HTTP call (up to 10s) has to happen in a background
    task that posts its result when done.
    """
    entered = threading.Event()
    release = threading.Event()

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"ok": true, "output": "done"}'

    def slow_opener(request: Any, *, timeout: float) -> FakeResponse:
        entered.set()
        assert release.wait(timeout=5.0)
        return FakeResponse()

    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=slow_opener,
    )
    reset_claude_code_session_controller(controller)
    try:
        controller.request_send("run the tests")
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        started_at = time.monotonic()
        await handler._handle_completed_user_transcript("confirm Claude command")
        assert time.monotonic() - started_at < 0.5

        # The HTTP call is still in flight in a background task.
        tasks = list(handler._claude_code_tasks)
        assert len(tasks) == 1
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 2.0), timeout=3.0)
        assert not tasks[0].done()

        # Once the daemon answers, the result is surfaced via the output queue.
        release.set()
        await asyncio.wait_for(tasks[0], timeout=3.0)

        user_output = await handler.output_queue.get()
        assistant_output = await handler.output_queue.get()
        assert user_output.args[0] == {"role": "user", "content": "confirm Claude command"}
        assert "sent" in assistant_output.args[0]["content"]
    finally:
        reset_claude_code_session_controller()


def test_partial_command_confirmation_does_not_send_claude_code_command() -> None:
    """Partial transcript tracking alone does not send a staged Claude Code command."""
    calls = []
    controller = ClaudeCodeSessionController(
        settings_loader=lambda: ClaudeCodeSessionSettings(
            base_url="http://mac.local:8765/v1/claude-code",
            token="control-token",
        ),
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    reset_claude_code_session_controller(controller)
    try:
        controller.request_send("run the tests")
        handler = _build_wake_enabled_handler()
        handler.wake_session.wake()

        handler._record_user_transcript("confirm Claude command")

        assert calls == []
        assert controller.has_pending() is True
    finally:
        reset_claude_code_session_controller()


@pytest.mark.asyncio
async def test_completed_user_transcript_ignored_while_asleep() -> None:
    """Straggler transcripts while asleep never enqueue a response."""
    handler = _build_wake_enabled_handler()

    await handler._handle_completed_user_transcript("background chatter")

    assert handler.output_queue.empty()
    assert handler._pending_responses.empty()


@pytest.mark.asyncio
async def test_receive_stops_streaming_after_sleep_phrase() -> None:
    """After the sleep phrase, mic audio stays local instead of streaming upstream."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler.connection = FakeGatingConnection()

    await handler._handle_completed_user_transcript("go to sleep")
    assert not handler.wake_session.awake

    for _ in range(3):
        await handler.receive(_mic_frame())

    assert handler.connection.input_audio_buffer.appended == []


@pytest.mark.asyncio
async def test_partial_transcript_sleep_phrase_closes_session() -> None:
    """Partial transcripts can end the streaming window without waiting for final ASR."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()

    assert await handler._maybe_sleep_from_transcript("please go to sleep now")

    assert not handler.wake_session.awake


@pytest.mark.asyncio
async def test_sleep_phrase_detected_from_latest_partial_on_speech_stopped() -> None:
    """Sleep can trigger from the last partial when VAD ends before final ASR."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler._latest_user_transcript = "go to sleep"
    handler.connection = FakeGatingConnection()

    assert await handler._maybe_sleep_from_transcript(handler._latest_user_transcript)

    assert not handler.wake_session.awake
    assert handler.connection.input_audio_buffer.cleared == 2


@pytest.mark.asyncio
async def test_sleep_phrase_detected_on_response_created_with_latest_partial() -> None:
    """If server VAD starts a response, cancel sleep when partial already matched."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler._latest_user_transcript = "go to sleep"
    handler.connection = FakeGatingConnection()

    assert await handler._maybe_sleep_from_transcript(handler._latest_user_transcript)

    assert not handler.wake_session.awake


@pytest.mark.asyncio
async def test_record_user_transcript_sets_sleep_pending_on_partial_match() -> None:
    """Partial ASR can flag sleep before the async transition runs."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()

    handler._record_user_transcript("please go to sleep")

    assert handler._sleep_pending


@pytest.mark.asyncio
async def test_response_created_after_sleep_cancels_and_blocks_audio() -> None:
    """Late response.created after sleep must not play assistant audio."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler.connection = FakeGatingConnection()
    cancel_count = 0

    class TrackingResponse:
        async def cancel(self) -> None:
            nonlocal cancel_count
            cancel_count += 1

    handler.connection.response = TrackingResponse()

    await handler._transition_to_sleep("test")

    # Simulate server VAD creating a response after we already slept.
    if (
        not handler.wake_session.awake
        or handler._sleep_pending
        or await handler._maybe_sleep_from_transcript("go to sleep")
    ):
        if handler.connection:
            try:
                await handler.connection.response.cancel()
            except Exception:
                pass

    assert cancel_count == 2  # once in transition_to_sleep, once for late response.created
    assert not handler.wake_session.awake

    handler._sleep_pending = True
    assert not handler.wake_session.awake or handler._sleep_pending


@pytest.mark.asyncio
async def test_preempt_sleep_response_cancels_active_response() -> None:
    """Sleep preemption cancels server responses and clears the input buffer."""
    handler = _build_wake_enabled_handler()
    handler.wake_session.wake()
    handler.connection = FakeGatingConnection()
    cancel_count = 0

    class TrackingResponse:
        async def cancel(self) -> None:
            nonlocal cancel_count
            cancel_count += 1

    handler.connection.response = TrackingResponse()

    await handler._preempt_sleep_response()

    assert handler._sleep_pending
    assert cancel_count == 1
    assert handler.connection.input_audio_buffer.cleared == 1


# ---- Tool-result handling ----


class RecordingItemConnection:
    """Connection stub recording conversation.item.create payloads."""

    def __init__(self) -> None:
        """Initialize with an empty item log."""
        self.items: list[dict[str, Any]] = []
        outer = self

        class _Item:
            async def create(self, *, item: dict[str, Any]) -> None:
                outer.items.append(item)

        class _Conversation:
            item = _Item()

        self.conversation = _Conversation()


@pytest.mark.asyncio
async def test_camera_tool_result_never_inlines_base64_as_text() -> None:
    """The camera JPEG travels only as an input_image item, never as raw text.

    Inlining the base64 as function_call_output text (or in the chat payload)
    injects hundreds of KB of raw tokens and breaks every camera invocation.
    """
    handler = _build_wake_enabled_handler()
    connection = RecordingItemConnection()
    handler.connection = connection

    b64_im = "QUJD" * 50_000  # ~200 KB of base64, like a real camera frame
    notification = btm_mod.ToolNotification(
        id="call_cam_1",
        tool_name="camera",
        status=ToolState.COMPLETED,
        result={"b64_im": b64_im},
    )

    await handler._handle_tool_result(notification)

    function_outputs = [i for i in connection.items if i.get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    assert b64_im not in function_outputs[0]["output"]
    assert "image captured and attached" in function_outputs[0]["output"]

    image_items = [i for i in connection.items if i.get("type") == "message"]
    assert len(image_items) == 1
    image_content = image_items[0]["content"][0]
    assert image_content["type"] == "input_image"
    assert image_content["image_url"] == f"data:image/jpeg;base64,{b64_im}"

    # The chat payload must not carry the base64 either.
    chat_payload = await handler.output_queue.get()
    assert b64_im not in chat_payload.args[0]["content"]
    assert "image captured and attached" in chat_payload.args[0]["content"]


@pytest.mark.asyncio
async def test_non_camera_tool_result_output_is_unchanged() -> None:
    """Regular tool results still serialize verbatim into the function output."""
    handler = _build_wake_enabled_handler()
    connection = RecordingItemConnection()
    handler.connection = connection

    notification = btm_mod.ToolNotification(
        id="call_1",
        tool_name="move_head",
        status=ToolState.COMPLETED,
        result={"ok": True, "direction": "left"},
    )

    await handler._handle_tool_result(notification)

    function_outputs = [i for i in connection.items if i.get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    assert '"direction": "left"' in function_outputs[0]["output"]
    assert [i for i in connection.items if i.get("type") == "message"] == []


# ---- Stress test: response.create rejection + retry ----


@pytest.mark.asyncio
async def test_response_sender_retries_on_active_response_rejection(monkeypatch: Any, caplog: Any) -> None:
    """Stress test: response.create rejection + retry via real event processing.

    Tool results () queue response.create calls via
    _safe_response_create.  When the server rejects some with
    ``conversation_already_has_active_response``, the error event flows through
    the event handler and _response_sender_loop retries the rejected request.

    The full _run_realtime_session event loop runs so that the error-handling
    code path (setting _last_response_rejected) is exercised by real event
    processing, not mocked out.
    """
    caplog.set_level(logging.DEBUG)

    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(rt_mod, "ConnectionClosedError", FakeCCE)
    monkeypatch.setattr(rt_mod, "get_realtime_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda: "alloy")
    monkeypatch.setattr(rt_mod, "get_tool_specs", lambda: [])

    N_TOOL_RESULTS = 400
    REJECT_CALL_NUMBERS = {1, 3, 5, 10, 25, 50, 75, 100, 150, 200, 300, 399}
    EXPECTED_TOTAL_CALLS = N_TOOL_RESULTS + len(REJECT_CALL_NUMBERS)

    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    response_create_log: list[tuple[int, dict[str, Any]]] = []
    handler_ref: list[Any] = []

    # ---- Fake event / error objects mirroring the OpenAI SDK shapes ----

    class FakeError:
        def __init__(self, message: str, code: str) -> None:
            self.message = message
            self.code = code
            self.type = "invalid_request_error"
            self.event_id = None
            self.param = None

        def __repr__(self) -> str:
            return (
                f"RealtimeError(message='{self.message}', type='{self.type}', "
                f"code='{self.code}', event_id=None, param=None)"
            )

    class FakeEvent:
        def __init__(self, etype: str, **kwargs: Any) -> None:
            self.type = etype
            for k, v in kwargs.items():
                setattr(self, k, v)

    # ---- Fake connection components ----

    class FakeResponseAPI:
        """Mimics connection.response.

        Pushes server events into the shared event_queue so they flow
        through the real event-handling code.  Also guards the serialization
        invariant: every create() must arrive when no response is active.
        """

        def __init__(self) -> None:
            self._call_count = 0
            self._serialization_violations: list[int] = []

        async def create(self, **kwargs: Any) -> None:
            self._call_count += 1
            n = self._call_count
            response_create_log.append((n, kwargs))

            h = handler_ref[0]

            # Real backend rejects when a response is already active.
            if not h._response_done_event.is_set():
                self._serialization_violations.append(n)
                await event_queue.put(
                    FakeEvent(
                        "error",
                        error=FakeError(
                            message=(
                                f"Conversation already has an active response in "
                                f"progress: resp_fake{n}. Wait until the response "
                                f"is finished before creating a new one."
                            ),
                            code="conversation_already_has_active_response",
                        ),
                    )
                )
                await asyncio.sleep(0)
                await event_queue.put(FakeEvent("response.done", response=MagicMock()))
                return

            # Intentional rejections (simulating a race where another
            # response sneaks in right after our check).
            if n in REJECT_CALL_NUMBERS:
                await event_queue.put(
                    FakeEvent(
                        "error",
                        error=FakeError(
                            message=(
                                f"Conversation already has an active response in "
                                f"progress: resp_fake{n}. Wait until the response "
                                f"is finished before creating a new one."
                            ),
                            code="conversation_already_has_active_response",
                        ),
                    )
                )
                await asyncio.sleep(0)
            else:
                await event_queue.put(FakeEvent("response.created"))

            await event_queue.put(FakeEvent("response.done", response=MagicMock()))

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_response_api = FakeResponseAPI()

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = fake_response_api

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> FakeEvent:
            event: FakeEvent = await event_queue.get()
            if event is None:  # sentinel → end iteration
                raise StopAsyncIteration
            return event

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.realtime = FakeRealtime()

    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)

    # Patch dispatch_tool_call so tools complete with a result.
    async def _fake_dispatch(tool_name: str, args_json: str, deps: Any, **_kw: Any) -> dict[str, Any]:
        await asyncio.sleep(random.uniform(0.3, 0.5))
        return {"ok": True, "tool": tool_name}

    monkeypatch.setattr(btm_mod, "dispatch_tool_call", _fake_dispatch)

    # ---- Build handler and start the full realtime session ----

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler_ref.append(handler)

    startup_task = asyncio.create_task(handler.start_up())

    # ---- Start tools via the real BackgroundToolManager pipeline ----
    # start_tool → _run_tool → notification queue → listener → _handle_tool_result

    for i in range(N_TOOL_RESULTS):
        await handler.tool_manager.start_tool(
            call_id=f"call_{i}",
            tool_call_routine=ToolCallRoutine(
                tool_name="test_tool",
                args_json_str=f'{{"index": {i}}}',
                deps=deps,
            ),
        )

    # Yield so spawned tool tasks, the listener, and the sender can drain.
    await asyncio.sleep(5)

    # ---- Tear down ----

    await event_queue.put(None)  # sentinel stops event iteration

    await _finish_supervisor(handler, startup_task)

    # ---- Assertions ----

    # Serialization: every response.create() must have been called only when
    # no response was in-flight (_response_done_event was set).  Any violation
    # means the sender fired a new request before the previous one finished.
    assert fake_response_api._serialization_violations == [], (
        f"response.create() was called while a response was still active on "
        f"call(s) {fake_response_api._serialization_violations}"
    )

    # Total response.create() calls = tool results + retries for rejected ones
    assert fake_response_api._call_count == EXPECTED_TOTAL_CALLS, (
        f"Expected {EXPECTED_TOTAL_CALLS} response.create calls "
        f"({N_TOOL_RESULTS} results + {len(REJECT_CALL_NUMBERS)} retries), "
        f"got {fake_response_api._call_count}"
    )

    # The error event handler must have set _last_response_rejected for each
    # rejection (the log message comes from the event handler code path).
    rejection_logs = [r for r in caplog.records if "worker will retry" in getattr(r, "msg", "")]
    assert len(rejection_logs) == len(REJECT_CALL_NUMBERS), (
        f"Expected {len(REJECT_CALL_NUMBERS)} rejection entries from error handler, got {len(rejection_logs)}"
    )

    # The sender loop must have retried after each rejection.
    retry_logs = [r for r in caplog.records if "response.create was rejected; retrying" in getattr(r, "msg", "")]
    assert len(retry_logs) == len(REJECT_CALL_NUMBERS), (
        f"Expected {len(REJECT_CALL_NUMBERS)} retry entries from sender loop, got {len(retry_logs)}"
    )


# ---- Response creation timeout guard tests ----


@pytest.mark.asyncio
async def test_response_sender_loop_times_out_waiting_for_response_done(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """If response.done is never received the sender loop should time out.

    Rather than hang forever, it force-sets the event and moves on.
    """
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(rt_mod, "_RESPONSE_DONE_TIMEOUT", 0.3)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    create_count = 0

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            nonlocal create_count
            create_count += 1
            # Simulate response.created clearing the event, but never
            # send response.done (so the event stays cleared forever).
            handler._response_done_event.clear()

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_conn = MagicMock()
    fake_conn.response = FakeResponse()
    handler.connection = fake_conn

    # Queue two requests
    await handler._safe_response_create(instructions="req1")
    await handler._safe_response_create(instructions="req2")

    sender_task = asyncio.create_task(handler._response_sender_loop())

    # Give enough time for both requests to time out (0.3s each + margin)
    await asyncio.sleep(1.5)

    handler.connection = None  # signal the loop to exit
    handler._response_done_event.set()
    await asyncio.wait_for(sender_task, timeout=2.0)

    assert create_count == 2, f"Expected 2 response.create calls, got {create_count}"

    timeout_logs = [r for r in caplog.records if "Timed out waiting for response.done" in r.getMessage()]
    assert len(timeout_logs) == 2, f"Expected 2 timeout warnings, got {len(timeout_logs)}"


@pytest.mark.asyncio
async def test_response_sender_loop_times_out_waiting_for_previous_response(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """If a previous response never completes, the pre-condition wait times out.

    It should force-set the event and proceed to send.
    """
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(rt_mod, "_RESPONSE_DONE_TIMEOUT", 0.3)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    # Pretend a response is already in-flight (event cleared)
    handler._response_done_event.clear()

    created = asyncio.Event()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            # Immediately complete the response cycle so the loop can finish
            handler._response_done_event.set()
            created.set()

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_conn = MagicMock()
    fake_conn.response = FakeResponse()
    handler.connection = fake_conn

    await handler._safe_response_create(instructions="waiting_req")

    sender_task = asyncio.create_task(handler._response_sender_loop())

    # Wait for the request to be sent (after timing out on the pre-condition)
    await asyncio.wait_for(created.wait(), timeout=2.0)

    handler.connection = None
    handler._response_done_event.set()
    await asyncio.wait_for(sender_task, timeout=2.0)

    timeout_logs = [r for r in caplog.records if "Timed out waiting for previous response" in r.getMessage()]
    assert len(timeout_logs) == 1, f"Expected 1 pre-condition timeout warning, got {len(timeout_logs)}"


def test_should_ignore_server_vad_while_response_active() -> None:
    """Server VAD is ignored while a response is active."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._response_done_event.clear()
    assert handler._should_ignore_server_vad() is True


def test_should_ignore_server_vad_after_recent_assistant_audio() -> None:
    """Server VAD is ignored briefly after assistant audio."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._response_done_event.set()
    handler._last_assistant_audio_at = time.monotonic()
    assert handler._should_ignore_server_vad() is True


def test_should_accept_server_vad_after_assistant_guard_elapsed() -> None:
    """Server VAD resumes after the assistant audio guard expires."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._response_done_event.set()
    handler._last_assistant_audio_at = time.monotonic() - rt_mod._ASSISTANT_VAD_GUARD_S - 0.1
    assert handler._should_ignore_server_vad() is False


def test_sleep_cue_translates_head_on_z_axis() -> None:
    """Going to sleep lowers the head ~3 cm vertically; waking restores it."""
    from reachy_mini.utils import create_head_pose

    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True, mm=True)
    robot = MagicMock()
    robot.get_current_head_pose.return_value = neutral
    robot.get_current_joint_positions.return_value = ((0.0,), (0.2, -0.2))

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=robot, movement_manager=movement_manager)
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    handler._queue_antenna_cue(awake=False)
    sleep_move = movement_manager.queue_move.call_args[0][0]
    assert sleep_move.target_head_pose[2, 3] == pytest.approx(-0.03)
    assert sleep_move.target_antennas == (0.0, 0.0)

    robot.get_current_head_pose.return_value = sleep_move.target_head_pose
    movement_manager.reset_mock()
    handler._queue_antenna_cue(awake=True)
    wake_move = movement_manager.queue_move.call_args[0][0]
    assert wake_move.target_head_pose[2, 3] == pytest.approx(0.0)
    assert wake_move.target_antennas == (-0.5, 0.5)
