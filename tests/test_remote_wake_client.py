# ruff: noqa: D101,D102,D103,D107
import json
import time
import queue
import asyncio

import numpy as np
import pytest

from bobe.wake.remote_client import AUTH_RETRY_S, RemoteWakeClient, _close_code


class FakeWebSocket:
    """Minimal ws double: records sends, recv blocks until cancelled."""

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, data: object) -> None:
        self.sent.append(data)

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        await asyncio.Event().wait()  # block forever until cancelled
        raise StopAsyncIteration


def test_remote_client_debug_state_is_json_serializable():
    client = RemoteWakeClient(lambda: None, url="ws://192.168.1.114:8765/v1/stream", token="secret")
    client._log_event("info", "Connected")
    client._apply_remote_stats(
        {
            "transcript": "hey jarvis",
            "rms": 512.0,
            "in_speech": True,
            "latency_ms_last": 82.5,
            "engine": "faster-whisper",
            "model": "tiny.en",
        }
    )
    state = client.debug_state()
    json.dumps(state)
    assert state["backend"] == "remote"
    assert state["remote_stats"]["transcript"] == "hey jarvis"
    assert len(state["events"]) == 1


def test_remote_client_wake_event_is_logged():
    client = RemoteWakeClient(lambda: None, url="ws://127.0.0.1:8765/v1/stream")
    client._log_event("wake", "Wake detected: 'hey jarvis'", latency_ms=75.0)
    events = client.debug_state()["events"]
    assert events[-1]["level"] == "wake"
    assert "hey jarvis" in str(events[-1]["message"])


def test_remote_client_ignores_wake_without_phrase_match():
    woke = False

    def on_wake():
        nonlocal woke
        woke = True

    client = RemoteWakeClient(on_wake, url="ws://127.0.0.1:8765/v1/stream")
    client._handle_wake_payload(
        {
            "type": "wake",
            "transcript": "good morning",
            "latency_ms": 50.0,
        }
    )
    assert not woke
    events = client.debug_state()["events"]
    assert events[-1]["level"] == "warn"


def test_remote_client_accepts_wake_with_phrase_match():
    woke = False

    def on_wake():
        nonlocal woke
        woke = True

    client = RemoteWakeClient(on_wake, url="ws://127.0.0.1:8765/v1/stream")
    client._handle_wake_payload(
        {
            "type": "wake",
            "transcript": "hey bobe",
            "latency_ms": 50.0,
        }
    )
    assert woke


def test_remote_client_accepts_sleep_with_phrase_match():
    slept = False

    def on_sleep():
        nonlocal slept
        slept = True

    client = RemoteWakeClient(
        lambda: None,
        url="ws://127.0.0.1:8765/v1/stream",
        on_sleep=on_sleep,
    )
    client._handle_sleep_payload(
        {
            "type": "sleep",
            "transcript": "go to sleep",
            "latency_ms": 40.0,
        }
    )
    assert slept


def test_remote_client_ignores_sleep_without_phrase_match():
    slept = False

    def on_sleep():
        nonlocal slept
        slept = True

    client = RemoteWakeClient(
        lambda: None,
        url="ws://127.0.0.1:8765/v1/stream",
        on_sleep=on_sleep,
    )
    client._handle_sleep_payload(
        {
            "type": "sleep",
            "transcript": "what time is it",
            "latency_ms": 40.0,
        }
    )
    assert not slept


# ---- reconnect handshake / session lifecycle ----


def _make_client(**kwargs) -> RemoteWakeClient:
    return RemoteWakeClient(lambda: None, url="ws://127.0.0.1:8765/v1/stream", token="secret", **kwargs)


def test_run_connection_replays_current_listen_mode_and_stops_on_sentinel():
    client = _make_client()
    client._listen_mode = "sleep"
    ws = FakeWebSocket()

    async def run() -> None:
        task = asyncio.create_task(client._run_connection(ws))
        await asyncio.sleep(0.3)
        # Stop sentinel, as stop() would enqueue after the handshake.
        client._audio_queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=5.0)

    # With the recv loop parked in `async for`, this only returns if the
    # session ends when the send loop consumes the stop sentinel.
    asyncio.run(run())

    payloads = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert payloads[0]["type"] == "hello"
    listens = [p for p in payloads if p["type"] == "listen"]
    assert listens, "current listen mode must be replayed after the handshake"
    assert listens[-1]["mode"] == "sleep"
    assert listens[-1]["sleep_phrases"]


def test_run_connection_drops_stale_queued_audio():
    client = _make_client()
    client.feed(np.ones(160, dtype=np.int16))
    client.feed(np.ones(160, dtype=np.int16))
    ws = FakeWebSocket()

    async def run() -> None:
        task = asyncio.create_task(client._run_connection(ws))
        await asyncio.sleep(0.3)
        client._audio_queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(run())
    assert not any(isinstance(m, (bytes, bytearray)) for m in ws.sent)


def test_queue_listen_mode_coalesces_to_latest():
    client = _make_client()
    for _ in range(10):
        client.listen_for_sleep()
        client.listen_for_wake()
    mode, payload = client._control_queue.get_nowait()
    assert mode == "wake"
    assert payload["mode"] == "wake"
    with pytest.raises(queue.Empty):
        client._control_queue.get_nowait()


def test_send_loop_requeues_control_payload_on_send_failure():
    client = _make_client()
    client.listen_for_sleep()

    class FailingWebSocket:
        async def send(self, data: object) -> None:
            raise ConnectionError("socket closed")

    with pytest.raises(ConnectionError):
        asyncio.run(client._send_loop(FailingWebSocket()))

    mode, payload = client._control_queue.get_nowait()
    assert mode == "sleep"
    assert payload["mode"] == "sleep"


def test_start_drains_stale_audio_and_sentinels(monkeypatch):
    client = _make_client()
    client._audio_queue.put_nowait(None)  # sentinel from a previous stop()
    client.feed(np.ones(160, dtype=np.int16))
    monkeypatch.setattr(client, "_run", lambda: None)
    client.start()
    client._thread.join(timeout=5.0)
    with pytest.raises(queue.Empty):
        client._audio_queue.get_nowait()


def test_sleep_unless_stopped_returns_promptly_on_stop():
    client = _make_client()

    async def run() -> None:
        async def stop_soon() -> None:
            await asyncio.sleep(0.05)
            client._stop_event.set()

        stopper = asyncio.create_task(stop_soon())
        started = time.monotonic()
        await asyncio.wait_for(client._sleep_unless_stopped(30.0), timeout=5.0)
        assert time.monotonic() - started < 2.0
        await stopper

    asyncio.run(run())


# ---- 1008 auth rejection ----


def test_close_code_extracts_websockets_policy_violation():
    from websockets.frames import Close
    from websockets.exceptions import ConnectionClosedError

    exc = ConnectionClosedError(Close(1008, "policy violation"), None)
    assert _close_code(exc) == 1008
    assert _close_code(ConnectionError("boom")) is None


def test_auth_rejection_surfaces_error_and_backs_off_long(monkeypatch):
    import websockets
    from websockets.frames import Close

    client = _make_client()

    class RejectingConnect:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            raise websockets.exceptions.ConnectionClosedError(Close(1008, "policy violation"), None)

        async def __aexit__(self, *exc) -> bool:
            return False

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        client._stop_event.set()

    monkeypatch.setattr(websockets, "connect", RejectingConnect)
    monkeypatch.setattr(client, "_sleep_unless_stopped", fake_sleep)

    asyncio.run(client._main())

    state = client.debug_state()
    assert state["auth_error"]
    assert "BOBE_WAKE_TOKEN" in state["auth_error"]
    assert sleeps == [AUTH_RETRY_S]
    assert any(e["level"] == "error" for e in state["events"])
