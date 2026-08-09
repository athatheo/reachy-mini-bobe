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


def test_remote_client_ignores_sleep_phrase_inside_conversation():
    """Re-validation must be a strict command match, not substring containment."""
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
            "transcript": "My toddler won't go to sleep, any tips?",
            "latency_ms": 40.0,
        }
    )
    assert not slept
    events = client.debug_state()["events"]
    assert events[-1]["level"] == "warn"


def test_remote_client_accepts_sleep_command_with_fillers():
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
            "transcript": "Okay Bobe, please go to sleep now.",
            "latency_ms": 40.0,
        }
    )
    assert slept


# ---- reconnect handshake / session lifecycle ----


def _make_client(**kwargs) -> RemoteWakeClient:
    return RemoteWakeClient(lambda: None, url="ws://127.0.0.1:8765/v1/stream", token="secret", **kwargs)


def _listen_payloads(ws: FakeWebSocket) -> list[dict]:
    payloads = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    return [p for p in payloads if p.get("type") == "listen"]


async def _wait_for_listen(ws: FakeWebSocket, *, mode: str | None = None, timeout: float = 2.0) -> None:
    """Poll until the send loop has flushed a listen payload (optionally for a given mode)."""
    deadline = time.monotonic() + timeout
    while not any(mode is None or p["mode"] == mode for p in _listen_payloads(ws)):
        assert time.monotonic() < deadline, f"no listen payload (mode={mode!r}) sent within {timeout}s"
        await asyncio.sleep(0.01)


def test_run_connection_replays_current_listen_mode_and_stops_on_sentinel():
    client = _make_client()
    client._listen_mode = "sleep"
    ws = FakeWebSocket()

    async def run() -> None:
        task = asyncio.create_task(client._run_connection(ws))
        await _wait_for_listen(ws)
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
        # The listen replay is sent after _drain_audio_queue(), so once it
        # shows up any stale frames were either dropped or already in ws.sent
        # (the queue is FIFO); the negative assertion below is deterministic.
        await _wait_for_listen(ws)
        client._audio_queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(run())
    assert not any(isinstance(m, (bytes, bytearray)) for m in ws.sent)


async def _drive_send_loop(client: RemoteWakeClient, ws: FakeWebSocket) -> None:
    task = asyncio.create_task(client._send_loop(ws))
    await _wait_for_listen(ws)
    client._audio_queue.put_nowait(None)  # stop sentinel
    await asyncio.wait_for(task, timeout=5.0)


def test_rapid_mode_toggles_send_only_latest_mode():
    client = _make_client()
    for _ in range(10):
        client.listen_for_sleep()
        client.listen_for_wake()
    ws = FakeWebSocket()

    asyncio.run(_drive_send_loop(client, ws))

    listens = _listen_payloads(ws)
    assert len(listens) == 1
    assert listens[0]["mode"] == "wake"


def test_send_loop_reflags_mode_send_on_failure():
    client = _make_client()
    client.listen_for_sleep()

    class FailingWebSocket:
        async def send(self, data: object) -> None:
            raise ConnectionError("socket closed")

    with pytest.raises(ConnectionError):
        asyncio.run(client._send_loop(FailingWebSocket()))

    assert client._mode_send_pending.is_set()

    # The retried send must resolve the *current* mode, not replay the
    # payload that failed: a mode change while disconnected wins.
    client.listen_for_wake()
    ws = FakeWebSocket()
    asyncio.run(_drive_send_loop(client, ws))
    listens = _listen_payloads(ws)
    assert listens
    assert listens[-1]["mode"] == "wake"


def test_reconnect_replay_cannot_clobber_concurrent_mode_change():
    """Regression: the replay used to snapshot the mode and drain the control
    queue, dropping a listen_for_sleep() that raced with the reconnect and
    leaving the daemon in wake mode while BoBe was awake."""
    client = _make_client()  # desired mode starts as "wake"

    class ModeSwitchingWebSocket(FakeWebSocket):
        """Switches modes while the replayed wake send is in flight."""

        def __init__(self, owner: RemoteWakeClient) -> None:
            super().__init__()
            self._owner = owner
            self._switched = False

        async def send(self, data: object) -> None:
            await super().send(data)
            if not self._switched and isinstance(data, str) and json.loads(data).get("type") == "listen":
                self._switched = True
                self._owner.listen_for_sleep()

    ws = ModeSwitchingWebSocket(client)

    async def run() -> None:
        task = asyncio.create_task(client._run_connection(ws))
        await _wait_for_listen(ws, mode="sleep")
        client._audio_queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(run())

    listens = _listen_payloads(ws)
    assert listens, "the reconnect must replay a listen mode"
    assert listens[-1]["mode"] == "sleep"
    assert listens[-1]["sleep_phrases"]


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


# ---- speak downlink ----


def _speak_payload(clip_id, seq, pcm, rate=24000, last=False):
    import base64

    return {
        "type": "speak",
        "id": clip_id,
        "seq": seq,
        "pcm_b64": base64.b64encode(pcm.tobytes()).decode("ascii"),
        "rate": rate,
        "last": last,
    }


def test_remote_client_assembles_speak_chunks_in_order():
    received = []
    client = RemoteWakeClient(
        lambda: None,
        url="ws://mac:8765/v1/stream",
        on_speak=lambda pcm, rate: received.append((pcm, rate)),
    )
    first = np.arange(10, dtype=np.int16)
    second = np.arange(10, 15, dtype=np.int16)

    client._handle_speak_payload(_speak_payload("clip", 0, first))
    assert received == []

    client._handle_speak_payload(_speak_payload("clip", 1, second, last=True))
    assert len(received) == 1
    pcm, rate = received[0]
    assert rate == 24000
    assert np.array_equal(pcm, np.concatenate([first, second]))
    assert client._speak_buffers == {}


def test_remote_client_speak_without_callback_is_ignored():
    client = RemoteWakeClient(lambda: None, url="ws://mac:8765/v1/stream")

    client._handle_speak_payload(_speak_payload("clip", 0, np.ones(10, dtype=np.int16), last=True))

    assert client._speak_buffers == {}


def test_remote_client_drops_oversized_speak_clip():
    from bobe.wake.remote_client import SPEAK_MAX_CLIP_SECONDS

    received = []
    client = RemoteWakeClient(
        lambda: None,
        url="ws://mac:8765/v1/stream",
        on_speak=lambda pcm, rate: received.append((pcm, rate)),
    )
    oversized = np.zeros(int(SPEAK_MAX_CLIP_SECONDS) + 1, dtype=np.int16)

    client._handle_speak_payload(_speak_payload("clip", 0, oversized, rate=1, last=True))

    assert received == []
    assert client._speak_buffers == {}


def test_remote_client_ignores_malformed_speak_payloads():
    received = []
    client = RemoteWakeClient(
        lambda: None,
        url="ws://mac:8765/v1/stream",
        on_speak=lambda pcm, rate: received.append((pcm, rate)),
    )

    client._handle_speak_payload({"type": "speak"})
    client._handle_speak_payload({"type": "speak", "id": "c", "pcm_b64": 123, "rate": 24000})
    client._handle_speak_payload({"type": "speak", "id": "c", "pcm_b64": "aGk=", "rate": 0})
    client._handle_speak_payload({"type": "speak", "id": "c", "pcm_b64": "!!!not-base64", "rate": 24000})

    assert received == []
    assert client._speak_buffers == {}


def test_remote_client_evicts_stale_speak_buffers(monkeypatch):
    from bobe.wake import remote_client as rc_mod

    received = []
    client = RemoteWakeClient(
        lambda: None,
        url="ws://mac:8765/v1/stream",
        on_speak=lambda pcm, rate: received.append((pcm, rate)),
    )
    now = {"t": 1000.0}
    monkeypatch.setattr(rc_mod.time, "monotonic", lambda: now["t"])

    client._handle_speak_payload(_speak_payload("stale", 0, np.ones(10, dtype=np.int16)))
    assert "stale" in client._speak_buffers

    now["t"] += rc_mod.SPEAK_CLIP_STALE_S + 1.0
    client._handle_speak_payload(_speak_payload("fresh", 0, np.ones(5, dtype=np.int16), last=True))

    assert "stale" not in client._speak_buffers
    assert len(received) == 1


def test_remote_client_dispatches_emotes():
    received = []
    client = RemoteWakeClient(
        lambda: None,
        url="ws://mac:8765/v1/stream",
        on_emote=received.append,
    )

    client._handle_emote_payload({"type": "emote", "emotion": "amazed1"})
    client._handle_emote_payload({"type": "emote", "emotion": "  "})

    assert received == ["amazed1"]


def test_remote_client_queues_presence_control_message():
    client = RemoteWakeClient(lambda: None, url="ws://mac:8765/v1/stream")

    client.notify_presence("aGVsbG8=")

    control = client._control_queue.get_nowait()
    assert control == {"type": "presence", "jpeg_b64": "aGVsbG8="}
