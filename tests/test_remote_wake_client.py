# ruff: noqa: D103
import json

from bobe.wake.remote_client import RemoteWakeClient


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
            "transcript": "hey jarvis",
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
