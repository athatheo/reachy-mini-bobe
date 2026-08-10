# ruff: noqa: D101,D102,D103,D107
import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import bobe.voice_handler as vh_mod
from bobe.wake_word import WakeConfig, WakeSession
from bobe.tools.core_tools import ToolDependencies


def _build_handler() -> vh_mod.BobeVoiceHandler:
    """Build a handler with wake gating and no detector thread."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = vh_mod.BobeVoiceHandler(deps)
    handler.wake_config = WakeConfig()
    handler.wake_session = WakeSession()
    handler._wake_detector = None
    handler.wake_gating_enabled = True
    handler.wake_error = None
    return handler


def _mic_frame(samples: int = 2400) -> tuple[int, Any]:
    return (24000, np.ones(samples, dtype=np.int16))


# ---- lifecycle ----


@pytest.mark.asyncio
async def test_start_up_parks_until_shutdown():
    handler = _build_handler()

    startup = asyncio.create_task(handler.start_up())
    await asyncio.sleep(0.05)
    assert not startup.done()

    await handler.shutdown()
    await asyncio.wait_for(startup, timeout=2.0)


@pytest.mark.asyncio
async def test_shutdown_cancels_background_tasks_and_drains_queue():
    handler = _build_handler()
    handler.wake_session.wake()
    await handler.output_queue.put((24000, np.zeros((1, 10), dtype=np.int16)))
    handler._start_announcement("late")

    await handler.shutdown()

    assert handler.output_queue.empty()
    assert handler._announce_tasks == set()


def test_copy_preserves_configuration():
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = vh_mod.BobeVoiceHandler(deps, gradio_mode=True, instance_path="/tmp/x")
    clone = handler.copy()
    assert clone.gradio_mode is True
    assert clone.instance_path == "/tmp/x"
    assert clone.deps is deps


# ---- wake gating in receive() ----


@pytest.mark.asyncio
async def test_receive_noops_when_gating_disabled(caplog: Any):
    caplog.set_level(logging.ERROR)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = vh_mod.BobeVoiceHandler(deps)
    handler.wake_gating_enabled = False

    await handler.receive(_mic_frame())  # must not raise

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_wake_event_starts_converse_transition():
    handler = _build_handler()
    listen_calls: list[str] = []
    handler.wake_gate.detector = type(
        "FakeDetector",
        (),
        {
            "listen_for_converse": lambda self: listen_calls.append("converse"),
            "listen_for_sleep": lambda self: listen_calls.append("sleep"),
            "is_running": lambda self: True,
            "feed": lambda self, frame: None,
        },
    )()
    handler.wake_session.request_wake()

    await handler.receive(_mic_frame())
    task = handler._wake_transition_task
    assert task is not None
    await task

    assert handler.wake_session.awake
    assert listen_calls == ["converse"]
    # Chime was queued during the transition.
    assert not handler.output_queue.empty()


@pytest.mark.asyncio
async def test_sleep_event_transitions_to_sleep():
    handler = _build_handler()
    handler.wake_session.wake()
    handler.wake_session.request_sleep()

    await handler.receive(_mic_frame())

    assert not handler.wake_session.awake


@pytest.mark.asyncio
async def test_expired_session_transitions_to_sleep():
    clock = {"now": 1000.0}
    handler = _build_handler()
    handler.wake_session = WakeSession(timeout_s=5.0, clock=lambda: clock["now"])
    handler.wake_session.wake()
    clock["now"] += 6.0

    await handler.receive(_mic_frame())

    assert not handler.wake_session.awake


@pytest.mark.asyncio
async def test_spoken_sleep_blocks_announce_wake_but_timeout_does_not():
    handler = _build_handler()
    handler.wake_session.wake()
    await handler._transition_to_sleep("local sleep phrase")
    assert handler._announce_wake_allowed is False

    handler.wake_session.wake()
    await handler._transition_to_sleep("inactivity timeout")
    assert handler._announce_wake_allowed is False  # stays off until a real wake

    await handler._transition_to_awake()
    assert handler._announce_wake_allowed is True


# ---- announcements ----


@pytest.mark.asyncio
async def test_announcement_surfaces_immediately_while_awake():
    handler = _build_handler()
    handler.wake_session.wake()

    await handler._handle_announcement("Build finished.")

    output = await handler.output_queue.get()
    assert output.args[0] == {"role": "assistant", "content": "Build finished."}
    assert handler._pending_announcements == []


@pytest.mark.asyncio
async def test_announcement_while_asleep_queues_and_requests_wake():
    handler = _build_handler()

    await handler._handle_announcement("Build finished.")

    assert handler._pending_announcements == ["Build finished."]
    assert handler.wake_session.consume_wake_request()
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_receive_drains_gate_announcements():
    handler = _build_handler()
    handler.wake_gate.request_announce("Ping from Hermes")

    await handler.receive(_mic_frame())
    for task in list(handler._announce_tasks):
        await task

    assert handler._pending_announcements == ["Ping from Hermes"]
    assert handler.wake_session.consume_wake_request()


@pytest.mark.asyncio
async def test_manual_sleep_holds_announcements_until_next_wake():
    handler = _build_handler()
    handler.wake_session.wake()
    await handler._transition_to_sleep("local sleep phrase")

    await handler._handle_announcement("Build finished.")

    assert handler._pending_announcements == ["Build finished."]
    assert not handler.wake_session.consume_wake_request()


@pytest.mark.asyncio
async def test_wake_drains_held_announcements():
    handler = _build_handler()
    handler.wake_session.wake()
    await handler._transition_to_sleep("local sleep phrase")
    await handler._handle_announcement("Held one.")

    assert await handler._transition_to_awake()

    assert handler._pending_announcements == []
    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())
    assert any(
        getattr(o, "args", [None])[0] == {"role": "assistant", "content": "Held one."} for o in outputs
    )


@pytest.mark.asyncio
async def test_speak_announcement_parks_when_sleep_races_in():
    handler = _build_handler()  # asleep

    await handler._speak_announcement("Late one.")

    assert handler._pending_announcements == ["Late one."]
    assert handler.output_queue.empty()


# ---- speech-clip playback (Hermes TTS downlink) ----


@pytest.mark.asyncio
async def test_play_speech_clip_queues_paced_chunks_and_feeds_wobbler():
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.head_wobbler = MagicMock()
    handler = vh_mod.BobeVoiceHandler(deps)
    handler.wake_gating_enabled = True
    handler.wake_session.wake()
    pcm = np.ones(36000, dtype=np.int16)  # 1.5 s @ 24 kHz -> 3 half-second chunks

    await handler._play_speech_clip(pcm, 24000)

    chunks = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        assert isinstance(item, tuple)
        rate, audio = item
        assert rate == 24000
        chunks.append(audio.reshape(-1))
    assert sum(chunk.size for chunk in chunks) == pcm.size
    assert deps.head_wobbler.feed.call_count == len(chunks)


@pytest.mark.asyncio
async def test_handle_speech_clip_parks_and_requests_wake_while_asleep():
    handler = _build_handler()
    pcm = np.ones(100, dtype=np.int16)

    await handler._handle_speech_clip(pcm, 24000)

    assert len(handler._pending_speech) == 1
    assert handler.wake_session.consume_wake_request() is True
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_speech_clip_wakes_even_after_spoken_sleep():
    """Clips are deliberate spoken content: the announce hold never applies."""
    handler = _build_handler()
    handler._announce_wake_allowed = False
    pcm = np.ones(100, dtype=np.int16)

    await handler._handle_speech_clip(pcm, 24000)

    assert len(handler._pending_speech) == 1
    assert handler.wake_session.consume_wake_request() is True


@pytest.mark.asyncio
async def test_play_speech_clip_stops_when_sleep_lands_mid_clip():
    handler = _build_handler()  # asleep: playback must refuse and park

    await handler._play_speech_clip(np.ones(48000, dtype=np.int16), 24000)

    assert handler.output_queue.empty()
    assert len(handler._pending_speech) == 1


@pytest.mark.asyncio
async def test_wake_drains_parked_speech_clips():
    handler = _build_handler()
    pcm = np.ones(1200, dtype=np.int16)
    await handler._handle_speech_clip(pcm, 24000)
    assert handler._pending_speech

    assert await handler._transition_to_awake()

    assert handler._pending_speech == []
    audio_items = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            audio_items.append(item)
    # Chime + clip audio both queued.
    assert len(audio_items) >= 2


# ---- Hermes-driven emotes ----


@pytest.mark.asyncio
async def test_emote_dispatches_play_emotion_while_awake(monkeypatch):
    handler = _build_handler()
    handler.wake_session.wake()
    calls: list[tuple[str, str]] = []

    async def fake_dispatch(tool_name, args_json, deps):
        calls.append((tool_name, args_json))
        return {"status": "queued"}

    monkeypatch.setattr(vh_mod, "dispatch_tool_call", fake_dispatch)

    await handler._handle_emote("amazed1")

    assert calls == [("play_emotion", '{"emotion": "amazed1"}', )]


@pytest.mark.asyncio
async def test_emote_parks_while_asleep_and_plays_on_wake(monkeypatch):
    handler = _build_handler()
    calls: list[str] = []

    async def fake_dispatch(tool_name, args_json, deps):
        calls.append(args_json)
        return {"status": "queued"}

    monkeypatch.setattr(vh_mod, "dispatch_tool_call", fake_dispatch)

    await handler._handle_emote("welcoming1")
    assert handler._pending_emotes == ["welcoming1"]
    assert calls == []

    assert await handler._transition_to_awake()

    assert handler._pending_emotes == []
    assert calls == ['{"emotion": "welcoming1"}']


@pytest.mark.asyncio
async def test_receive_drains_gate_emotes(monkeypatch):
    handler = _build_handler()
    handler.wake_session.wake()
    calls: list[str] = []

    async def fake_dispatch(tool_name, args_json, deps):
        calls.append(tool_name)
        return {"status": "queued"}

    monkeypatch.setattr(vh_mod, "dispatch_tool_call", fake_dispatch)
    handler.wake_gate.request_emote("proud3")

    await handler.receive(_mic_frame())
    for task in list(handler._announce_tasks):
        await task

    assert calls == ["play_emotion"]
