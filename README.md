---
title: Reachy Mini BoBe
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
short_description: Claude-backed Reachy Mini assistant named BoBe.
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Mini BoBe

BoBe is a Reachy Mini assistant foundation. It starts from Pollen Robotics' official conversation app template and locks the robot personality to a BoBe profile.

## Current milestone

- Local wake word: say `Hey Jarvis` to wake BoBe (openWakeWord, fully on-device).
- Voice input/output uses the official Reachy Mini conversation app pipeline.
- Normal assistant answers are routed through Claude with the `ask_claude` profile tool.
- Expressive robot responses use the existing Reachy Mini motion tools, including `play_emotion`, `move_head`, and `sweep_look`.

## Privacy model

- While asleep, microphone audio never leaves the robot: frames go to a local openWakeWord detector and a short in-memory ring buffer that is continuously discarded.
- Saying `Hey Jarvis` opens a streaming window (chime + antennas up). During that window audio streams to OpenAI Realtime for transcription and speech, like any cloud voice assistant.
- The window closes (chime + antennas relaxed) when you say `go to sleep` (or Greek `κοιμήσου`) or after `BOBE_WAKE_TIMEOUT_S` (default 300s) without session activity.
- Tune with `BOBE_WAKE_MODEL`, `BOBE_WAKE_THRESHOLD`, `BOBE_WAKE_TIMEOUT_S`, `BOBE_SLEEP_PHRASE`. Wake-word gating is always on: say the wake phrase to stream, `go to sleep` to stop.

Claude Code session launching is intentionally not enabled yet. That needs a later authorization and shell-safety layer before voice commands can start local coding sessions safely.

## Configuration

Copy `.env.example` to `.env` for local development and set the keys you need:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
CLAUDE_MODEL="claude-sonnet-4-6"
BOBE_WAKE_BACKEND=heed
BOBE_WAKE_MODEL="hey_jarvis"
```

The OpenAI key is used by the inherited realtime speech bridge. The Anthropic key is used by BoBe's `ask_claude` tool for Claude-backed answers.

## Development with uv

```bash
uv sync --group dev
uv run pytest tests/test_claude.py
uv run reachy-mini-app-assistant check .
```

For a local simulation smoke test, start the daemon in one terminal and run the app in another:

```bash
uv run reachy-mini-daemon --sim
uv run bobe --gradio
```

Simulation can validate app startup and UI wiring, but physical audio, wake-word behavior, and robot motion still need hardware testing on a Reachy Mini Lite or Wireless unit.

## BoBe profile files

- `src/bobe/profiles/_bobe_locked_profile/instructions.txt`: BoBe's system behavior.
- `src/bobe/profiles/_bobe_locked_profile/tools.txt`: enabled tool list.
- `src/bobe/profiles/_bobe_locked_profile/ask_claude.py`: Claude-backed answer tool.

