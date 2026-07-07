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

- Remote wake word: say `Hey Jarvis` to wake BoBe (Mac-side Whisper daemon; robot streams PCM while asleep).
- Voice input/output uses the official Reachy Mini conversation app pipeline.
- Normal assistant answers are routed through Claude with the `ask_claude` profile tool.
- Claude Code can be launched on the Mac mini after a second spoken confirmation, when the Mac launch endpoint is explicitly enabled.
- Expressive robot responses use the existing Reachy Mini motion tools, including `play_emotion`, `move_head`, and `sweep_look`.

## Privacy model

- While asleep, microphone PCM streams from the robot to a Mac-side wake daemon over WebSocket. A short in-memory ring buffer on the robot is continuously discarded; nothing goes to OpenAI until wake.
- Saying `Hey Jarvis` opens a streaming window (chime + antennas up). During that window audio streams to OpenAI Realtime for transcription and speech, like any cloud voice assistant.
- The window closes (chime + antennas relaxed) when you say `go to sleep` (or Greek `κοιμήσου`) or after `BOBE_WAKE_TIMEOUT_S` (default 300s) without session activity.
- Tune with `BOBE_WAKE_REMOTE_URL`, `BOBE_WAKE_TOKEN`, `BOBE_WAKE_GAIN`, `BOBE_WAKE_TIMEOUT_S`, `BOBE_SLEEP_PHRASE`. Wake-word gating is always on: say the wake phrase to stream, `go to sleep` to stop.

Claude Code launching is disabled by default. To enable it, set a launch-specific token on both robot and Mac, then say `confirm launch Claude Code` after BoBe asks for confirmation.

## Configuration

Copy `.env.example` to `.env` for local development and set the keys you need:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
CLAUDE_MODEL="claude-sonnet-4-6"
BOBE_WAKE_BACKEND=remote
BOBE_WAKE_REMOTE_URL=ws://Mac.local:8765/v1/stream
BOBE_WAKE_TOKEN=
# Optional: required for confirmed Claude Code voice launch
BOBE_CLAUDE_CODE_LAUNCH_TOKEN=
```

The OpenAI key is used by the inherited realtime speech bridge. The Anthropic key is used by BoBe's `ask_claude` tool for Claude-backed answers.

## Remote wake runbook (Mac + robot)

BoBe wake detection runs on a Mac host. The robot streams microphone PCM over WebSocket while asleep; the Mac runs Whisper and sends a wake event when it hears `Hey Jarvis`.

### 1. Mac: start the wake daemon

On the Mac that will listen for the wake phrase (same LAN as the robot):

```bash
uv sync --extra wake-daemon
export BOBE_WAKE_TOKEN="$(openssl rand -hex 16)"   # pick a shared secret
echo "BOBE_WAKE_TOKEN=$BOBE_WAKE_TOKEN" >> .env
uv run bobe-wake-daemon
```

Defaults: WebSocket on port **8765**, path `/v1/stream`, Whisper model `base.en`. Optional tuning: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `VAD_*` (see `.env.example`).

Note the Mac hostname or IP (e.g. `Mac.local` or `192.168.1.114`).

To opt in to Claude Code voice launch on the Mac, add a separate launch token to `config/wake-daemon.env`:

```env
BOBE_CLAUDE_CODE_LAUNCH_ENABLED=1
BOBE_CLAUDE_CODE_LAUNCH_TOKEN=<separate long random secret>
BOBE_CLAUDE_CODE_WORKDIR=~/repos/bobe-claude-code-workspace
BOBE_CLAUDE_CODE_BIN=claude
```

### 2. Robot: configure wake settings

Set these in the robot app instance `.env` (Reachy settings UI or instance file):

```env
BOBE_WAKE_BACKEND=remote
BOBE_WAKE_REMOTE_URL=ws://Mac.local:8765/v1/stream
BOBE_WAKE_TOKEN=<same secret as Mac>
BOBE_WAKE_GAIN=1.75
BOBE_CLAUDE_CODE_LAUNCH_TOKEN=<same Claude Code launch secret as Mac>
# Optional; derived from BOBE_WAKE_REMOTE_URL when unset:
# BOBE_CLAUDE_CODE_LAUNCH_URL=http://Mac.local:8765/v1/launch/claude-code
```

Restart the BoBe app after saving. The settings page at `/wake-config` can persist the same values when running headless.

### 3. Verify pairing

1. Daemon running on Mac; firewall allows inbound **8765** from the robot.
2. Start BoBe on the robot; check `/status`: `wake_enabled`, `wake_backend=remote`, `wake_debug.connected=true`.
3. Say **Hey Jarvis** → chime + antennas up; Realtime session opens.
4. Optional Claude Code launch: ask BoBe to launch Claude Code, then say **confirm launch Claude Code** when prompted.
5. Say **go to sleep** (or wait for `BOBE_WAKE_TIMEOUT_S`) → session closes.

### 4. Optional: deploy script

If you use the robot apps API, `scripts/deploy_robot_wake.py` can install/update BoBe and push wake env vars in one step (requires robot API on port 8000 and a local `.env` with `BOBE_WAKE_TOKEN`).

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
- `src/bobe/profiles/_bobe_locked_profile/launch_claude_code.py`: confirmed Claude Code launch request tool.

