---
title: Reachy Mini BoBe
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
short_description: Reachy Mini voice assistant named BoBe.
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Mini BoBe

BoBe is a Reachy Mini assistant foundation. It starts from Pollen Robotics' official conversation app template and locks the robot personality to a BoBe profile.

## Current milestone

- Remote wake word: say `Hey Bobe` to wake BoBe (Mac-side Whisper daemon; robot streams PCM while asleep).
- Conversations run entirely through [Hermes](https://github.com/NousResearch/hermes-agent): the wake daemon's converse mode transcribes utterances locally, the Hermes agent answers, and its TTS reply audio plays through the robot's speaker. No cloud realtime-speech API is involved.
- Hermes can push spoken announcements and TTS clips to the robot ("send it to bobe", cron `deliver=bobe`).
- Expressive robot motion via the Reachy Mini motion tools (`play_emotion`, `move_head`, `sweep_look`).

## Privacy model

- While asleep, microphone PCM streams from the robot to a Mac-side wake daemon over WebSocket; the daemon listens for the wake phrase with local Whisper. Audio never leaves the LAN.
- Saying `Hey Bobe` opens a conversation window (chime + antennas up). Awake audio still only reaches the Mac daemon; utterance *transcripts* go to your Hermes agent, whose configured LLM/TTS providers may be cloud services.
- The window closes (chime + antennas relaxed) when you say `go to sleep` (or Greek `κοιμήσου`) or after `BOBE_WAKE_TIMEOUT_S` (default 300s) without session activity.
- Tune with `BOBE_WAKE_REMOTE_URL`, `BOBE_WAKE_TOKEN`, `BOBE_WAKE_GAIN`, `BOBE_WAKE_TIMEOUT_S`, `BOBE_WAKE_PHRASE`, `BOBE_SLEEP_PHRASE`.

## Configuration

Copy `.env.example` to `.env` for local development and set the keys you need:

```env
BOBE_WAKE_BACKEND=remote
BOBE_WAKE_REMOTE_URL=ws://Mac.local:8765/v1/stream
BOBE_WAKE_TOKEN=
# Hermes personal-agent integration (see "Hermes integration")
BOBE_HERMES_URL=http://192.168.1.172:8642/v1
BOBE_HERMES_API_KEY=
```

## Hermes integration

Hermes **is** BoBe's voice pipeline (requires Hermes ≥ 0.20.0 with the
`integrations/hermes-bobe-plugin/` installed on the Mac):

- **Voice → Hermes**: while awake the robot streams mic PCM to the Mac wake
  daemon, whose **converse mode** segments utterances with local Whisper and
  hands the transcripts to the Hermes `bobe` platform plugin
  (`GET /v1/utterances` long-poll). Hermes answers as the full agent (tools,
  kanban, memory).
- **Hermes → BoBe**: reply audio (auto-TTS) is POSTed to the daemon's
  `POST /v1/speak`, decoded to PCM, and played through the robot's speaker
  with head-wobble sync. Plain-text pushes go to `POST /v1/announce` and are
  synthesized by the plugin when possible. The robot wakes first if asleep.

Turn-taking is half-duplex: the daemon pauses capture while a reply plays
(there is no echo cancellation). Sleep phrases and the inactivity timeout
behave the same asleep and awake.

Setup:

1. **Enable the Hermes API server** (Mac, `~/.hermes/.env`): `API_SERVER_ENABLED=true`, `API_SERVER_KEY=$(openssl rand -hex 32)`, `API_SERVER_HOST=<Mac LAN IP>`, then `hermes gateway restart`.
2. **Robot env**: set `BOBE_HERMES_URL=http://<Mac LAN IP>:8642/v1` and `BOBE_HERMES_API_KEY=<same key>`.
3. **Install the plugin** (Mac): copy `integrations/hermes-bobe-plugin/` to `~/.hermes/plugins/bobe/`, set `BOBE_WAKE_TOKEN` in `~/.hermes/.env` (the wake daemon's token), enable `platforms: bobe: enabled: true` in `~/.hermes/config.yaml`, then `hermes gateway restart`.

Verify: `curl -X POST http://127.0.0.1:8765/v1/announce -H "X-BoBe-Wake-Token: $BOBE_WAKE_TOKEN" -H 'Content-Type: application/json' -d '{"message":"hello"}'` with the robot app running → the robot says "hello".

## Remote wake runbook (Mac + robot)

BoBe wake detection runs on a Mac host. The robot streams microphone PCM over WebSocket while asleep; the Mac runs Whisper and sends a wake event when it hears `Hey Bobe`.

### 1. Mac: start the wake daemon

On the Mac that will listen for the wake phrase (same LAN as the robot):

```bash
uv sync --extra wake-daemon
export BOBE_WAKE_TOKEN="$(openssl rand -hex 16)"   # pick a shared secret
echo "BOBE_WAKE_TOKEN=$BOBE_WAKE_TOKEN" >> .env
uv run bobe-wake-daemon
```

Defaults: WebSocket on port **8765**, path `/v1/stream`, Whisper model `distil-small.en`. Optional tuning: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `VAD_*` (see `.env.example`).

Note the Mac hostname or IP (e.g. `Mac.local` or `192.168.1.114`).

### 2. Robot: configure wake settings

Set these in the robot app instance `.env` (Reachy settings UI or instance file):

```env
BOBE_WAKE_BACKEND=remote
BOBE_WAKE_REMOTE_URL=ws://Mac.local:8765/v1/stream
BOBE_WAKE_TOKEN=<same secret as Mac>
BOBE_WAKE_GAIN=1.75
```

Restart the BoBe app after saving. The settings page at `/wake-config` can persist the same values when running headless.

### 3. Verify pairing

1. Daemon running on Mac; firewall allows inbound **8765** from the robot.
2. Start BoBe on the robot; check `/status`: `wake_enabled`, `wake_backend=remote`, `wake_debug.connected=true`.
3. Say **Hey Bobe** → chime + antennas up; Realtime session opens.
4. Say **go to sleep** (or wait for `BOBE_WAKE_TIMEOUT_S`) → session closes.

### 4. Optional: deploy script

If you use the robot apps API, `scripts/deploy_robot_wake.py` can install/update BoBe and push wake env vars in one step (requires robot API on port 8000 and a local `.env` with `BOBE_WAKE_TOKEN`).

## Development with uv

```bash
uv sync --group dev
uv run pytest
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
