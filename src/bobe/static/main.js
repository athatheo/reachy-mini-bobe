async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

let statusPollInFlight = false;

async function fetchStatus() {
  const url = new URL("/status", window.location.origin);
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, {}, 2000);
  if (!resp.ok) return null;
  return resp.json();
}

async function pollStatusOnce() {
  if (statusPollInFlight) return null;
  statusPollInFlight = true;
  try {
    return await fetchStatus();
  } catch (e) {
    return null;
  } finally {
    statusPollInFlight = false;
  }
}

function updateStatusUI(st, loading) {
  renderWakeStatus(st);
  renderWakeDebug(st);
  renderCredentials(st);
  if (st.has_key) show(loading, false);
}

function startStatusPolling({ loading, loadingText, onInitialReady, timeoutMs = 15000 }) {
  let attempts = 0;
  let timedOut = false;
  let statusReady = false;
  const deadline = Date.now() + timeoutMs;

  const tick = async () => {
    const st = await pollStatusOnce();
    if (st) {
      updateStatusUI(st, loading);
      if (!statusReady) {
        statusReady = true;
        onInitialReady(st);
      }
      return;
    }
    if (statusReady) return;

    attempts += 1;
    if (loadingText) {
      if (Date.now() >= deadline) {
        loadingText.textContent = "BoBe is still starting\u2026";
        if (!timedOut) {
          timedOut = true;
          onInitialReady(null);
        }
      } else {
        loadingText.textContent = attempts > 8 ? "Starting backend..." : "Loading...";
      }
    } else if (Date.now() >= deadline && !timedOut) {
      timedOut = true;
      onInitialReady(null);
    }
  };

  tick();
  setInterval(tick, 1000);
}

async function saveKeys({ openaiApiKey, anthropicApiKey, claudeModel }) {
  const resp = await fetch("/api_keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      openai_api_key: openaiApiKey,
      anthropic_api_key: anthropicApiKey,
      claude_model: claudeModel,
    }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "save_failed");
  }
  return data;
}

function looksLikeOpenAIKey(value) {
  return value.startsWith("sk-") && value.length >= 20;
}

function looksLikeAnthropicKey(value) {
  return value.startsWith("sk-ant-") && value.length >= 20;
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

function renderCredentials(st) {
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  if (!formPanel || !configuredPanel) return;
  if (st.has_key) {
    show(configuredPanel, true);
    show(formPanel, false);
  } else {
    show(configuredPanel, false);
    show(formPanel, true);
  }
}

function renderWakeStatus(st) {
  const panel = document.getElementById("live-status");
  const chip = document.getElementById("wake-chip");
  const text = document.getElementById("wake-text");
  if (!panel || !chip || !text) return;
  show(panel, true);

  if (!st.wake_enabled) {
    if (st.wake_error) {
      chip.textContent = "Wake misconfigured";
      chip.className = "chip chip-warn";
      text.textContent =
        "Wake-word detection is unavailable: " +
        st.wake_error +
        " BoBe is in always-on mode (mic streams continuously). Fix wake settings and restart.";
    } else {
      chip.textContent = "Always on";
      chip.className = "chip";
      text.textContent = "Wake-word gating is disabled: audio streams continuously while the app runs.";
    }
  } else if (st.awake) {
    chip.textContent = "Awake \u00b7 streaming";
    chip.className = "chip";
    const mins = Math.round((st.wake_timeout_s || 300) / 60);
    text.textContent =
      "Conversation window open: audio is streaming to OpenAI. " +
      "Say 'go to sleep' or stay quiet for " + mins + " minutes to close it.";
  } else if (st.wake_backend === "remote") {
    chip.textContent = "Asleep \u00b7 Mac wake";
    chip.className = "chip chip-ok";
    text.textContent =
      "Mic audio streams to the Mac Whisper daemon while asleep. Say 'Hey Jarvis' to wake BoBe.";
  } else {
    chip.textContent = "Asleep \u00b7 local only";
    chip.className = "chip chip-ok";
    text.textContent =
      "Listening locally for 'Hey Jarvis'. No audio leaves the robot until the wake word is heard.";
  }
}

function formatMetric(label, value) {
  return `<div><span>${label}</span>${value ?? "\u2014"}</div>`;
}

function renderWakeDebug(st) {
  const panel = document.getElementById("wake-debug");
  const chip = document.getElementById("wake-debug-chip");
  const metrics = document.getElementById("wake-debug-metrics");
  const streamEl = document.getElementById("wake-transcript-stream");
  const logEl = document.getElementById("wake-debug-log");
  if (!panel || !chip || !metrics || !streamEl || !logEl) return;

  const debug = st.wake_debug || {};
  const remote = debug.remote_stats || {};
  const showPanel = st.wake_enabled && (st.wake_backend === "remote" || debug.backend === "remote");
  show(panel, showPanel);
  if (!showPanel) return;

  const connected = Boolean(debug.connected);
  const listenMode = debug.listen_mode || (debug.paused ? "sleep" : "wake");
  chip.textContent = connected
    ? listenMode === "sleep"
      ? "Connected \u00b7 sleep listen"
      : "Connected \u00b7 wake listen"
    : "Disconnected";
  chip.className = connected ? "chip chip-ok" : "chip";

  const transcript = remote.transcript || debug.transcript_last || "";
  const partial = remote.partial || debug.transcript_partial || "";
  const rms = remote.rms ?? debug.rms_last ?? 0;
  const inSpeech = remote.in_speech ? "yes" : "no";
  const latency = remote.latency_ms_last ?? remote.latency_ms ?? debug.latency_ms_last ?? "\u2014";
  const engine = debug.daemon_engine || remote.engine || "faster-whisper";
  const model = remote.model || "\u2014";
  const url = st.wake_remote_url || debug.url || "\u2014";

  metrics.innerHTML = [
    formatMetric("Daemon URL", url),
    formatMetric("Engine", `${engine} / ${model}`),
    formatMetric("Latest transcript", transcript || "\u2014"),
    formatMetric("Live partial", partial || "\u2014"),
    formatMetric("Mic RMS", Number(rms).toFixed ? Number(rms).toFixed(1) : rms),
    formatMetric("In speech", inSpeech),
    formatMetric("Whisper latency (ms)", latency),
  ].join("");

  const stream = debug.transcript_display?.length
    ? debug.transcript_display
    : (debug.transcript_stream ?? [])
        .map((entry) => {
          const text = entry.text || "";
          return text ? (entry.partial ? `[live] ${text}` : `[final] ${text}`) : "";
        })
        .filter(Boolean);

  if (listenMode === "sleep" && connected && stream.length === 0 && !partial && !transcript) {
    streamEl.textContent =
      "Whisper is listening for \"go to sleep\" while BoBe is awake.";
  } else if (stream.length > 0) {
    streamEl.textContent = stream.join("\n");
  } else if (partial) {
    streamEl.textContent = `[live] ${partial}`;
  } else if (transcript) {
    streamEl.textContent = `[final] ${transcript}`;
  } else if (remote.in_speech) {
    streamEl.textContent = "[live] listening…";
  } else {
    streamEl.textContent = connected
      ? "Listening… speak near the robot while BoBe is asleep."
      : "Not connected to the Mac wake daemon yet.";
  }
  streamEl.scrollTop = streamEl.scrollHeight;

  const events = Array.isArray(debug.events) ? debug.events : [];
  const connectionEvents = events.filter((entry) => entry.level !== "transcript");
  if (connectionEvents.length === 0) {
    logEl.textContent = connected
      ? "Waiting for daemon events..."
      : "Not connected to the Mac wake daemon yet.";
    return;
  }

  logEl.innerHTML = connectionEvents
    .slice(-20)
    .map((entry) => {
      const level = entry.level || "info";
      const time = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "";
      const msg = entry.message || "";
      return `<span class="log-${level}">[${time}] ${msg}</span>`;
    })
    .join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function markError(input, flag) {
  input.classList.toggle("error", flag);
}

function init() {
  const loading = document.getElementById("loading");
  const loadingText = document.querySelector("#loading p");
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const openaiInput = document.getElementById("openai-api-key");
  const anthropicInput = document.getElementById("anthropic-api-key");
  const modelInput = document.getElementById("claude-model");

  show(loading, true);
  show(formPanel, false);
  show(configuredPanel, false);

  startStatusPolling({
    loading,
    loadingText,
    onInitialReady(st) {
      modelInput.value = (st && st.claude_model) || "claude-sonnet-4-6";
      if (!st) {
        show(loading, true);
        show(formPanel, false);
        show(configuredPanel, false);
        return;
      }
      show(loading, false);
      renderCredentials(st);
      wireFormHandlers();
    },
  });

  let formHandlersWired = false;

  function wireFormHandlers() {
    if (formHandlersWired) return;
    formHandlersWired = true;

    changeKeyBtn.addEventListener("click", () => {
      show(configuredPanel, false);
      show(formPanel, true);
      openaiInput.value = "";
      anthropicInput.value = "";
      statusEl.textContent = "";
      statusEl.className = "status";
    });

    for (const input of [openaiInput, anthropicInput, modelInput]) {
      input.addEventListener("input", () => markError(input, false));
    }

    saveBtn.addEventListener("click", async () => {
      const openaiApiKey = openaiInput.value.trim();
      const anthropicApiKey = anthropicInput.value.trim();
      const claudeModel = modelInput.value.trim() || "claude-sonnet-4-6";

      markError(openaiInput, !looksLikeOpenAIKey(openaiApiKey));
      markError(anthropicInput, !looksLikeAnthropicKey(anthropicApiKey));
      markError(modelInput, !claudeModel);

      if (!looksLikeOpenAIKey(openaiApiKey) || !looksLikeAnthropicKey(anthropicApiKey) || !claudeModel) {
        statusEl.textContent = "Please enter valid OpenAI and Anthropic keys plus a Claude model.";
        statusEl.className = "status warn";
        return;
      }

      statusEl.textContent = "Saving private keys...";
      statusEl.className = "status";
      try {
        await saveKeys({ openaiApiKey, anthropicApiKey, claudeModel });
        statusEl.textContent = "Saved. Reloading...";
        statusEl.className = "status ok";
        window.location.reload();
      } catch (e) {
        if (e.message === "invalid_openai_api_key") {
          statusEl.textContent = "OpenAI key should start with sk-.";
        } else if (e.message === "invalid_anthropic_api_key") {
          statusEl.textContent = "Anthropic key should start with sk-ant-.";
        } else {
          statusEl.textContent = "Failed to save keys. Please try again.";
        }
        statusEl.className = "status error";
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", init);
