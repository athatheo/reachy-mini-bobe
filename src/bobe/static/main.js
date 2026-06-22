const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function waitForStatus(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend..." : "Loading...";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
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
    chip.textContent = "Always on";
    chip.className = "chip";
    text.textContent = "Wake-word gating is disabled: audio streams continuously while the app runs.";
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
  const paused = Boolean(debug.paused);
  chip.textContent = connected ? (paused ? "Connected \u00b7 paused" : "Connected") : "Disconnected";
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

  const stream = Array.isArray(debug.transcript_stream) ? debug.transcript_stream : [];
  if (partial) {
    streamEl.textContent = stream
      .map((entry) => {
        const text = entry.text || "";
        return text ? `[final] ${text}` : "";
      })
      .filter(Boolean)
      .concat(partial ? [`[live] ${partial}`] : [])
      .join("\n");
  } else if (stream.length === 0) {
    streamEl.textContent = connected
      ? "Listening... partial Whisper text will appear here while you speak."
      : "Connect to the Mac wake daemon to see live Whisper transcripts.";
  } else {
    streamEl.textContent = stream
      .map((entry) => `[final] ${entry.text || ""}`)
      .filter((line) => line !== "[final] ")
      .join("\n");
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

function startWakeStatusPolling() {
  const poll = async () => {
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) {
        const st = await resp.json();
        renderWakeStatus(st);
        renderWakeDebug(st);
        renderCredentials(st);
      }
    } catch (e) {}
  };
  poll();
  setInterval(poll, 2000);
}

function markError(input, flag) {
  input.classList.toggle("error", flag);
}

async function init() {
  const loading = document.getElementById("loading");
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

  const st = (await waitForStatus()) || { has_key: false, claude_model: "claude-sonnet-4-6" };
  modelInput.value = st.claude_model || "claude-sonnet-4-6";

  if (st.has_key) {
    show(configuredPanel, true);
  } else {
    show(formPanel, true);
  }
  renderCredentials(st);
  show(loading, false);
  startWakeStatusPolling();

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

window.addEventListener("DOMContentLoaded", init);
