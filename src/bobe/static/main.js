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
  } else {
    chip.textContent = "Asleep \u00b7 local only";
    chip.className = "chip chip-ok";
    text.textContent =
      "Listening locally for 'Hey Jarvis'. No audio leaves the robot until the wake word is heard.";
  }
}

function startWakeStatusPolling() {
  const poll = async () => {
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) renderWakeStatus(await resp.json());
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
