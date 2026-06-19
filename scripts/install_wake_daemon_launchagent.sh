#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.bobe.wake-daemon"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
RUN_SCRIPT="${ROOT}/scripts/run_wake_daemon.sh"
LOG_DIR="${HOME}/Library/Logs"
STDOUT_LOG="${LOG_DIR}/bobe-wake-daemon.log"
STDERR_LOG="${LOG_DIR}/bobe-wake-daemon.err.log"
ENV_FILE="${ROOT}/config/wake-daemon.env"
ENV_EXAMPLE="${ROOT}/config/wake-daemon.env.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "s/change-me-to-a-long-random-string/${TOKEN}/" "${ENV_FILE}"
  else
    sed -i "s/change-me-to-a-long-random-string/${TOKEN}/" "${ENV_FILE}"
  fi
  echo "Created ${ENV_FILE} with a generated BOBE_WAKE_TOKEN."
fi

chmod +x "${RUN_SCRIPT}"

mkdir -p "${LOG_DIR}" "${HOME}/Library/LaunchAgents"

cat > "${PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUN_SCRIPT}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${STDOUT_LOG}</string>
  <key>StandardErrorPath</key>
  <string>${STDERR_LOG}</string>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${PLIST}"
launchctl enable "gui/${UID}/${LABEL}"
launchctl kickstart -k "gui/${UID}/${LABEL}"

echo "Installed and started ${LABEL}"
echo "Logs: ${STDOUT_LOG} ${STDERR_LOG}"
echo "Token: $(grep '^BOBE_WAKE_TOKEN=' "${ENV_FILE}" | cut -d= -f2-)"
echo
echo "Next: configure the robot app instance .env with the same token:"
echo "  python3 scripts/configure_wake_remote_env.py --env src/bobe/.env --mac-host $(hostname -s)"
