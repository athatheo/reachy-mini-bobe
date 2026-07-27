#!/usr/bin/env python3
"""Try to restart BoBe on a Reachy robot; print recovery steps if the apps API is wedged."""

from __future__ import annotations
import sys
import argparse
import urllib.error

from _common import _request


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart BoBe or print robot recovery steps")
    parser.add_argument("--robot-host", default="192.168.1.117")
    parser.add_argument("--app-name", default="bobe")
    args = parser.parse_args()

    base = f"http://{args.robot_host}:8000"
    try:
        status = _request("GET", f"{base}/api/apps/current-app-status", timeout=8.0)
        print("Current app status:", status)
        state = status.get("state")
        if state == "running":
            print("Restarting current app...")
            _request("POST", f"{base}/api/apps/restart-current-app", timeout=30.0)
        else:
            print(f"Starting {args.app_name}...")
            _request("POST", f"{base}/api/apps/start-app/{args.app_name}", timeout=30.0)
        print("Done. Check http://%s:7860/status in ~30s." % args.robot_host)
    except (TimeoutError, urllib.error.URLError) as exc:
        print("Reachy apps API is not responding (%s)." % exc, file=sys.stderr)
        print(
            "\nRecovery steps:\n"
            "1. Power-cycle the robot or run: sudo systemctl restart reachy-mini-daemon\n"
            "2. Wait ~60s, then start BoBe from the Reachy dashboard\n"
            "3. Open http://%s:7860/ and confirm OpenAI + Anthropic keys\n"
            "4. On the Mac: uv run bobe-wake-daemon\n"
            "5. Redeploy if needed: uv run python scripts/deploy_robot_wake.py\n"
            % args.robot_host,
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
