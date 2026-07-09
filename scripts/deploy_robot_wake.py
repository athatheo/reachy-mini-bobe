#!/usr/bin/env python3
"""Deploy BoBe to the robot and configure remote wake via daemon APIs."""

from __future__ import annotations
import json
import time
import argparse
import urllib.error
import urllib.request
from pathlib import Path


def _read_token(daemon_env: Path) -> str:
    for line in daemon_env.read_text(encoding="utf-8").splitlines():
        if line.startswith("BOBE_WAKE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"BOBE_WAKE_TOKEN not found in {daemon_env}")


def _request(method: str, url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def _wait_job(robot_host: str, job_id: str, timeout_s: float = 600.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        last = _request("GET", f"http://{robot_host}:8000/api/apps/job-status/{job_id}")
        status = last.get("status")
        if status in {"completed", "failed", "done"}:
            if status == "failed":
                return last
            if status in {"completed", "done"}:
                return {**last, "status": "completed"}
        time.sleep(5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s: {last.get('status')}")


def deploy_robot_app(robot_host: str, app_name: str = "bobe", space_id: str = "athatheo/bobe") -> None:
    installed = _request("GET", f"http://{robot_host}:8000/api/apps/list-available/installed")
    is_installed = any(item.get("name") == app_name for item in (installed or []))

    status = _request("GET", f"http://{robot_host}:8000/api/apps/current-app-status")
    if status and status.get("state") == "running":
        print("Stopping current app...")
        _request("POST", f"http://{robot_host}:8000/api/apps/stop-current-app")

    if is_installed:
        print(f"Updating app '{app_name}'...")
        job = _request("POST", f"http://{robot_host}:8000/api/apps/update/{app_name}")
    else:
        print(f"Installing app '{app_name}' from {space_id}...")
        job = _request(
            "POST",
            f"http://{robot_host}:8000/api/apps/install",
            {
                "name": app_name,
                "source_kind": "hf_space",
                "url": space_id,
                "description": "BoBe assistant",
            },
        )
    job_id = job.get("job_id")
    if not job_id:
        raise SystemExit(f"update did not return job_id: {job}")
    print(f"Update job: {job_id}")
    result = _wait_job(robot_host, job_id)
    if result.get("status") != "completed":
        logs = result.get("logs") or []
        raise SystemExit(f"update failed: {(logs[-1] if logs else result)}")

    print("Starting app...")
    _request("POST", f"http://{robot_host}:8000/api/apps/start-app/{app_name}")


def configure_remote_wake(robot_host: str, mac_host: str, token: str, port: int = 8765) -> None:
    remote_url = f"ws://{mac_host}:{port}/v1/stream"
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            _request(
                "POST",
                f"http://{robot_host}:7860/wake-config",
                {
                    "backend": "remote",
                    "remote_url": remote_url,
                    "token": token,
                    "gain": 1.75,
                },
                timeout=10,
            )
            print(f"Configured wake-config: {remote_url}")
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                time.sleep(3)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(3)
    raise TimeoutError("BoBe settings server did not expose /wake-config in time")


def restart_robot_app(robot_host: str) -> None:
    _request("POST", f"http://{robot_host}:8000/api/apps/restart-current-app")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy BoBe and configure remote wake on Reachy")
    parser.add_argument("--robot-host", default="192.168.1.117")
    parser.add_argument("--mac-host", default="192.168.1.114")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--daemon-env", type=Path, default=Path("config/wake-daemon.env"))
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-configure", action="store_true")
    args = parser.parse_args()

    token = _read_token(args.daemon_env)
    if not args.skip_deploy:
        deploy_robot_app(args.robot_host)
    if not args.skip_configure:
        configure_remote_wake(args.robot_host, args.mac_host, token, args.port)
        restart_robot_app(args.robot_host)
    print("Done.")


if __name__ == "__main__":
    main()
