"""Entrypoint for the Mac-side BoBe wake daemon."""

from __future__ import annotations
import logging
import argparse

import uvicorn

from bobe.wake_daemon.config import load_wake_daemon_config
from bobe.wake_daemon.server import create_app


def main() -> None:
    """Run the wake daemon HTTP/WebSocket server."""
    parser = argparse.ArgumentParser(description="BoBe Mac wake daemon (faster-whisper)")
    parser.add_argument("--host", default=None, help="Bind host (default: WAKE_DAEMON_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: WAKE_DAEMON_PORT)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_wake_daemon_config()
    host = args.host or config.host
    port = args.port or config.port

    uvicorn.run(
        lambda: create_app(config),
        factory=True,
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
