"""MicrowaveOS entry point.

Usage:
    microwaveos              # REPL mode (default)
    microwaveos --telegram   # Telegram bot
    microwaveos --http       # HTTP API server
    microwaveos --http --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.pipeline.orchestrator import Orchestrator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


async def run_repl(config, verbose: bool = False) -> None:
    from src.channels.repl import REPLChannel

    orchestrator = Orchestrator(config)
    await orchestrator.start(channel="repl")

    channel = REPLChannel(orchestrator, show_metadata=verbose)
    try:
        await channel.start()
    finally:
        await orchestrator.stop()


async def run_telegram(config) -> None:
    from src.channels.telegram import TelegramChannel

    if not config.telegram_bot_token:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    orchestrator = Orchestrator(config)
    await orchestrator.start(channel="telegram")

    channel = TelegramChannel(orchestrator, config.telegram_bot_token)
    try:
        await channel.start()
        # Keep running until interrupted
        await asyncio.Event().wait()
    finally:
        await channel.stop()
        await orchestrator.stop()


async def run_http(config, host: str = "127.0.0.1", port: int = 8080) -> None:
    from src.channels.http import HTTPChannel

    orchestrator = Orchestrator(config)
    await orchestrator.start(channel="http")

    channel = HTTPChannel(orchestrator, host=host, port=port)
    try:
        await channel.start()
        # Keep running until interrupted
        await asyncio.Event().wait()
    finally:
        await channel.stop()
        await orchestrator.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="MicrowaveOS — cognitive agent runtime")
    parser.add_argument("--telegram", action="store_true", help="Run Telegram bot")
    parser.add_argument("--http", action="store_true", help="Run HTTP API server")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging + show pipeline metadata")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = load_config()

    try:
        if args.telegram:
            asyncio.run(run_telegram(config))
        elif args.http:
            asyncio.run(run_http(config, host=args.host, port=args.port))
        else:
            asyncio.run(run_repl(config, verbose=args.verbose))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
