"""`microwaveos scheduler ...` subcommand implementation.

Kept separate from main.py so the scheduler's argparse surface doesn't
collide with the daemon flags (--signal, --telegram, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from croniter import croniter

from src.config import load_config
from src.scheduler.store import ScheduledJob, SchedulerStore

log = logging.getLogger(__name__)


def scheduler_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos scheduler",
        description="Manage scheduled jobs (recurring LLM tasks and fixed reminders).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # list
    sub.add_parser("list", help="Show all jobs")

    # add
    p_add = sub.add_parser("add", help="Create a new job")
    p_add.add_argument("--name", required=True, help="Unique job name (used in CLI commands)")
    p_add.add_argument("--cron", required=True, help='Cron expression, e.g. "57 6 * * *"')
    p_add.add_argument("--mode", required=True, choices=["llm", "direct"])
    p_add.add_argument("--channel", required=True, help="Target channel: 'signal' or 'telegram'")
    p_add.add_argument("--recipient", required=True, help="Phone number (Signal) or chat id (Telegram)")
    g = p_add.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt-file", type=Path, help="File whose contents become the LLM prompt")
    g.add_argument("--prompt", help="Inline LLM prompt (prefer --prompt-file for long prompts)")
    g.add_argument("--text", help="Literal text to send (direct mode only)")
    p_add.add_argument("--timezone", default="America/New_York")
    p_add.add_argument(
        "--no-card-view", action="store_true",
        help="LLM mode: deliver as plain text instead of HTML card-view",
    )
    p_add.add_argument(
        "--card-split", default="---",
        help="LLM mode: separator that splits output into cards (default: ---)",
    )

    # remove
    p_rm = sub.add_parser("remove", help="Delete a job")
    p_rm.add_argument("name", help="Job name")

    # enable / disable
    p_en = sub.add_parser("enable", help="Enable a job")
    p_en.add_argument("name")
    p_dis = sub.add_parser("disable", help="Disable a job (keeps the row)")
    p_dis.add_argument("name")

    # run (fire once)
    p_run = sub.add_parser("run", help="Fire a job right now, regardless of schedule")
    p_run.add_argument("name")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    store = SchedulerStore(config.db_path)
    store.connect()

    try:
        if args.action == "list":
            return _cmd_list(store)
        if args.action == "add":
            return _cmd_add(store, args)
        if args.action == "remove":
            return _cmd_remove(store, args.name)
        if args.action == "enable":
            return _cmd_set_enabled(store, args.name, True)
        if args.action == "disable":
            return _cmd_set_enabled(store, args.name, False)
        if args.action == "run":
            return _cmd_run(store, config, args.name)
    finally:
        store.close()
    return 1


def _cmd_list(store: SchedulerStore) -> int:
    jobs = store.list_all()
    if not jobs:
        print("(no jobs)")
        return 0
    # Fixed-width layout — readable without external dep
    headers = ("#", "NAME", "CRON", "MODE", "CH", "ENABLED", "LAST RUN", "LAST ERROR")
    rows = []
    for j in jobs:
        rows.append((
            str(j.id or "?"),
            j.name,
            j.cron_expr,
            j.mode,
            j.target_channel,
            "yes" if j.enabled else "no",
            j.last_run_at.strftime("%Y-%m-%d %H:%M") if j.last_run_at else "—",
            (j.last_error or "")[:40],
        ))
    widths = [max(len(r[i]) for r in (headers,) + tuple(rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    return 0


def _cmd_add(store: SchedulerStore, args) -> int:
    # Validate the cron expression up front — croniter raises helpfully
    try:
        croniter(args.cron)
    except Exception as e:
        print(f"Invalid cron expression {args.cron!r}: {e}", file=sys.stderr)
        return 2

    if args.mode == "direct":
        if not args.text:
            print("--mode direct requires --text", file=sys.stderr)
            return 2
        prompt_or_text = args.text
    else:  # llm
        if args.prompt_file:
            if not args.prompt_file.is_file():
                print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
                return 2
            prompt_or_text = args.prompt_file.read_text(encoding="utf-8")
        elif args.prompt:
            prompt_or_text = args.prompt
        else:
            print("--mode llm requires --prompt or --prompt-file", file=sys.stderr)
            return 2

    # Disallow duplicate names — SQLite would raise anyway but catch early.
    if store.get_by_name(args.name) is not None:
        print(f"Job {args.name!r} already exists. Remove it first.", file=sys.stderr)
        return 2

    job = ScheduledJob(
        name=args.name,
        cron_expr=args.cron,
        mode=args.mode,
        prompt_or_text=prompt_or_text,
        target_channel=args.channel,
        recipient_id=args.recipient,
        enabled=True,
        timezone=args.timezone,
        card_split=args.card_split,
        card_view=(args.mode == "llm" and not args.no_card_view),
    )
    store.add(job)
    print(f"Added job {args.name!r} ({args.mode}, {args.cron}, {args.channel}:{args.recipient})")
    return 0


def _cmd_remove(store: SchedulerStore, name: str) -> int:
    if store.remove(name):
        print(f"Removed job {name!r}")
        return 0
    print(f"No job named {name!r}", file=sys.stderr)
    return 1


def _cmd_set_enabled(store: SchedulerStore, name: str, enabled: bool) -> int:
    if store.set_enabled(name, enabled):
        print(f"{'Enabled' if enabled else 'Disabled'} job {name!r}")
        return 0
    print(f"No job named {name!r}", file=sys.stderr)
    return 1


def _cmd_run(store: SchedulerStore, config, name: str) -> int:
    """Fire a job right now. Spins up just enough infrastructure to deliver."""
    job = store.get_by_name(name)
    if job is None:
        print(f"No job named {name!r}", file=sys.stderr)
        return 1

    async def _go() -> None:
        # Build a transient channel matching the job's target.
        channel = await _build_channel(job.target_channel, config)
        from src.scheduler.engine import Scheduler
        scheduler = Scheduler(store=store, channels={job.target_channel: channel}, config=config)
        try:
            await scheduler.run_once(name)
        finally:
            # Be tidy — Signal opens a session; close it.
            closer = getattr(channel, "_session", None)
            if closer:
                await closer.close()

    asyncio.run(_go())
    return 0


async def _build_channel(target: str, config):
    """Build a minimal send-only channel instance for `scheduler run`.

    We don't call `.start()` (no websocket listener, no receiving) — we
    just need the send side. The scheduler talks to it via send_text /
    send_attachment.
    """
    if target == "signal":
        import aiohttp
        from src.channels.signal import SignalChannel

        if not config.signal_rest_url or not config.signal_phone_number:
            raise RuntimeError("Signal env vars not set — see .env")
        ch = SignalChannel(
            orchestrator=None,  # unused for send-only mode
            rest_url=config.signal_rest_url,
            phone_number=config.signal_phone_number,
            allowed_senders=list(config.signal_allowed_senders),
            openai_api_key=config.openai_api_key,
        )
        # We skipped ch.start() but still need an aiohttp session.
        ch._session = aiohttp.ClientSession()
        return ch

    if target == "telegram":
        from src.channels.telegram import TelegramChannel
        from telegram.ext import Application

        if not config.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
        ch = TelegramChannel(orchestrator=None, bot_token=config.telegram_bot_token)
        ch.app = Application.builder().token(config.telegram_bot_token).build()
        await ch.app.initialize()
        return ch

    raise RuntimeError(f"Unknown target channel: {target!r}")
