"""Scheduler engine — poll loop, job dispatch, and fire logic.

Two dispatch paths:
- `llm`  — run the job's prompt through a one-shot LLM call
  (`SingleTurnClient`), optionally wrap the output as an HTML card-view
  attachment, and deliver via the target channel.
- `direct` — send a literal string as a plain text message. Zero LLM cost.

LLM jobs deliberately bypass `Orchestrator.process()` to avoid polluting
the live user session: no shared SDK connection, no recent-turn
contamination, no compaction triggers on your real conversation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from croniter import croniter

from src.config import Config
from src.llm.client import SingleTurnClient
from src.scheduler.card_view import (
    plain_text_fallback,
    render_card_view,
    split_into_cards,
)
from src.scheduler.store import ScheduledJob, SchedulerStore

log = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 30
# If a job's next scheduled fire is more than this many minutes in the past,
# treat it as "missed while offline" and skip forward to the next upcoming
# fire instead of firing a stale backlog. Matches the spec's (b) behavior.
MAX_CATCHUP_MINUTES = 5


# A channel "sender" is anything that can deliver a text message and
# optionally an attachment. The scheduler doesn't depend on specific Channel
# subclasses — it takes a dict of named senders.
class ChannelSender:
    """Protocol-like wrapper — any object with send_text() + send_attachment()."""

    async def send_text(self, recipient: str, text: str) -> None: ...
    async def send_attachment(
        self, recipient: str, filename: str, content: str | bytes
    ) -> None: ...


class Scheduler:
    def __init__(
        self,
        store: SchedulerStore,
        channels: dict[str, ChannelSender],
        config: Config,
    ):
        self.store = store
        self.channels = channels
        self.config = config
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Per-job locks prevent a second fire of the same job from running
        # while the previous one is still in flight (slow LLM call, etc.)
        self._locks: dict[int, asyncio.Lock] = {}

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop())
        log.info("Scheduler started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # --- loop ---

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                log.exception("Scheduler tick failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=POLL_INTERVAL_SEC
                )
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        for job in self.store.list_enabled():
            try:
                should_fire = self._should_fire(job, now_utc)
            except Exception as e:
                log.warning(f"Job {job.name!r} schedule eval failed: {e}")
                continue
            if should_fire:
                asyncio.create_task(self._guarded_fire(job))

    def _should_fire(self, job: ScheduledJob, now_utc: datetime) -> bool:
        """Decide whether to fire `job` on this tick.

        Also performs the skip-stale baseline update: if a job hasn't fired
        for a long time (daemon was offline), we set last_run_at to a recent
        baseline without actually firing, so catch-up backlogs don't flood
        the user's phone on restart.
        """
        tz = ZoneInfo(job.timezone)
        base_local = self._base_time(job, now_utc).astimezone(tz)

        itr = croniter(job.cron_expr, base_local)
        next_fire_local = itr.get_next(datetime)
        next_fire_utc = next_fire_local.astimezone(ZoneInfo("UTC"))

        # Upcoming — not yet due.
        if next_fire_utc > now_utc:
            return False

        # Due, but more than the catch-up threshold stale — skip, record baseline.
        age_min = (now_utc - next_fire_utc).total_seconds() / 60.0
        if age_min > MAX_CATCHUP_MINUTES:
            log.info(
                f"Job {job.name!r} missed scheduled fire at {next_fire_local.isoformat()} "
                f"({age_min:.0f} min ago); skipping to next upcoming fire"
            )
            if job.id is not None:
                self.store.mark_baseline(job.id, now_utc)
            return False

        return True

    def _base_time(self, job: ScheduledJob, now_utc: datetime) -> datetime:
        """Compute the base time croniter should start iterating from."""
        if job.last_run_at:
            ts = job.last_run_at
            if ts.tzinfo is None:
                # Stored as naive UTC via datetime('now'). Tag it.
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))
            return ts
        # First run ever: start one minute ago so a cron matching "right now"
        # fires at the next tick rather than waiting a full cron cycle.
        return now_utc - timedelta(minutes=1)

    async def _guarded_fire(self, job: ScheduledJob) -> None:
        """Fire a job with per-job locking to prevent overlap."""
        if job.id is None:
            return
        lock = self._locks.setdefault(job.id, asyncio.Lock())
        if lock.locked():
            log.info(f"Skipping overlap for job {job.name!r} — previous fire still running")
            return
        async with lock:
            try:
                await self._fire(job)
                self.store.mark_ran(job.id, error=None)
            except Exception as e:
                log.exception(f"Job {job.name!r} failed")
                self.store.mark_ran(job.id, error=str(e)[:500])

    async def _fire(self, job: ScheduledJob) -> None:
        """Execute a job's actual work. No DB state touched here."""
        log.info(f"Firing job {job.name!r} ({job.mode}) → {job.target_channel}:{job.recipient_id}")
        channel = self.channels.get(job.target_channel)
        if channel is None:
            raise RuntimeError(
                f"No channel registered for {job.target_channel!r}; "
                f"known: {sorted(self.channels)}"
            )

        if job.mode == "direct":
            await channel.send_text(job.recipient_id, job.prompt_or_text)
            return

        if job.mode == "llm":
            response = await self._run_llm(job)
            await self._deliver_llm(job, channel, response)
            return

        raise RuntimeError(f"Unknown job mode: {job.mode!r}")

    async def _run_llm(self, job: ScheduledJob) -> str:
        """One-shot LLM call — no session, no pipeline.

        Uses SingleTurnClient so the job's prompt doesn't touch the
        user's live SDK connection or turn history.
        """
        client = SingleTurnClient(
            model=self.config.model_main,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
        )
        # SingleTurnClient takes (system_prompt, user_message). For scheduler
        # jobs the prompt IS the task — put it in user_message with an
        # empty/neutral system prompt so the prompt is treated as the
        # authoritative instruction.
        return await client.query(
            system_prompt="You are executing a scheduled task. Follow the user's "
            "instructions exactly and return only the requested output.",
            user_message=job.prompt_or_text,
        )

    async def _deliver_llm(
        self, job: ScheduledJob, channel: ChannelSender, response: str
    ) -> None:
        """Ship the LLM output via the target channel.

        In card-view mode, parse the response into cards and deliver as:
        - short plain-text message body (for desktop/fallback)
        - HTML card-view attachment (per-card Copy buttons, mobile-friendly)
        """
        response = response.strip()
        if not response:
            await channel.send_text(job.recipient_id, f"_{job.name}: (empty response)_")
            return

        if not job.card_view:
            await channel.send_text(job.recipient_id, response)
            return

        cards = split_into_cards(response, separator=job.card_split)
        if len(cards) < 2:
            # Not really list-shaped — just send the text.
            await channel.send_text(job.recipient_id, response)
            return

        title = job.name
        html_doc = render_card_view(cards, title=title)
        plain = plain_text_fallback(cards)

        # Message body first so the user sees something instantly on
        # platforms that don't preview HTML quickly.
        header = f"_{title} — {len(cards)} items, tap the attachment for copy buttons._"
        await channel.send_text(job.recipient_id, f"{header}\n\n{plain}")

        filename = _safe_filename(title) + ".html"
        await channel.send_attachment(job.recipient_id, filename, html_doc)

    # --- manual invocation ---

    async def run_once(self, name: str) -> None:
        """Fire a named job right now, regardless of its schedule.

        Used by the `scheduler run <name>` CLI and for testing.
        """
        job = self.store.get_by_name(name)
        if job is None:
            raise RuntimeError(f"No job named {name!r}")
        if job.id is None:
            raise RuntimeError("Job has no id (not persisted)")
        await self._guarded_fire(job)


def _safe_filename(s: str) -> str:
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")
    return cleaned or "output"
