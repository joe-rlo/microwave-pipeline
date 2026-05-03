"""Scheduler engine — poll loop, job dispatch, and fire logic.

Three dispatch paths:
- `llm`    — run the job's prompt through a one-shot LLM call
             (`SingleTurnClient`), optionally wrap the output as an HTML
             card-view attachment, and deliver via the target channel.
- `direct` — send a literal string as a plain text message. Zero LLM cost.
- `script` — run a shell command (cwd = project root), capture stdout,
             and deliver it. If stdout is a complete HTML document it
             ships as an attachment; otherwise it routes through the same
             card-view delivery as `llm`. Use this for deterministic
             metrics reports, log digests, or anything that needs real
             numbers (LLM mode has no tool access).

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
# Hard cap for `script` mode subprocesses. Long enough for real reports
# (queries, aggregation, HTML rendering) but short enough to catch hangs.
SCRIPT_TIMEOUT_SEC = 300
# How many bytes of stderr to capture in the job's last_error on failure.
# Enough to diagnose without blowing up the DB column.
SCRIPT_STDERR_TAIL_BYTES = 600


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

        if job.mode == "script":
            stdout = await self._run_script(job)
            await self._deliver_script(job, channel, stdout)
            return

        raise RuntimeError(f"Unknown job mode: {job.mode!r}")

    async def _run_llm(self, job: ScheduledJob) -> str:
        """One-shot LLM call — no session, no pipeline.

        Uses SingleTurnClient so the job's prompt doesn't touch the
        user's live SDK connection or turn history.

        If the job references a skill, the skill's body becomes the
        system prompt and any `fetch.py` alongside it is awaited first —
        its output is prepended to the user message so the LLM has
        fresh context (e.g., current open PRs for a github-tool job).
        """
        client = SingleTurnClient(
            model=self.config.model_main,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            workspace_dir=str(self.config.workspace_dir),
        )

        system_prompt, user_message = await self._compose_prompt(job)
        return await client.query(
            system_prompt=system_prompt, user_message=user_message
        )

    async def _compose_prompt(self, job: ScheduledJob) -> tuple[str, str]:
        """Build (system_prompt, user_message) for a job.

        No skill: legacy behavior — generic system prompt, job.prompt_or_text
        as the user message.

        With skill: skill body as system prompt. User message is the skill's
        pre-fetch output (if any) plus the job's trigger text.
        """
        if not job.skill_name:
            return (
                "You are executing a scheduled task. Follow the user's "
                "instructions exactly and return only the requested output.",
                job.prompt_or_text,
            )

        from src.skills import SkillLoader, SkillNotFound
        loader = SkillLoader(self.config.workspace_dir / "skills")
        try:
            skill = loader.load(job.skill_name)
        except SkillNotFound:
            raise RuntimeError(
                f"Job {job.name!r} references missing skill {job.skill_name!r}"
            )

        fetch_output = ""
        if skill.has_fetch:
            try:
                fetch_output = await _run_fetch(
                    skill,
                    context={
                        "job_name": job.name,
                        "recipient": job.recipient_id,
                        "channel": job.target_channel,
                        "timezone": job.timezone,
                    },
                )
            except Exception as e:
                log.warning(f"Pre-fetch for skill {skill.name!r} failed: {e}")
                fetch_output = f"[pre-fetch failed: {e}]"

        parts: list[str] = []
        if fetch_output.strip():
            parts.append(f"[Pre-fetch context]\n{fetch_output.strip()}")
        if job.prompt_or_text.strip():
            parts.append(job.prompt_or_text.strip())
        if not parts:
            # No trigger text AND no fetch output — fall back so the call
            # doesn't ship an empty user message.
            parts.append(f"Run the {skill.name!r} skill.")
        return skill.body, "\n\n".join(parts)

    async def _deliver_llm(
        self, job: ScheduledJob, channel: ChannelSender, response: str
    ) -> None:
        """Ship the LLM output via the target channel.

        Three paths, in priority order:
        1. LLM emitted a complete HTML document → ship as attachment only,
           with a short header text. No duplicate plain-text body. Skills
           that want full styling control (e.g. clickable links) opt in
           by returning HTML directly.
        2. Card-view mode + multi-card output → header + plain-text
           fallback + HTML card-view attachment.
        3. Anything else → plain text.
        """
        response = response.strip()
        if not response:
            await channel.send_text(job.recipient_id, f"_{job.name}: (empty response)_")
            return

        if _looks_like_html_doc(response):
            await self._send_html_attachment(job, channel, response)
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

        # Attachment-only delivery — the HTML card-view IS the content.
        # We used to also post `plain_text_fallback(cards)` as a body, but
        # that double-posted everything. Header alone tells the user
        # there's something to tap.
        header = f"_{title} — {len(cards)} items, tap the attachment for copy buttons._"
        await channel.send_text(job.recipient_id, header)

        filename = _safe_filename(title) + ".html"
        await channel.send_attachment(job.recipient_id, filename, html_doc)

    async def _send_html_attachment(
        self, job: ScheduledJob, channel: ChannelSender, html: str
    ) -> None:
        """Deliver a complete HTML document as a file attachment with a
        terse text header. Used by both LLM-emitted HTML and script-mode
        HTML output. The body intentionally omits the rendered content —
        the attachment IS the content, no duplication."""
        title = job.name
        kb = len(html.encode("utf-8")) / 1024
        header = (
            f"_{title} — {kb:.1f} KB, tap the attachment to view._"
        )
        await channel.send_text(job.recipient_id, header)
        filename = _safe_filename(title) + ".html"
        await channel.send_attachment(job.recipient_id, filename, html)

    # --- script mode ---

    async def _run_script(self, job: ScheduledJob) -> str:
        """Run `job.prompt_or_text` as a shell command, return decoded stdout.

        cwd is the project root so a job command like
        `python3 scripts/weekly_pipeline_report.py` works without absolute
        paths. Shell syntax (pipes, redirects) is supported — the command
        comes from an operator (via the CLI), not user input.

        Non-zero exit codes raise, with a stderr tail bubbled up so it
        lands in the job's `last_error` row.
        """
        project_root = Path(__file__).resolve().parents[2]
        proc = await asyncio.create_subprocess_shell(
            job.prompt_or_text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=SCRIPT_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            raise RuntimeError(
                f"Script timed out after {SCRIPT_TIMEOUT_SEC}s"
            )

        stderr_text = (stderr_b or b"").decode("utf-8", "replace").strip()
        if stderr_text:
            # Log stderr even on success — scripts often emit progress there.
            log.info(f"Script stderr for {job.name!r}: {stderr_text[-500:]}")

        if proc.returncode != 0:
            tail = stderr_text[-SCRIPT_STDERR_TAIL_BYTES:] or "(no stderr)"
            raise RuntimeError(
                f"Script exited {proc.returncode}: {tail}"
            )

        return (stdout_b or b"").decode("utf-8", "replace")

    async def _deliver_script(
        self, job: ScheduledJob, channel: ChannelSender, stdout: str
    ) -> None:
        """Ship script stdout via the target channel.

        If the output is a complete HTML document, send a brief message
        body plus the HTML as an attachment. Otherwise reuse the LLM
        delivery path (card-split → plain text or card-view attachment).
        """
        out = stdout.strip()
        if not out:
            await channel.send_text(
                job.recipient_id, f"_{job.name}: (empty script output)_"
            )
            return

        if _looks_like_html_doc(out):
            await self._send_html_attachment(job, channel, out)
            return

        # Non-HTML: reuse the LLM delivery path. Respects card_view +
        # card_split so a script can still output "---"-separated cards
        # and get the same rich card-view treatment.
        await self._deliver_llm(job, channel, out)

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


def _looks_like_html_doc(s: str) -> bool:
    """True if `s` looks like a complete HTML document.

    Allows leading whitespace or a UTF-8 BOM. Case-insensitive on the
    opening tag so `<HTML>` and `<!DOCTYPE html>` both match.
    """
    lead = s.lstrip("﻿ \n\r\t").lower()
    return lead.startswith("<!doctype") or lead.startswith("<html")


async def _run_fetch(skill, context: dict) -> str:
    """Execute a skill's fetch.py and return its output.

    The script must expose either `async def fetch(context)` or
    `def fetch(context)`. Blocking fetch() runs via asyncio.to_thread so
    it doesn't stall the event loop — useful for subprocess calls like
    the github-tool skill running `gh pr list`.
    """
    import importlib.util
    fetch_path = skill.fetch_path
    if fetch_path is None:
        return ""

    spec = importlib.util.spec_from_file_location(
        f"skill_fetch_{skill.name}", fetch_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {fetch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "fetch", None)
    if fn is None:
        raise RuntimeError(f"{fetch_path} has no `fetch(context)` function")

    import asyncio, inspect
    if inspect.iscoroutinefunction(fn):
        result = await fn(context)
    else:
        result = await asyncio.to_thread(fn, context)
    return str(result) if result is not None else ""
