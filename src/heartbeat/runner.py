"""Heartbeat runner — registry of lightweight tasks on sub-day cadence.

The existing scheduler (src/scheduler) handles cron-style jobs that
fire from once-a-day to once-every-few-minutes via separate processes
or LLM calls. Heartbeat is a complementary, lighter mechanism for
in-process Python callables that share a single async loop and can
report state changes between ticks.

Design:

- Each `HookSpec` declares a name, an interval_minutes, and an async
  runner function that returns a dict snapshot of "current state."
- The runner has a 60-second tick loop. Each tick, hooks whose
  interval has elapsed since their last run are fired in parallel via
  asyncio.gather (failures isolated per hook).
- After a hook runs, its raw snapshot goes to a Judge — by default a
  Haiku-based LLM judge that compares current vs previous snapshot
  and returns a `HeartbeatEvent` with `should_notify` + `summary` +
  `severity`. Hooks can override with their own judge (e.g. pure
  threshold checks).
- When `should_notify` is True, the runner calls the registered
  notify function (typically: send to Signal / Telegram via the
  scheduler's ChannelSender abstraction).
- Snapshots persist to JSON files under workspace/heartbeat/<hook>.json
  so the next tick has a "previous" to compare against, even after
  process restart.

What heartbeat is NOT:
- Not a scheduler replacement. Cron-style "fire LLM job at 9am" still
  belongs in src/scheduler. Heartbeat is for "every 15 min check X
  and tell me if something looks off."
- Not a backfill mechanism. If the bot was offline for 12 hours, the
  heartbeat doesn't fire 48 ticks on resume — it just runs the next
  one on schedule.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)


# Sentinel for "this tick failed transiently — skip and try next time"
# without filling the log with traceback noise. Use for known-fragile
# upstream calls (rate limits, temporary network drops).
class HeartbeatSkip(Exception):
    pass


@dataclass(frozen=True)
class HeartbeatEvent:
    """What a hook's judge produces. The runner uses `should_notify`
    to decide whether to call the notify function."""

    hook_name: str
    fired_at: int                # epoch seconds
    raw: dict
    summary: str                 # one-sentence user-facing text
    should_notify: bool
    severity: str = "info"       # "info" | "warn" | "critical"


# A hook's runner function: async () → raw dict snapshot
HookRunner = Callable[[], Awaitable[dict]]

# A judge function: async (hook_name, current_raw, previous_raw_or_None)
#                   → HeartbeatEvent
JudgeFn = Callable[[str, dict, Optional[dict]], Awaitable[HeartbeatEvent]]

# A notify function: async (event) → None
NotifyFn = Callable[[HeartbeatEvent], Awaitable[None]]


@dataclass
class HookSpec:
    name: str
    interval_minutes: int
    runner: HookRunner
    judge: Optional[JudgeFn] = None     # falls back to runner's default
    # Optional human-readable description for `microwaveos heartbeat list`.
    description: str = ""


class HeartbeatRunner:
    """Manages a registry of lightweight tasks that fire at sub-day intervals."""

    # The loop wakes once a minute. Hooks with interval_minutes=1 fire
    # on every wake; hooks with interval_minutes=15 fire roughly every
    # 15th wake. We don't honor sub-minute intervals — if you need
    # those, use the cron scheduler with a continuous process.
    TICK_INTERVAL_SECONDS = 60

    def __init__(
        self,
        *,
        state_dir: Path,
        judge: JudgeFn,
        notify: NotifyFn,
    ):
        self.state_dir = Path(state_dir)
        self._default_judge = judge
        self._notify = notify

        self._hooks: dict[str, HookSpec] = {}
        # epoch seconds of last fire per hook; 0 = never fired
        self._last_run: dict[str, int] = {}

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # --- Registry -------------------------------------------------------

    def register(self, spec: HookSpec) -> None:
        if spec.name in self._hooks:
            raise ValueError(f"Hook already registered: {spec.name!r}")
        if spec.interval_minutes < 1:
            raise ValueError(
                f"interval_minutes must be >= 1, got {spec.interval_minutes}"
            )
        self._hooks[spec.name] = spec
        log.info(
            "[heartbeat] registered hook %s (every %d min)",
            spec.name, spec.interval_minutes,
        )

    @property
    def hooks(self) -> list[HookSpec]:
        return list(self._hooks.values())

    # --- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return  # already running
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="heartbeat-loop")
        log.info(
            "[heartbeat] runner started with %d hook(s)", len(self._hooks),
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except asyncio.TimeoutError:
            log.warning("[heartbeat] runner did not stop in 5s; cancelling")
            self._task.cancel()
        except Exception as e:
            log.warning("[heartbeat] runner exited with error: %s", e)
        self._task = None

    # --- Internals ------------------------------------------------------

    async def _loop(self) -> None:
        """Wake every TICK_INTERVAL_SECONDS, run due hooks, sleep."""
        # Fire once immediately on startup so a fresh process doesn't
        # wait a full minute before the first tick. Useful for testing.
        try:
            await self._tick()
        except Exception as e:
            log.warning("[heartbeat] initial tick error: %s", e)

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.TICK_INTERVAL_SECONDS,
                )
                break  # stop_event set during wait
            except asyncio.TimeoutError:
                pass
            try:
                await self._tick()
            except Exception as e:
                log.warning("[heartbeat] tick error: %s", e)

    async def _tick(self) -> None:
        """Fire any hook whose interval has elapsed."""
        now = int(time.time())
        due = [
            spec for spec in self._hooks.values()
            if now - self._last_run.get(spec.name, 0) >= spec.interval_minutes * 60
        ]
        if not due:
            return
        await asyncio.gather(
            *(self._fire(spec, now) for spec in due),
            return_exceptions=True,
        )

    async def _fire(self, spec: HookSpec, now: int) -> None:
        """Run one hook end-to-end: runner → judge → optional notify."""
        try:
            raw = await spec.runner()
        except HeartbeatSkip as e:
            log.info("[heartbeat] %s skipped: %s", spec.name, e)
            self._last_run[spec.name] = now
            return
        except Exception as e:
            log.warning("[heartbeat] %s runner failed: %s", spec.name, e)
            self._last_run[spec.name] = now  # avoid hot-loop retry on failure
            return

        # Lazy import so the runner module is testable in isolation.
        from src.heartbeat.state import load_hook_state, save_hook_state
        previous = load_hook_state(self.state_dir, spec.name)

        judge_fn = spec.judge or self._default_judge
        try:
            event = await judge_fn(spec.name, raw, previous)
        except Exception as e:
            log.warning("[heartbeat] %s judge failed: %s", spec.name, e)
            self._last_run[spec.name] = now
            return

        save_hook_state(self.state_dir, spec.name, raw)
        self._last_run[spec.name] = now

        if event.should_notify:
            log.info(
                "[heartbeat] %s → notify (severity=%s): %s",
                spec.name, event.severity, event.summary,
            )
            try:
                await self._notify(event)
            except Exception as e:
                log.warning("[heartbeat] %s notify failed: %s", spec.name, e)
        else:
            log.debug(
                "[heartbeat] %s → no notify (judge said quiet)", spec.name,
            )
