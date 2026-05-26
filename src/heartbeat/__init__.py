"""Heartbeat — lightweight, sub-day-cadence task runner.

See src/heartbeat/runner.py for design notes.

Public surface:
    HeartbeatRunner, HookSpec, HeartbeatEvent, HeartbeatSkip
    llm_judge, make_default_judge
    state_path, load_hook_state, save_hook_state
    fetch_blink_status, build_blink_hook_spec
    setup_runner — convenience factory the bot uses at startup
"""

from src.heartbeat.judge import llm_judge, make_default_judge
from src.heartbeat.runner import (
    HeartbeatEvent,
    HeartbeatRunner,
    HeartbeatSkip,
    HookSpec,
)
from src.heartbeat.state import (
    load_hook_state,
    save_hook_state,
    state_path,
)
from src.heartbeat.hooks.blink import build_blink_hook_spec, fetch_blink_status


def setup_runner(config, channels: dict) -> "HeartbeatRunner | None":
    """Build a configured HeartbeatRunner for the bot to start.

    Returns None when heartbeat is disabled — caller should not start.
    Otherwise returns a runner with the default judge wired (Haiku via
    selector), the notify function pointed at the configured channel
    (Signal/Telegram via `channels`), and any enabled hooks already
    registered.
    """
    if not getattr(config, "heartbeat_enabled", False):
        return None

    state_dir = config.workspace_dir / "heartbeat"
    judge = make_default_judge(config=config)
    notify = _build_notify(config, channels)

    runner = HeartbeatRunner(state_dir=state_dir, judge=judge, notify=notify)

    # --- Register configured hooks ---
    if getattr(config, "blink_heartbeat_enabled", False):
        try:
            runner.register(build_blink_hook_spec(
                interval_minutes=int(
                    getattr(config, "blink_heartbeat_interval_minutes", 15)
                ),
            ))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "[heartbeat] failed to register blink hook: %s", e,
            )

    return runner


def _build_notify(config, channels: dict):
    """Notify factory — closes over the channel registry.

    The heartbeat notify function takes a HeartbeatEvent and sends a
    short text to the configured Signal/Telegram recipient. Failures
    are logged but never raised — a notification glitch must not break
    the runner loop.
    """
    import logging
    log = logging.getLogger("src.heartbeat")

    channel_name = (getattr(config, "heartbeat_notify_channel", "") or "").strip()
    recipient = (getattr(config, "heartbeat_notify_recipient", "") or "").strip()

    async def notify(event: HeartbeatEvent) -> None:
        if not channel_name or not recipient:
            log.info(
                "[heartbeat] notify (no channel configured) %s: %s",
                event.hook_name, event.summary,
            )
            return
        ch = channels.get(channel_name)
        if ch is None:
            log.warning(
                "[heartbeat] channel %r not registered; known=%s",
                channel_name, sorted(channels),
            )
            return
        icon = {"info": "ℹ️", "warn": "⚠️", "critical": "🚨"}.get(
            event.severity, "ℹ️",
        )
        text = f"{icon} [{event.hook_name}] {event.summary}"
        try:
            await ch.send_text(recipient, text)
        except Exception as e:
            log.warning("[heartbeat] send_text failed: %s", e)

    return notify


__all__ = [
    "HeartbeatEvent",
    "HeartbeatRunner",
    "HeartbeatSkip",
    "HookSpec",
    "build_blink_hook_spec",
    "fetch_blink_status",
    "llm_judge",
    "load_hook_state",
    "make_default_judge",
    "save_hook_state",
    "setup_runner",
    "state_path",
]
