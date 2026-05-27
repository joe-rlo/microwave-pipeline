"""LLM-facing tools for managing the bot's own cron schedule.

Five tools, mirroring the `microwaveos scheduler` CLI:

- `scheduler_list`     — names + cron + mode + last run (one-line view)
- `scheduler_get`      — full details on one job (prompt body, skill, etc.)
- `scheduler_add`      — create a job (llm | direct | script)
- `scheduler_remove`   — delete by name
- `scheduler_set_enabled` — enable or disable by name (no row delete)

Why expose these. Before this module existed, the bot deflected
schedule questions with "I don't know where the cron lives" because
the table is in SQLite, not workspace/. Same scheduler the user
already runs from CLI — these tools just give the LLM a typed
interface to the same `SchedulerStore` API.

Channel + recipient defaults. When the bot is adding a job at the
user's request, it usually wants to send to "the same place we're
talking now." But the tool handler doesn't see chat context. So we
fall back to the operator's `HEARTBEAT_NOTIFY_*` config (the existing
"primary user" knobs) when the LLM omits these fields. The LLM should
still pass them explicitly when it knows them; the fallback is the
safety net, not the default style.

What's deliberately NOT here: on-demand fire. Firing a job out of
schedule needs a live channel handle (the Signal/Telegram sender
already in use) and a real Scheduler instance, neither of which a
stateless tool handler has. For instant fires the user still uses
`microwaveos scheduler run <name>` from the CLI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from croniter import croniter

from src.scheduler.store import ScheduledJob, SchedulerStore

log = logging.getLogger(__name__)


# What the LLM sees in its system context. Goes in the dynamic
# catalog block (not the stable prompt) like every other tool's docs.
SCHEDULER_TOOL_DOCS = """\
**scheduler_list** — List all scheduled jobs (name, cron, mode, channel, enabled, last-run).

When to use:
- "What's on my schedule?", "show my crons", "what's scheduled?".
- As the first step before adding a similar job — read an existing
  one's shape with `scheduler_get` and mirror it.

**scheduler_get** — Get full details on one job, including its prompt body, skill, recipient, and timezone.

When to use:
- Before `scheduler_add` when the user says "in the same format as the morning briefing".
- Debugging a job that didn't fire or sent wrong content.

How to use:
- `name`: the job's name (case-sensitive).

**scheduler_add** — Create a new scheduled job.

When to use:
- The user asks you to set up a recurring message, briefing, reminder, or report.

How to use:
- `name`: unique short identifier (kebab-case), e.g. "fitness-morning".
- `cron`: standard 5-field cron expression in the job's timezone, e.g. "0 7 * * *".
- `mode`: one of "llm" | "direct" | "script".
  - llm: prompt becomes the LLM ask; output is delivered.
  - direct: literal text is sent verbatim, no LLM call.
  - script: shell command runs; stdout is delivered (use for metrics, log digests, anything needing real numbers).
- For mode=llm: pass either `prompt` (free-form ask) or `skill` (existing skill name) + `trigger` (short kickoff message).
- For mode=direct: pass `text`.
- For mode=script: pass `command`.
- `channel` + `recipient` may be omitted to default to the operator's primary recipient (HEARTBEAT_NOTIFY_*).
- `timezone` defaults to America/New_York; `card_view` defaults to true for llm/script modes.

**scheduler_remove** — Delete a job permanently.

How to use:
- `name`: the job to delete.

**scheduler_set_enabled** — Enable or disable a job without deleting it.

When to use:
- Pausing a job temporarily (vacation, broken script) without losing its config.

How to use:
- `name`: the job.
- `enabled`: true to enable, false to disable.

NOTE: There is no on-demand fire tool. For instant fires the user runs
`microwaveos scheduler run <name>` from the CLI.
"""


# --- JSON schemas ----------------------------------------------------------

_VALID_MODES = ("llm", "direct", "script")
_VALID_CHANNELS = ("signal", "telegram")


SCHEDULER_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

SCHEDULER_GET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Job name"},
    },
    "required": ["name"],
    "additionalProperties": False,
}

SCHEDULER_ADD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Unique kebab-case job name"},
        "cron": {
            "type": "string",
            "description": 'Standard 5-field cron, e.g. "0 7 * * *"',
        },
        "mode": {
            "type": "string",
            "enum": list(_VALID_MODES),
            "description": "llm | direct | script",
        },
        "prompt": {
            "type": "string",
            "description": "LLM prompt body (mode=llm without --skill)",
        },
        "skill": {
            "type": "string",
            "description": "Existing skill name (mode=llm, alternative to prompt)",
        },
        "trigger": {
            "type": "string",
            "description": "Short user-message kickoff when skill is set",
        },
        "text": {
            "type": "string",
            "description": "Literal text to send (mode=direct)",
        },
        "command": {
            "type": "string",
            "description": "Shell command to run (mode=script)",
        },
        "channel": {
            "type": "string",
            "enum": list(_VALID_CHANNELS),
            "description": "Target channel; defaults to operator's primary",
        },
        "recipient": {
            "type": "string",
            "description": "Phone (Signal) or chat id (Telegram); defaults to operator's primary",
        },
        "timezone": {
            "type": "string",
            "description": "IANA tz; default America/New_York",
        },
        "card_view": {
            "type": "boolean",
            "description": "llm/script only; default true",
        },
        "card_split": {
            "type": "string",
            "description": 'Card separator; default "---"',
        },
    },
    "required": ["name", "cron", "mode"],
    "additionalProperties": False,
}

SCHEDULER_REMOVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}

SCHEDULER_SET_ENABLED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "enabled": {"type": "boolean"},
    },
    "required": ["name", "enabled"],
    "additionalProperties": False,
}


# --- Handlers --------------------------------------------------------------
#
# Each handler takes `args` (the LLM's tool input) and `config` (for DB
# path + defaults). Returns a string the LLM reads as the tool result.
# Raises RuntimeError on invalid input — the session loop converts that
# into is_error=True so the model can self-correct.


def _open_store(config) -> SchedulerStore:
    store = SchedulerStore(config.db_path)
    store.connect()
    return store


def _job_to_brief(j: ScheduledJob) -> dict[str, Any]:
    """One-line summary shape — what the LLM sees from `scheduler_list`."""
    return {
        "name": j.name,
        "cron": j.cron_expr,
        "mode": j.mode,
        "channel": j.target_channel,
        "enabled": j.enabled,
        "last_run_at": j.last_run_at.isoformat() if j.last_run_at else None,
        "last_error": j.last_error,
    }


def _job_to_full(j: ScheduledJob) -> dict[str, Any]:
    """Full detail shape — what `scheduler_get` returns."""
    return {
        "id": j.id,
        "name": j.name,
        "cron": j.cron_expr,
        "mode": j.mode,
        "prompt_or_text": j.prompt_or_text,
        "skill_name": j.skill_name or None,
        "channel": j.target_channel,
        "recipient": j.recipient_id,
        "enabled": j.enabled,
        "timezone": j.timezone,
        "card_view": j.card_view,
        "card_split": j.card_split,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "last_run_at": j.last_run_at.isoformat() if j.last_run_at else None,
        "last_error": j.last_error,
    }


async def _handle_list(args: dict[str, Any], *, config) -> str:
    store = _open_store(config)
    try:
        jobs = store.list_all()
    finally:
        store.close()
    return json.dumps(
        {"jobs": [_job_to_brief(j) for j in jobs], "count": len(jobs)},
        indent=2,
    )


async def _handle_get(args: dict[str, Any], *, config) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        raise RuntimeError("scheduler_get requires `name`")
    store = _open_store(config)
    try:
        job = store.get_by_name(name)
    finally:
        store.close()
    if job is None:
        raise RuntimeError(f"No job named {name!r}")
    return json.dumps(_job_to_full(job), indent=2)


async def _handle_add(args: dict[str, Any], *, config) -> str:
    name = (args.get("name") or "").strip()
    cron_expr = (args.get("cron") or "").strip()
    mode = (args.get("mode") or "").strip()
    if not name or not cron_expr or not mode:
        raise RuntimeError("scheduler_add requires `name`, `cron`, and `mode`")
    if mode not in _VALID_MODES:
        raise RuntimeError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    try:
        croniter(cron_expr)
    except Exception as e:
        raise RuntimeError(f"Invalid cron expression {cron_expr!r}: {e}") from e

    # Resolve prompt body by mode. Mirrors the CLI's _cmd_add exactly so
    # behavior is identical whether the user goes via Signal or shell.
    skill_name = ""
    prompt_or_text = ""
    if mode == "direct":
        prompt_or_text = (args.get("text") or "").strip()
        if not prompt_or_text:
            raise RuntimeError("mode=direct requires `text`")
    elif mode == "script":
        prompt_or_text = (args.get("command") or "").strip()
        if not prompt_or_text:
            raise RuntimeError("mode=script requires `command`")
    else:  # llm
        skill = (args.get("skill") or "").strip()
        prompt = (args.get("prompt") or "").strip()
        trigger = (args.get("trigger") or "").strip()
        if skill:
            if not trigger:
                raise RuntimeError(
                    "mode=llm with `skill` also requires `trigger` "
                    "(the short kickoff user message)"
                )
            # Verify the skill exists; fail loudly at add time instead of
            # at the first fire.
            from src.skills import SkillLoader, SkillNotFound

            workspace_dir = getattr(config, "workspace_dir", None)
            if workspace_dir is None:
                raise RuntimeError(
                    "Cannot validate skill: workspace_dir not configured"
                )
            loader = SkillLoader(workspace_dir / "skills")
            try:
                loader.load(skill)
            except SkillNotFound as e:
                raise RuntimeError(
                    f"No skill named {skill!r} at {loader.skills_dir}"
                ) from e
            skill_name = skill
            prompt_or_text = trigger
        elif prompt:
            prompt_or_text = prompt
        else:
            raise RuntimeError("mode=llm requires `prompt` or `skill`+`trigger`")

    # Channel + recipient: explicit > config default. Fail loudly only if
    # both are missing in args AND config — better than silently picking
    # the wrong target.
    channel = (args.get("channel") or "").strip()
    if not channel:
        channel = getattr(config, "heartbeat_notify_channel", "") or ""
    if channel not in _VALID_CHANNELS:
        raise RuntimeError(
            f"channel must be one of {_VALID_CHANNELS}; got {channel!r}. "
            "Pass `channel` explicitly or set HEARTBEAT_NOTIFY_CHANNEL."
        )

    recipient = (args.get("recipient") or "").strip()
    if not recipient:
        recipient = getattr(config, "heartbeat_notify_recipient", "") or ""
    if not recipient:
        raise RuntimeError(
            "recipient missing — pass `recipient` explicitly or set "
            "HEARTBEAT_NOTIFY_RECIPIENT"
        )

    tz = (args.get("timezone") or "").strip() or "America/New_York"
    card_view = bool(args.get("card_view", True))
    card_split = (args.get("card_split") or "").strip() or "---"

    store = _open_store(config)
    try:
        if store.get_by_name(name) is not None:
            raise RuntimeError(
                f"Job {name!r} already exists. Remove it first with scheduler_remove."
            )
        job = ScheduledJob(
            name=name,
            cron_expr=cron_expr,
            mode=mode,
            prompt_or_text=prompt_or_text,
            target_channel=channel,
            recipient_id=recipient,
            enabled=True,
            timezone=tz,
            card_split=card_split,
            card_view=(mode in ("llm", "script") and card_view),
            skill_name=skill_name,
        )
        job_id = store.add(job)
        job.id = job_id
        return json.dumps(
            {"status": "added", "job": _job_to_full(job)},
            indent=2,
        )
    finally:
        store.close()


async def _handle_remove(args: dict[str, Any], *, config) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        raise RuntimeError("scheduler_remove requires `name`")
    store = _open_store(config)
    try:
        ok = store.remove(name)
    finally:
        store.close()
    if not ok:
        raise RuntimeError(f"No job named {name!r}")
    return json.dumps({"status": "removed", "name": name})


async def _handle_set_enabled(args: dict[str, Any], *, config) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        raise RuntimeError("scheduler_set_enabled requires `name`")
    if "enabled" not in args:
        raise RuntimeError("scheduler_set_enabled requires `enabled` (boolean)")
    enabled = bool(args["enabled"])
    store = _open_store(config)
    try:
        ok = store.set_enabled(name, enabled)
    finally:
        store.close()
    if not ok:
        raise RuntimeError(f"No job named {name!r}")
    return json.dumps(
        {"status": "enabled" if enabled else "disabled", "name": name}
    )


