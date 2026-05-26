"""LLM-based judge for heartbeat hooks.

The judge takes the current snapshot + previous snapshot and decides
whether the user should be notified. Uses Haiku via the selector
under stage `heartbeat_judge` so the env override
`LLM_STAGE_HEARTBEAT_JUDGE=near:anthropic/claude-haiku-4-5` works
the same as other stages.

Be conservative: most ticks should NOT notify. The judge's
instructions explicitly say "no notify unless something the user
would care about RIGHT NOW changed since the previous tick."

Hooks that prefer pure code-based logic (threshold checks) can pass
their own `judge` function on the HookSpec.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable, Optional

from src.heartbeat.runner import HeartbeatEvent

log = logging.getLogger(__name__)


# (system, user) → response_text — same shape as src.llm.selector.get_stage_callable
LLMCall = Callable[[str, str], Awaitable[str]]


JUDGE_SYSTEM_PROMPT = """\
You are a heartbeat judge for a personal AI assistant. The assistant
runs lightweight monitoring tasks on a fixed cadence (every 5–15
minutes) and asks you whether the current snapshot is worth
notifying the user about.

Be CONSERVATIVE. Most ticks should NOT notify — the default answer
is no. Only flag when one of:

- A status changed meaningfully since the previous snapshot
  (e.g., camera went from enabled to disabled, network went from
  armed to disarmed unexpectedly, sync module dropped offline)
- A threshold was crossed
  (battery dropped to "low", wifi signal dropped to 1/5)
- Something needs the user's attention RIGHT NOW
  (offline device that should be online, recent motion event)

Do NOT notify for:
- Routine status that hasn't changed
- Minor variation (wifi 4/5 → 3/5)
- "Everything's fine" — silence IS the all-clear

Output ONLY valid JSON, no other text:

{
  "notify": <true | false>,
  "summary": "<one short sentence the user will read, plain language;
              include WHICH device / WHAT changed; max 120 chars>",
  "severity": "info" | "warn" | "critical"
}

If notify=false, summary can be empty.
"""


def _truncate(s: str, n: int) -> str:
    """Trim long snapshot dumps so the judge prompt stays cheap."""
    return s if len(s) <= n else s[:n] + " ...[truncated]"


async def llm_judge(
    hook_name: str,
    current: dict,
    previous: Optional[dict],
    *,
    llm_call: LLMCall,
) -> HeartbeatEvent:
    """The default judge — single Haiku call comparing current vs previous.

    Returns a HeartbeatEvent. On any failure (LLM down, malformed
    JSON), returns `should_notify=False` — the heartbeat is best-effort;
    a silent tick beats a false alarm.
    """
    user_msg = (
        f"[Hook: {hook_name}]\n\n"
        "[Current snapshot]\n"
        f"{_truncate(json.dumps(current, default=str, indent=2), 4000)}\n\n"
        "[Previous snapshot]\n"
        f"{_truncate(json.dumps(previous, default=str, indent=2), 4000) if previous else '(none — this is the first tick after a fresh install or restart)'}\n\n"
        "Should the user be notified? Return JSON per the schema."
    )

    raw = ""
    try:
        raw = await llm_call(JUDGE_SYSTEM_PROMPT, user_msg)
    except Exception as e:
        log.warning("[heartbeat-judge] LLM call failed: %s", e)
        return _silent(hook_name, current)

    data = _extract_json(raw)
    if data is None:
        log.warning(
            "[heartbeat-judge] malformed JSON from %s judge: %r",
            hook_name, raw[:200],
        )
        return _silent(hook_name, current)

    severity = data.get("severity") or "info"
    if severity not in ("info", "warn", "critical"):
        severity = "info"

    return HeartbeatEvent(
        hook_name=hook_name,
        fired_at=int(time.time()),
        raw=current,
        summary=(data.get("summary") or "").strip(),
        should_notify=bool(data.get("notify")),
        severity=severity,
    )


def _silent(hook_name: str, current: dict) -> HeartbeatEvent:
    return HeartbeatEvent(
        hook_name=hook_name,
        fired_at=int(time.time()),
        raw=current,
        summary="",
        should_notify=False,
    )


def _extract_json(raw: str) -> Optional[dict]:
    """Tolerant JSON extract — same shape as src.pipeline.json_utils."""
    from src.pipeline.json_utils import extract_json
    return extract_json(raw)


# --- Convenience factory ---


def make_default_judge(*, config) -> Callable:
    """Build the default judge bound to the configured Haiku stage.

    Returned function matches the JudgeFn signature. The selector resolves
    `heartbeat_judge` via env override; falls back to triage's default
    (Claude Haiku 4.5 in current setups).
    """
    from src.llm.selector import get_stage_callable

    async def judge(hook_name: str, current: dict, previous: Optional[dict]):
        llm_call = get_stage_callable(
            "heartbeat_judge",
            fallback_model=config.model_triage,
            auth_mode=config.auth_mode,
            api_key=config.anthropic_api_key,
            cli_path=config.cli_path,
            workspace_dir=str(config.workspace_dir),
        )
        return await llm_judge(
            hook_name, current, previous, llm_call=llm_call,
        )

    return judge
