"""Profile extractor (Phase G.2.a).

Runs after a health-classified turn lands. Reads the user message +
assistant response, asks Haiku to identify candidate profile updates,
returns them as `PendingUpdate` proposals. The proposals land in
`profile.pending_updates` for the user to accept / edit / reject via
the confirmation surface (Phase G.2.b).

What the extractor sees:
- The most recent turn (user message + assistant response)
- The profile's *structure* — section names + counts, NOT values

What it deliberately doesn't see (per spec §G.5):
- Existing field values. Prevents the extractor from "helpfully"
  re-surfacing facts the user might not want re-surfaced. The job
  is "what NEW information appeared in this turn," not "what does
  this turn imply about the existing profile."

Confidence floor: 0.7 (spec's "below 0.4 do not propose" plus a
margin so we don't pollute the queue with low-confidence noise).

Skip conditions:
- Active writing project: fictional characters' health facts must
  not enter the user's real profile (spec §G.4 "novel about a
  character with diabetes" carveout).
- Non-health turn: caller should only invoke this when triage
  flagged the turn as health-related (phi_class in general/personal/
  unknown). The extractor doesn't re-validate; it trusts the caller.

The extractor never modifies the profile directly. It returns
proposals; persistence is a separate `persist_proposals()` step that
loads, appends, saves with optimistic-concurrency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import apsw

from src.health.profile.models import HealthProfile, PendingUpdate
from src.health.profile.store import KeySource, load_profile, save_profile

log = logging.getLogger(__name__)


MIN_CONFIDENCE = 0.7

# Pending updates auto-expire after 24h if the user doesn't respond
# (spec §G.4 "silence is not consent"). Stored in the proposal's
# id-derivation for stable IDs; expiry enforced at confirm time.
DEFAULT_PENDING_TTL_HOURS = 24


# Section names the extractor is allowed to propose updates against.
# Matches HealthProfile attribute names that hold user data.
EXTRACTABLE_SECTIONS = (
    "demographics",
    "conditions",
    "medications",
    "allergies",
    "family_history",
    "lifestyle",
    "labs",
    "concerns",
)


EXTRACTOR_SCHEMA_HINT = """{
  "proposals": [
    {
      "section": "medications" | "conditions" | "allergies" |
                 "family_history" | "lifestyle" | "labs" |
                 "concerns" | "demographics",
      "operation": "add" | "modify" | "remove",
      "proposed_value": { ...fields appropriate for the section... },
      "confidence": <float 0.0-1.0>,
      "reasoning": "<one sentence the user will see, plain language>"
    }
  ]
}"""


EXTRACTOR_SYSTEM_PROMPT = """\
You read one turn of conversation between a user and a health-aware AI
assistant. Your job is to identify any new, factual health information
the user shared, and propose structured updates to their profile.

You can see the profile's STRUCTURE (section names + counts) but NOT
its current values. Your job is to identify what's NEW in this turn,
not to reason about what's already there.

## What to extract

Only facts the user stated directly about themselves:
- "I take metformin" → propose adding metformin to medications
- "My doctor said my A1C is 6.2" → propose adding to labs
- "I've been having headaches for two weeks" → propose adding to concerns
- "I quit smoking last year" → propose updating lifestyle.smoking to "former"
- "My mom has type 2 diabetes" → propose adding to family_history

## What NOT to extract

- Hypotheticals: "If I had diabetes, what would I do?"
- Questions about others: "My friend takes Lipitor, is that safe?"
- Past mentions you cannot verify as current: "I used to take X..."
  (only extract as discontinued status, not as active)
- Anything in fictional contexts (writing projects)
- Anything the user phrased uncertainly: "I might be allergic to X"
  (extract with low confidence, or skip)
- Anything the user only IMPLIED but didn't state

## Confidence calibration

- 1.0: User stated the fact directly and unambiguously
- 0.7-0.9: User stated it but with minor ambiguity (no dose, no date)
- 0.4-0.6: User implied it strongly but did not state it
- Below 0.4: Do NOT propose

## Output

Return ONLY valid JSON matching this schema. If nothing should be
extracted, return {"proposals": []}. Never return free-form text.
""" + EXTRACTOR_SCHEMA_HINT


# (system, user) -> response_text, matching SingleTurnClient.query
LLMCall = Callable[[str, str], Awaitable[str]]


async def run_extractor(
    *,
    user_message: str,
    assistant_response: str,
    profile_structure: dict[str, int],
    llm_call: LLMCall,
    active_project: Optional[str] = None,
    source_turn_id: str = "",
    now: Optional[datetime] = None,
) -> list[PendingUpdate]:
    """Run the extractor on one turn. Returns candidate proposals.

    `profile_structure` is `{section_name: count}` — the extractor sees
    counts only, never values. Build with `summarize_structure(profile)`.

    `active_project` is the user's currently-active writing project, if
    any. When set, this is a no-op — fictional content does not
    populate the user's real profile.

    Failures are swallowed (logged) and return []. The extractor must
    never crash the host pipeline.
    """
    if active_project:
        log.debug(
            "Extractor skipped — active project %r (fictional context)",
            active_project,
        )
        return []

    if not user_message.strip() and not assistant_response.strip():
        return []

    now_dt = now if now is not None else datetime.now(timezone.utc).replace(tzinfo=None)
    user_payload = _format_user_payload(
        user_message=user_message,
        assistant_response=assistant_response,
        profile_structure=profile_structure,
    )

    try:
        raw = await llm_call(EXTRACTOR_SYSTEM_PROMPT, user_payload)
    except Exception as e:
        log.warning("Extractor LLM call failed: %s", e)
        return []

    data = _parse_extractor_response(raw)
    if data is None:
        log.warning("Extractor: malformed JSON; dropping")
        return []

    proposals: list[PendingUpdate] = []
    for raw_proposal in data.get("proposals", []):
        proposal = _validate_and_build_proposal(
            raw_proposal,
            now=now_dt,
            source_turn_id=source_turn_id,
        )
        if proposal is not None:
            proposals.append(proposal)

    log.info("Extractor: %d proposal(s) generated", len(proposals))
    return proposals


def summarize_structure(profile: HealthProfile) -> dict[str, int]:
    """Build `{section_name: count}` for the extractor's input.

    Used by callers right before invoking the extractor. The dict is
    intentionally tiny — keeps the extractor's view minimal.
    """
    return {
        "demographics": _count_demographics(profile),
        "conditions": len(profile.conditions),
        "medications": len(profile.medications),
        "allergies": len(profile.allergies),
        "family_history": len(profile.family_history),
        "lifestyle": _count_lifestyle(profile),
        "labs": len(profile.labs),
        "concerns": len(profile.concerns),
    }


# --- Persistence ----------------------------------------------------------


def persist_proposals(
    *,
    conn: apsw.Connection,
    proposals: list[PendingUpdate],
    user_id: str = "self",
    key_source: KeySource = "keychain",
    max_retries: int = 3,
) -> int:
    """Load the profile, append proposals to pending_updates, save.

    Idempotent: proposals carry stable IDs (content-hashed), so
    re-running with the same input doesn't duplicate. Returns the
    number of NEW proposals persisted (existing ones are skipped).

    Optimistic concurrency: on StaleProfileError, reloads and retries
    up to `max_retries`. The extractor runs after every health turn;
    a concurrent CLI edit could trigger a race that the retry handles
    transparently.
    """
    from src.health.profile.store import StaleProfileError

    if not proposals:
        return 0

    for attempt in range(max_retries):
        loaded = load_profile(conn, user_id, key_source=key_source)
        existing_ids = {p.id for p in loaded.profile.pending_updates}

        new_proposals = [p for p in proposals if p.id not in existing_ids]
        if not new_proposals:
            log.debug("All proposals already in pending_updates; no-op")
            return 0

        loaded.profile.pending_updates.extend(new_proposals)
        try:
            save_profile(
                conn,
                loaded.profile,
                expected_version=loaded.version,
                key_source=key_source,
                operation="propose",
                section="pending_updates",
            )
            return len(new_proposals)
        except StaleProfileError:
            if attempt == max_retries - 1:
                raise
            log.info(
                "persist_proposals: stale version, retry %d/%d",
                attempt + 2, max_retries,
            )
            continue
    return 0  # unreachable; loop either returns or raises


# --- Helpers --------------------------------------------------------------


def _format_user_payload(
    *,
    user_message: str,
    assistant_response: str,
    profile_structure: dict[str, int],
) -> str:
    structure_lines = [
        f"  {name}: {count}" for name, count in profile_structure.items()
    ]
    return (
        "[Profile structure — counts only, never values]\n"
        + "\n".join(structure_lines)
        + "\n\n"
        + "[User message]\n"
        + user_message.strip()
        + "\n\n"
        + "[Assistant response]\n"
        + assistant_response.strip()
        + "\n\n"
        + "Identify NEW facts the user stated about themselves. "
        + "Return JSON per the schema."
    )


def _parse_extractor_response(raw: str) -> dict | None:
    from src.pipeline.json_utils import extract_json
    return extract_json(raw)


def _validate_and_build_proposal(
    item: dict, *, now: datetime, source_turn_id: str,
) -> PendingUpdate | None:
    if not isinstance(item, dict):
        return None

    section = item.get("section")
    if section not in EXTRACTABLE_SECTIONS:
        log.debug("Extractor: dropping proposal with bad section %r", section)
        return None

    operation = item.get("operation")
    if operation not in ("add", "modify", "remove"):
        log.debug("Extractor: dropping proposal with bad operation %r", operation)
        return None

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        log.debug("Extractor: dropping proposal with non-numeric confidence")
        return None
    if confidence < MIN_CONFIDENCE:
        log.debug(
            "Extractor: dropping low-confidence proposal (%.2f < %.2f)",
            confidence, MIN_CONFIDENCE,
        )
        return None

    proposed_value = item.get("proposed_value") or {}
    if not isinstance(proposed_value, dict):
        return None
    if not proposed_value:
        log.debug("Extractor: dropping proposal with empty proposed_value")
        return None

    reasoning = (item.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = f"{operation} on {section}"

    return PendingUpdate(
        id=_proposal_id(section, operation, proposed_value),
        proposed_at=now,
        target_section=section,
        operation=operation,
        proposed_value=proposed_value,
        source_turn_id=source_turn_id,
        extractor_reasoning=reasoning,
        status="pending",
    )


def _proposal_id(section: str, operation: str, proposed_value: dict) -> str:
    """Stable id derived from (section, operation, sorted-value-keys-and-values).

    Same input → same id, so re-extracting from the same turn doesn't
    duplicate proposals in pending_updates. We sort the JSON dump so
    key-order variation doesn't change the id.
    """
    canonical = json.dumps(
        {"section": section, "operation": operation, "value": proposed_value},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"prop_{digest}"


def _count_demographics(profile: HealthProfile) -> int:
    d = profile.demographics
    return sum(
        1 for f in [
            d.age_range, d.sex_assigned_at_birth, d.gender_identity,
            d.height_range, d.weight_range, d.pregnancy_status,
        ] if f is not None
    )


def _count_lifestyle(profile: HealthProfile) -> int:
    lf = profile.lifestyle
    return sum(
        1 for f in [
            lf.smoking, lf.alcohol, lf.exercise_frequency,
            lf.sleep_hours_typical, lf.diet_pattern,
        ] if f is not None
    )
