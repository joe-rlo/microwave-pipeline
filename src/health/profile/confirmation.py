"""User confirmation surface for profile proposals (Phase G.2.b).

Two pure-function surfaces — `parse_user_reply` decodes a chat
message against the most-recently-shown proposal list, and
`apply_proposal` mutates a HealthProfile by promoting one proposal
into the appropriate section.

Channel-agnostic by design — works in REPL / Telegram / Signal alike
because the inputs and outputs are plain text (and the proposal
acceptance is shown via a `profile_update` chunk that channels render
however they want).

Reply intents recognized:

  ProposalReplyIntent.YES        accept ALL pending proposals
  ProposalReplyIntent.NO         reject ALL pending proposals
  ProposalReplyIntent.INDICES    accept the proposals at the named
                                 indices (1-based, as shown in the
                                 footer)
  ProposalReplyIntent.EDIT       not yet wired — surfaced for future
                                 interactive edits; today these get
                                 treated as NORMAL
  ProposalReplyIntent.NORMAL     not a reply — pass through to the
                                 main pipeline as a regular message

Auto-expire is the spec's "silence is not consent" mechanism: pending
proposals older than 24h get marked `auto_expired` so the queue
doesn't accumulate forever. Enforced lazily via `auto_expire_old`.

Section application coverage:

  add → list sections (conditions/medications/allergies/family_history/
        labs/concerns)
  modify (set field) → demographics + lifestyle
  modify (overlay onto existing entry) → list sections
        Find by identifier (name / substance / relation+condition /
        test_name / text); overlay proposed fields onto the existing
        entry; preserve field_meta.added_at and update last_modified.
        First-match wins when multiple identical identifiers exist.

  remove → not yet wired. Discontinuing a medication is naturally a
        `modify status="discontinued"` so this matters less than it
        looked at first. Lands when there's a real use case
        (mistakenly-added entries that need full deletion).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from src.health.profile.models import (
    Allergy,
    Concern,
    Condition,
    Demographics,
    FamilyHistoryEntry,
    HealthProfile,
    LabResult,
    LifestyleFactors,
    Medication,
    PendingUpdate,
    ProfileField,
)

log = logging.getLogger(__name__)


# Pending proposals older than this become auto_expired on next sweep.
# Spec §G.4: silence is not consent — 24h is the published TTL.
DEFAULT_PENDING_TTL_HOURS = 24


# Sections that hold a list of items (add appends an entry).
_LIST_SECTIONS = {
    "conditions": (Condition, "conditions"),
    "medications": (Medication, "medications"),
    "allergies": (Allergy, "allergies"),
    "family_history": (FamilyHistoryEntry, "family_history"),
    "labs": (LabResult, "labs"),
    "concerns": (Concern, "concerns"),
}

# Identifier fields per list section — used by modify to locate the
# existing entry that the proposal refers to. Tuple = compound match
# (all fields must equal); single value = match by that one field.
# Match is case-insensitive + whitespace-trimmed.
_LIST_SECTION_IDENTIFIERS: dict[str, tuple[str, ...]] = {
    "medications": ("name",),
    "conditions": ("name",),
    "allergies": ("substance",),
    "family_history": ("relation", "condition"),
    "labs": ("test_name",),     # most-recent if multiple
    "concerns": ("text",),      # exact match — fuzzy could land later
}

# Sections that hold a flat record (modify sets one field).
# Maps section name → set of valid field names.
_RECORD_SECTIONS = {
    "demographics": {
        "age_range", "sex_assigned_at_birth", "gender_identity",
        "height_range", "weight_range", "pregnancy_status",
    },
    "lifestyle": {
        "smoking", "alcohol", "exercise_frequency",
        "sleep_hours_typical", "diet_pattern",
    },
}


class ProposalReplyIntent(Enum):
    YES = "yes"
    NO = "no"
    INDICES = "indices"
    EDIT = "edit"          # reserved for future interactive flow
    NORMAL = "normal"      # pass-through


@dataclass(frozen=True)
class ProposalReply:
    intent: ProposalReplyIntent
    indices: tuple[int, ...] = ()    # 1-based, only set when intent=INDICES
    edit_text: Optional[str] = None  # only set when intent=EDIT


# --- Parser ---------------------------------------------------------------


# Affirmative tokens. Order matters for the "no" detection — "no" is
# checked before "yes" because "no thanks" should NOT be parsed as
# yes-with-typo.
_YES_TOKENS = ("yes", "yep", "yeah", "yup", "sure", "ok", "okay", "accept", "all", "confirm")
_NO_TOKENS = ("no", "nope", "skip", "reject", "decline", "dismiss")


# Match "1", "1 2", "1, 2", "1,2", etc. Trailing punctuation ok.
_INDEX_PATTERN = re.compile(r"^\s*[\d,\s]+[\.!]?\s*$")


def parse_user_reply(
    message: str, pending_count: int,
) -> ProposalReply:
    """Decode the user's message against the last-shown proposal list.

    `pending_count` is how many proposals were just offered. We only
    accept indices within `[1, pending_count]`.

    The parser is intentionally narrow — anything not clearly a reply
    falls through to `NORMAL` so the main pipeline runs unchanged.
    False-positive on "yes I have a question" is much worse than
    false-negative on "yeah that's right" (the user can re-send a
    cleaner reply; an accidental accept silently mutates state).
    """
    if pending_count <= 0:
        return ProposalReply(intent=ProposalReplyIntent.NORMAL)

    text = (message or "").strip().lower()
    if not text:
        return ProposalReply(intent=ProposalReplyIntent.NORMAL)

    # Trim trailing punctuation that doesn't change intent
    text_clean = text.rstrip(".!?,")

    # Single-token / two-token replies. Multi-sentence messages are
    # almost certainly normal turns, even if they start with "yes".
    word_count = len(text_clean.split())
    if word_count > 4:
        return ProposalReply(intent=ProposalReplyIntent.NORMAL)

    # Explicit "no" / "skip" wins over yes (per docstring).
    for tok in _NO_TOKENS:
        if text_clean == tok or text_clean.startswith(tok + " "):
            return ProposalReply(intent=ProposalReplyIntent.NO)

    # Index list — must match the digits-and-commas-only pattern.
    if _INDEX_PATTERN.match(text):
        try:
            raw = re.findall(r"\d+", text)
            indices = tuple(int(s) for s in raw)
        except ValueError:
            return ProposalReply(intent=ProposalReplyIntent.NORMAL)
        # Filter to valid range; drop dupes preserving order.
        seen: set[int] = set()
        clean: list[int] = []
        for i in indices:
            if 1 <= i <= pending_count and i not in seen:
                clean.append(i)
                seen.add(i)
        if not clean:
            return ProposalReply(intent=ProposalReplyIntent.NORMAL)
        return ProposalReply(
            intent=ProposalReplyIntent.INDICES,
            indices=tuple(clean),
        )

    # Affirmative — single short token only.
    for tok in _YES_TOKENS:
        if text_clean == tok:
            return ProposalReply(intent=ProposalReplyIntent.YES)
    # "yes all", "yes please", "accept all"
    if word_count == 2:
        head, tail = text_clean.split()
        if head in _YES_TOKENS and tail in ("all", "everything", "please", "them"):
            return ProposalReply(intent=ProposalReplyIntent.YES)

    return ProposalReply(intent=ProposalReplyIntent.NORMAL)


# --- Footer formatter ----------------------------------------------------


def format_confirmation_footer(proposals: list[PendingUpdate]) -> str:
    """Build the inline footer the orchestrator appends to a response.

    Empty list → empty string (caller doesn't append anything).
    """
    if not proposals:
        return ""

    lines = [
        "",
        "---",
        "While we were talking, I noticed something worth noting in your profile:",
    ]
    for idx, p in enumerate(proposals, start=1):
        summary = _summarize_proposal(p)
        lines.append(f"  {idx}. {summary}")
        if p.extractor_reasoning:
            lines.append(f"     ({p.extractor_reasoning})")

    if len(proposals) == 1:
        lines.append("")
        lines.append('Reply "yes" to add it, "no" to skip.')
    else:
        lines.append("")
        lines.append(
            'Reply "yes" to add them all, "no" to skip, '
            'or specific numbers like "1" or "1 2" to pick.'
        )
    return "\n".join(lines)


def _summarize_proposal(p: PendingUpdate) -> str:
    """Human-readable one-liner for the footer."""
    val = p.proposed_value
    section = p.target_section
    if section == "medications":
        bits = [val.get("name", "?")]
        if val.get("dose"):
            bits.append(val["dose"])
        if val.get("status"):
            bits.append(f"({val['status']})")
        return f"{p.operation} medication: " + " ".join(bits)
    if section == "conditions":
        bits = [val.get("name", "?")]
        if val.get("status"):
            bits.append(f"[{val['status']}]")
        return f"{p.operation} condition: " + " ".join(bits)
    if section == "allergies":
        return f"{p.operation} allergy: {val.get('substance', '?')}"
    if section == "family_history":
        return (
            f"{p.operation} family history: "
            f"{val.get('relation', '?')} — {val.get('condition', '?')}"
        )
    if section == "labs":
        bits = [val.get("test_name", "?")]
        if val.get("value"):
            bits.append(f"= {val['value']}")
        return f"{p.operation} lab: " + " ".join(bits)
    if section == "concerns":
        text = val.get("text", "?")
        if len(text) > 80:
            text = text[:77] + "..."
        return f"{p.operation} concern: {text}"
    if section == "lifestyle":
        # value usually has one field set, e.g. {"smoking": "former"}
        first_field = next(iter(val.items()), (None, None))
        return f"{p.operation} lifestyle: {first_field[0]} = {first_field[1]}"
    if section == "demographics":
        first_field = next(iter(val.items()), (None, None))
        return f"{p.operation} demographics: {first_field[0]} = {first_field[1]}"
    return f"{p.operation} {section}"


# --- Application logic ---------------------------------------------------


def apply_proposal(profile: HealthProfile, proposal: PendingUpdate) -> bool:
    """Apply one accepted proposal to the profile in-place.

    Returns True when the proposal was applied, False when it was
    skipped (unsupported section/operation combo, malformed value
    that survived defensive defaults).

    All accepted proposals get `field_meta.source = "extracted_confirmed"` —
    the spec's provenance shape for "extractor proposed, user accepted."

    Defensive defaults: extractor output often omits required fields
    (e.g. medication without `status`) or uses near-synonyms (`frequency`
    instead of `dose`). `_defensive_defaults` fills in the common cases
    so the proposal lands instead of being silently rejected.
    """
    section = proposal.target_section
    op = proposal.operation
    val = _defensive_defaults(section, proposal.proposed_value or {})
    now = _utc_now()

    if section in _LIST_SECTIONS and op == "add":
        cls, attr = _LIST_SECTIONS[section]
        try:
            entry = cls(
                **{k: v for k, v in val.items() if k != "field_meta"},
                field_meta=_extracted_provenance(val, now),
            )
        except Exception as e:
            log.warning(
                "apply_proposal: could not build %s entry from %r: %s",
                section, val, e,
            )
            return False
        getattr(profile, attr).append(entry)
        return True

    if section in _LIST_SECTIONS and op == "modify":
        cls, attr = _LIST_SECTIONS[section]
        idx, existing = _find_existing_entry(profile, attr, val, section)
        if existing is None:
            log.info(
                "apply_proposal: modify on %s but no matching entry "
                "for %r — skipping (user can re-add it)",
                section, val,
            )
            return False
        # Overlay proposed fields onto the existing entry. Unknown keys
        # in `val` are silently dropped (Pydantic default extra='ignore').
        # field_meta is rebuilt to keep added_at from the original but
        # update last_modified to now — preserves the provenance trail.
        existing_dump = existing.model_dump()
        for k, v in val.items():
            if k == "field_meta":
                continue
            existing_dump[k] = v
        existing_dump["field_meta"] = _modify_provenance(
            existing.field_meta, existing_dump, now,
        )
        try:
            new_entry = cls(**existing_dump)
        except Exception as e:
            log.warning(
                "apply_proposal: modify on %s failed validation for %r: %s",
                section, val, e,
            )
            return False
        getattr(profile, attr)[idx] = new_entry
        return True

    if section in _RECORD_SECTIONS and op == "modify":
        allowed = _RECORD_SECTIONS[section]
        target = profile.demographics if section == "demographics" else profile.lifestyle
        applied_any = False
        for key, raw_value in val.items():
            if key not in allowed:
                log.debug(
                    "apply_proposal: ignoring unknown field %r in %s",
                    key, section,
                )
                continue
            field_obj = ProfileField(
                value=raw_value,
                added_at=now,
                last_modified=now,
                confirmed=True,
                source="extracted_confirmed",
                confidence=1.0,
            )
            setattr(target, key, field_obj)
            applied_any = True
        return applied_any

    log.info(
        "apply_proposal: unsupported combo section=%r operation=%r — skipped",
        section, op,
    )
    return False


def mark_proposals(
    profile: HealthProfile,
    proposal_ids: list[str],
    status: str,
) -> int:
    """Update the status of the named proposals in pending_updates.

    Returns the count actually updated. Pending updates are immutable
    via `model_copy(update=...)` since they're frozen Pydantic models,
    so we rebuild the list with the updated entries.
    """
    if status not in ("pending", "accepted", "rejected", "auto_expired"):
        raise ValueError(f"unknown proposal status: {status!r}")
    target = set(proposal_ids)
    if not target:
        return 0

    updated = 0
    new_list: list[PendingUpdate] = []
    for p in profile.pending_updates:
        if p.id in target and p.status == "pending":
            new_list.append(p.model_copy(update={"status": status}))
            updated += 1
        else:
            new_list.append(p)
    profile.pending_updates = new_list
    return updated


def auto_expire_old(
    profile: HealthProfile, *, ttl_hours: int = DEFAULT_PENDING_TTL_HOURS,
    now: Optional[datetime] = None,
) -> int:
    """Mark pending proposals older than `ttl_hours` as `auto_expired`.

    Spec §G.4: silence is not consent. A timed-out proposal becomes
    auto_expired (not auto_accepted); the user has to re-prompt to
    re-add it.

    Returns the count expired in this pass. Idempotent — re-running
    on the same profile does nothing the second time.
    """
    now_dt = now if now is not None else _utc_now()
    cutoff = now_dt - timedelta(hours=ttl_hours)

    expired_ids: list[str] = []
    for p in profile.pending_updates:
        if p.status == "pending" and p.proposed_at < cutoff:
            expired_ids.append(p.id)

    if expired_ids:
        mark_proposals(profile, expired_ids, "auto_expired")
        log.info("auto_expire_old: %d proposal(s) auto-expired", len(expired_ids))
    return len(expired_ids)


# --- Helpers ------------------------------------------------------------


def _extracted_provenance(value: dict, now: datetime) -> ProfileField:
    """ProfileField wrapper for a freshly-added extractor proposal."""
    # Use the headline-ish field for the wrapper's `value` field —
    # this is informational, not load-bearing.
    headline = _headline_for(value)
    return ProfileField(
        value=headline,
        added_at=now,
        last_modified=now,
        confirmed=True,
        source="extracted_confirmed",
        confidence=1.0,
    )


def _modify_provenance(
    existing: ProfileField, value: dict, now: datetime,
) -> ProfileField:
    """ProfileField wrapper for an entry being MODIFIED via a proposal.

    Preserves `added_at` (the original creation timestamp) and `notes`
    (the user's own annotation), so the modification doesn't erase
    when-was-this-first-recorded. Updates `last_modified` to now and
    re-stamps source as `extracted_confirmed`.
    """
    return ProfileField(
        value=_headline_for(value),
        added_at=existing.added_at,
        last_modified=now,
        confirmed=True,
        source="extracted_confirmed",
        confidence=1.0,
        notes=existing.notes,
    )


def _headline_for(value: dict) -> str:
    """Pick the section-agnostic headline field for ProfileField.value."""
    return (
        value.get("name")
        or value.get("substance")
        or value.get("test_name")
        or value.get("relation")
        or value.get("text")
        or "(extracted)"
    )


def _find_existing_entry(
    profile, attr: str, value: dict, section: str,
) -> tuple[int | None, object | None]:
    """Locate the existing entry in `profile.<attr>` that the modify
    proposal refers to. Match is case-insensitive + whitespace-trimmed
    on the identifier fields for the section.

    Returns `(index, entry)` for the first match, or `(None, None)` if
    no match (or the proposal's identifier fields are missing).

    "First match" is a deliberate simplification — if the user has two
    entries with the same name (rare but possible), the older one
    wins. A future fix could prefer the active/non-superseded one.
    """
    identifiers = _LIST_SECTION_IDENTIFIERS.get(section, ())
    if not identifiers:
        return None, None

    def _norm(s) -> str:
        return (s or "").strip().lower() if isinstance(s, str) else ""

    target = {f: _norm(value.get(f)) for f in identifiers}
    if not all(target.values()):
        # The proposed_value didn't include enough to identify which
        # entry to modify. Don't guess — return no match.
        log.debug(
            "modify on %s missing identifier fields %s in %r",
            section, identifiers, value,
        )
        return None, None

    items = getattr(profile, attr)
    for i, entry in enumerate(items):
        candidate = {f: _norm(getattr(entry, f, None)) for f in identifiers}
        if candidate == target:
            return i, entry
    return None, None


def _utc_now() -> datetime:
    """Naive UTC now() — datetime.utcnow() is deprecated in 3.12+."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _defensive_defaults(section: str, value: dict) -> dict:
    """Fill in commonly-missing required fields with sensible defaults.

    Real-world failure that motivated this: extractor returned
    `{'name': 'CoQ10', 'frequency': 'daily'}` for a medication. The
    Medication Pydantic model requires `status` so the apply silently
    rejected. With this default, the same input lands as
    `{'name': 'CoQ10', 'status': 'active', 'frequency': 'daily'}` —
    `frequency` is ignored (Medication has no such field) and the
    entry is constructed cleanly with the user's `name`.

    Defaults only fill REQUIRED fields the extractor is most likely
    to omit. Optional fields stay unset.
    """
    val = dict(value)  # don't mutate caller's dict
    now = _utc_now()

    if section == "medications":
        # Required: name, status. Status defaults to "active" since
        # the extractor only fires when the user mentioned taking
        # something currently.
        val.setdefault("status", "active")
    elif section == "conditions":
        # Required: name, status. Same logic — extractor surfaces
        # things the user currently has.
        val.setdefault("status", "active")
    elif section == "concerns":
        # Required: text, raised_at, status. Default the latter two.
        val.setdefault("status", "active")
        val.setdefault("raised_at", now.isoformat())
    # allergies, family_history, labs have no required Literal fields
    # the extractor commonly omits — let them through unchanged.
    return val
