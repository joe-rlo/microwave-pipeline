"""Pydantic models for the Health Profile (Phase G.1.b).

Direct port of the schema in `microwave-health-profile-spec.md`. Every
value that holds user-stated data is wrapped in `ProfileField` for
provenance — when was it added, was it user-confirmed or extracted,
where did it come from. That's the design discipline the spec calls
out as load-bearing for trust.

Model layout:

- ProfileField — provenance wrapper for one value
- Demographics — age range, sex/gender, height/weight buckets
- Condition / Medication / Allergy / FamilyHistoryEntry / LabResult —
  list-shaped sections, one entry per item, each with field_meta
- LifestyleFactors — flat record of habit-shaped fields
- Concern — user's own words about things on their mind
- PendingUpdate — extractor proposals awaiting user review (Phase G.2)
- DeletedEntry — soft-delete buffer (30-day undo window, Phase G.1.c)
- HealthProfile — top-level container of everything above

Why bucketed numeric fields (age_range, height_range, weight_range):
exact age + zip is enough to reidentify someone in many datasets.
Ranges give the model enough to reason ("user is in their 40s")
without storing identifiers. Spec §H.1 calls this out explicitly.

Why field_meta on every entry, not just top-level: a condition added
two years ago is different information than one added yesterday. The
LLM needs the temporal signal, and per-item provenance is what gives
it.

Why Pydantic v2 BaseModel: most of the codebase uses Pydantic v2 (see
HealthConfig). Consistency wins. The validation it gives us at JSON
parse time matters most when loading a profile from disk that was
written by a previous version of the schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- Provenance wrapper ----------------------------------------------------


class ProfileField(BaseModel):
    """Wraps one value with the context of how it got here."""

    value: Any
    added_at: datetime
    last_modified: datetime
    confirmed: bool                  # True if user explicitly accepted
    source: Literal[
        "user_stated_setup",         # entered during initial setup
        "user_stated_inline",        # said directly in conversation
        "extracted_confirmed",       # extractor proposed, user accepted
        "extracted_inferred",        # extractor proposed, user did not respond
        "user_imported",             # bulk import (future)
    ]
    confidence: float                # 0-1; 1.0 if user_stated
    notes: Optional[str] = None      # user's own annotation


# --- Sections --------------------------------------------------------------


class Demographics(BaseModel):
    """Bucketed-only by design — see module docstring on reidentification."""

    age_range: Optional[ProfileField] = None              # "30-39", "40-49"
    sex_assigned_at_birth: Optional[ProfileField] = None
    gender_identity: Optional[ProfileField] = None
    height_range: Optional[ProfileField] = None
    weight_range: Optional[ProfileField] = None
    pregnancy_status: Optional[ProfileField] = None       # when applicable


class Condition(BaseModel):
    name: str                                              # canonical or user's term
    canonical_term: Optional[str] = None                   # mapped to MedlinePlus where possible
    status: Literal["active", "resolved", "in_remission", "monitoring"]
    diagnosed_when: Optional[str] = None                   # "approximate:2022", "exact:2024-03-15"
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    notes: Optional[str] = None
    field_meta: ProfileField


class Medication(BaseModel):
    name: str
    canonical_name: Optional[str] = None                   # mapped via openFDA (Phase G.4)
    dose: Optional[str] = None                             # "500mg twice daily"
    started: Optional[str] = None
    stopped: Optional[str] = None                          # if status="discontinued"
    status: Literal["active", "as_needed", "discontinued"]
    reason: Optional[str] = None                           # condition it treats
    field_meta: ProfileField


class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe", "anaphylactic"]] = None
    field_meta: ProfileField


class FamilyHistoryEntry(BaseModel):
    relation: str                                           # "mother", "paternal grandfather"
    condition: str
    age_of_onset: Optional[str] = None
    field_meta: ProfileField


class LifestyleFactors(BaseModel):
    smoking: Optional[ProfileField] = None                  # "never" / "former" / "current"
    alcohol: Optional[ProfileField] = None                  # "none" / "occasional" / "regular" / "heavy"
    exercise_frequency: Optional[ProfileField] = None       # bucketed
    sleep_hours_typical: Optional[ProfileField] = None      # bucketed
    diet_pattern: Optional[ProfileField] = None             # free text, user's words


class LabResult(BaseModel):
    test_name: str
    value: str                                              # user's words, not parsed
    units: Optional[str] = None
    date: Optional[str] = None
    reference_range: Optional[str] = None
    field_meta: ProfileField


class Concern(BaseModel):
    text: str                                               # user's own words
    raised_at: datetime
    status: Literal["active", "addressed", "ongoing"]
    field_meta: ProfileField


# --- Queue items (Phase G.2 + G.1.c) --------------------------------------


class PendingUpdate(BaseModel):
    """Extractor proposal awaiting user review.

    Lives inside the HealthProfile rather than a separate table — that
    way the user's view of "things waiting for my attention" is
    co-located with the rest of the profile.
    """

    id: str                                                 # short id for "yes 3"
    proposed_at: datetime
    target_section: str                                     # "medications", etc.
    operation: Literal["add", "modify", "remove"]
    proposed_value: dict
    source_turn_id: str                                     # link back to triggering turn
    extractor_reasoning: str                                # short user-facing explanation
    status: Literal["pending", "accepted", "rejected", "auto_expired"]


class DeletedEntry(BaseModel):
    """One soft-deleted entry awaiting permanent purge.

    The user gets a 30-day undo window. After purges_at, the entry is
    eligible for permanent deletion by a maintenance job.
    """

    deleted_at: datetime
    section: str
    original_value: dict
    deletion_reason: Optional[str] = None
    purges_at: datetime


# --- Top-level container --------------------------------------------------


class HealthProfile(BaseModel):
    """The full profile — one per user.

    Persisted as an encrypted JSON blob in `health_profiles`. The
    serialized form roundtrips cleanly through `model_dump_json()` /
    `model_validate_json()`.
    """

    schema_version: int = 1
    user_id: str
    created_at: datetime
    last_updated: datetime
    demographics: Demographics = Field(default_factory=Demographics)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    family_history: list[FamilyHistoryEntry] = Field(default_factory=list)
    lifestyle: LifestyleFactors = Field(default_factory=LifestyleFactors)
    labs: list[LabResult] = Field(default_factory=list)
    concerns: list[Concern] = Field(default_factory=list)
    pending_updates: list[PendingUpdate] = Field(default_factory=list)
    deleted_recently: list[DeletedEntry] = Field(default_factory=list)

    @classmethod
    def empty(cls, user_id: str, now: Optional[datetime] = None) -> "HealthProfile":
        """Build a fresh profile with all sections empty.

        Used when a fresh-install user runs `/profile show` before any
        data has been entered — gives the CLI / store a real object to
        work with instead of None.
        """
        if now is not None:
            ts = now
        else:
            # datetime.utcnow() is deprecated in 3.12+; use the
            # timezone-aware variant and strip to match field type.
            from datetime import timezone
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
        return cls(
            user_id=user_id,
            created_at=ts,
            last_updated=ts,
        )

    @property
    def is_empty(self) -> bool:
        """True when no user data has been recorded yet.

        Lifestyle and demographics use field-presence (any non-None
        field counts); list sections use length. Pending updates and
        soft-deletes don't count — they're machinery, not user data.
        """
        if any([
            self.demographics.age_range,
            self.demographics.sex_assigned_at_birth,
            self.demographics.gender_identity,
            self.demographics.height_range,
            self.demographics.weight_range,
            self.demographics.pregnancy_status,
        ]):
            return False
        if any([
            self.lifestyle.smoking,
            self.lifestyle.alcohol,
            self.lifestyle.exercise_frequency,
            self.lifestyle.sleep_hours_typical,
            self.lifestyle.diet_pattern,
        ]):
            return False
        return not any([
            self.conditions, self.medications, self.allergies,
            self.family_history, self.labs, self.concerns,
        ])
