"""Tests for the profile confirmation surface (Phase G.2.b).

Three layers:
- parse_user_reply — pure function over (message, pending_count).
  No DB, no profile. Covers every recognized intent + the
  pass-through cases that must NOT be treated as replies.
- apply_proposal — pure function over (profile, proposal). Covers
  each supported section + the unsupported combos that skip cleanly.
- format_confirmation_footer — rendering for single vs multiple
  proposals, summary text per section type.
- mark_proposals + auto_expire_old — small state-mutation helpers.
- End-to-end: extract + persist + footer + reply parsing → apply
  cycle (no orchestrator — that's covered by integration tests).
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import apsw
import pytest

from src.health.profile import (
    Allergy,
    Concern,
    Condition,
    HealthProfile,
    Medication,
    PendingUpdate,
    ProfileField,
    init_tables,
    load_profile,
    save_profile,
)
from src.health.profile.confirmation import (
    DEFAULT_PENDING_TTL_HOURS,
    ProposalReply,
    ProposalReplyIntent,
    apply_proposal,
    auto_expire_old,
    format_confirmation_footer,
    mark_proposals,
    parse_user_reply,
)
from src.health.profile.crypto import ENV_MASTER_KEY


@pytest.fixture(autouse=True)
def env_master_key(monkeypatch):
    monkeypatch.setenv(
        ENV_MASTER_KEY,
        base64.b64encode(b"test-key-exactly-32-bytes-long!!").decode(),
    )


@pytest.fixture
def conn() -> apsw.Connection:
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _provenance(value: str = "x") -> ProfileField:
    """Minimal ProfileField wrapper for tests that don't care about
    the wrapper's specific values — just need a valid field_meta on
    a Condition / Medication / etc."""
    now = _utc_now()
    return ProfileField(
        value=value, added_at=now, last_modified=now,
        confirmed=True, source="user_stated_setup", confidence=1.0,
    )


def _proposal(
    section: str = "medications",
    operation: str = "add",
    value: dict | None = None,
    pid: str = "prop_test_1",
    proposed_at: datetime | None = None,
) -> PendingUpdate:
    return PendingUpdate(
        id=pid,
        proposed_at=proposed_at or _utc_now(),
        target_section=section,
        operation=operation,
        proposed_value=value or {"name": "metformin", "status": "active"},
        source_turn_id="turn-1",
        extractor_reasoning="test",
        status="pending",
    )


# --- parse_user_reply -----------------------------------------------------


class TestParseUserReply:
    def test_no_pending_always_normal(self):
        for msg in ["yes", "no", "1", "anything"]:
            assert parse_user_reply(msg, pending_count=0).intent == ProposalReplyIntent.NORMAL

    @pytest.mark.parametrize("msg", ["yes", "yep", "yeah", "yup", "sure", "ok", "okay", "accept", "confirm", "all"])
    def test_yes_tokens(self, msg):
        assert parse_user_reply(msg, pending_count=2).intent == ProposalReplyIntent.YES

    @pytest.mark.parametrize("msg", ["YES", "Yes.", "Yes!"])
    def test_yes_case_and_punctuation(self, msg):
        assert parse_user_reply(msg, pending_count=2).intent == ProposalReplyIntent.YES

    def test_yes_two_word_variants(self):
        for msg in ["yes please", "accept all", "yes all", "ok please"]:
            assert parse_user_reply(msg, pending_count=2).intent == ProposalReplyIntent.YES

    @pytest.mark.parametrize("msg", ["no", "nope", "skip", "reject", "dismiss", "decline"])
    def test_no_tokens(self, msg):
        assert parse_user_reply(msg, pending_count=2).intent == ProposalReplyIntent.NO

    def test_no_thanks_is_no_not_yes(self):
        # "no thanks" starts with "no" — that's a no, NOT a yes-typo.
        assert parse_user_reply("no thanks", pending_count=2).intent == ProposalReplyIntent.NO

    def test_index_single(self):
        r = parse_user_reply("1", pending_count=3)
        assert r.intent == ProposalReplyIntent.INDICES
        assert r.indices == (1,)

    def test_index_multiple_space_sep(self):
        r = parse_user_reply("1 3", pending_count=3)
        assert r.intent == ProposalReplyIntent.INDICES
        assert r.indices == (1, 3)

    def test_index_multiple_comma_sep(self):
        r = parse_user_reply("1, 2, 3", pending_count=3)
        assert r.indices == (1, 2, 3)

    def test_index_out_of_range_dropped(self):
        # 5 is out of range; 1 is kept
        r = parse_user_reply("1 5", pending_count=2)
        assert r.indices == (1,)

    def test_index_all_out_of_range_falls_through(self):
        r = parse_user_reply("5 6 7", pending_count=2)
        assert r.intent == ProposalReplyIntent.NORMAL

    def test_index_dedupes(self):
        r = parse_user_reply("1 1 2", pending_count=2)
        assert r.indices == (1, 2)

    def test_long_message_passes_through(self):
        # "yes I have another question..." would be a false-positive
        # accept; the parser correctly leaves it as NORMAL.
        msg = "yes also what time is the meeting tomorrow"
        assert parse_user_reply(msg, pending_count=2).intent == ProposalReplyIntent.NORMAL

    def test_empty_message_is_normal(self):
        assert parse_user_reply("", pending_count=2).intent == ProposalReplyIntent.NORMAL
        assert parse_user_reply("   ", pending_count=2).intent == ProposalReplyIntent.NORMAL

    def test_unrelated_short_message_normal(self):
        assert parse_user_reply("hello", pending_count=2).intent == ProposalReplyIntent.NORMAL


# --- apply_proposal -------------------------------------------------------


class TestApplyProposal:
    def test_add_medication(self):
        p = HealthProfile.empty("self")
        prop = _proposal(value={"name": "metformin", "dose": "500mg", "status": "active"})
        applied = apply_proposal(p, prop)
        assert applied is True
        assert len(p.medications) == 1
        med = p.medications[0]
        assert med.name == "metformin"
        assert med.dose == "500mg"
        # field_meta carries the extractor_confirmed provenance
        assert med.field_meta.source == "extracted_confirmed"
        assert med.field_meta.confirmed is True

    def test_add_condition(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="conditions",
            value={"name": "Type 2 Diabetes", "status": "active"},
        )
        assert apply_proposal(p, prop) is True
        assert len(p.conditions) == 1
        assert p.conditions[0].name == "Type 2 Diabetes"

    def test_add_allergy(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="allergies",
            value={"substance": "penicillin", "severity": "severe"},
        )
        assert apply_proposal(p, prop) is True
        assert p.allergies[0].substance == "penicillin"

    def test_add_concern(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="concerns",
            value={
                "text": "dull headache 3 days",
                "raised_at": _utc_now().isoformat(),
                "status": "active",
            },
        )
        assert apply_proposal(p, prop) is True
        assert "headache" in p.concerns[0].text

    def test_modify_lifestyle_field(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="lifestyle", operation="modify",
            value={"smoking": "former"},
        )
        assert apply_proposal(p, prop) is True
        assert p.lifestyle.smoking is not None
        assert p.lifestyle.smoking.value == "former"
        assert p.lifestyle.smoking.source == "extracted_confirmed"

    def test_modify_demographics_field(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="demographics", operation="modify",
            value={"age_range": "30-39"},
        )
        assert apply_proposal(p, prop) is True
        assert p.demographics.age_range.value == "30-39"

    def test_modify_ignores_unknown_field(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="demographics", operation="modify",
            value={"made_up_field": "x"},
        )
        # No valid field → no application
        assert apply_proposal(p, prop) is False

    def test_unsupported_section_skips(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="not_a_real_section", operation="add",
            value={"x": 1},
        )
        assert apply_proposal(p, prop) is False

    def test_remove_operation_skips_today(self):
        # remove isn't supported yet (covered by `modify status=discontinued`
        # for the common discontinue case)
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="medications", operation="remove",
            value={"name": "metformin"},
        )
        assert apply_proposal(p, prop) is False

    def test_malformed_value_skips_gracefully(self):
        # Pydantic will reject — apply_proposal returns False, no raise.
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="medications", operation="add",
            value={"name": "metformin", "status": "not-a-valid-status"},
        )
        assert apply_proposal(p, prop) is False
        assert p.medications == []


# --- Defensive defaults (regression: real-world bug 2026-05-23) ----------


class TestDefensiveDefaults:
    """The extractor often omits required fields like `status`. Without
    defaults, those proposals silently fail to apply (Pydantic raises,
    apply_proposal logs + returns False, user sees nothing). With
    defaults, the common cases land cleanly."""

    def test_coq10_case_from_real_bug(self):
        """The literal proposal that surfaced the bug in production —
        extractor returned name + frequency (no status, no dose).
        With defaults: status='active', frequency silently dropped
        by Pydantic, medication entry created with name only."""
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="medications",
            value={"name": "CoQ10", "frequency": "daily"},
        )
        assert apply_proposal(p, prop) is True
        assert len(p.medications) == 1
        med = p.medications[0]
        assert med.name == "CoQ10"
        assert med.status == "active"  # defaulted
        # `frequency` isn't a Medication field — Pydantic ignores it
        # cleanly (extra='ignore' is the default for v2 BaseModel).

    def test_medication_missing_status_defaults_active(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="medications", value={"name": "metformin"},
        )
        assert apply_proposal(p, prop) is True
        assert p.medications[0].status == "active"

    def test_explicit_status_not_overridden(self):
        # Default only fires when the field is MISSING.
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="medications",
            value={"name": "old-prescription", "status": "discontinued"},
        )
        assert apply_proposal(p, prop) is True
        assert p.medications[0].status == "discontinued"

    def test_condition_missing_status_defaults_active(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="conditions", value={"name": "Type 2 Diabetes"},
        )
        assert apply_proposal(p, prop) is True
        assert p.conditions[0].status == "active"

    def test_concern_missing_status_and_raised_at_defaults(self):
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="concerns",
            value={"text": "dull headache three days"},
        )
        assert apply_proposal(p, prop) is True
        c = p.concerns[0]
        assert c.status == "active"
        # raised_at filled with now-ish — just confirm it's set
        assert c.raised_at is not None

    def test_allergy_no_required_status_field(self):
        # Allergy doesn't have a required-with-no-default field that
        # we'd default. Verify it lands without the helper interfering.
        p = HealthProfile.empty("self")
        prop = _proposal(
            section="allergies",
            value={"substance": "penicillin"},
        )
        assert apply_proposal(p, prop) is True
        assert p.allergies[0].substance == "penicillin"
        # severity stays None (optional)
        assert p.allergies[0].severity is None


# --- Modify list sections (regression: real-world bug 2026-05-25) -------


class TestModifyListSection:
    """Real bug: user said 'I'll drop the DHEA' → extractor proposed
    {section: medications, op: modify, value: {name: DHEA, status: discontinued}}.
    Before this fix, apply_proposal returned False because modify on
    list sections wasn't wired."""

    def _setup_with_dhea(self) -> HealthProfile:
        p = HealthProfile.empty("self")
        added_long_ago = datetime(2026, 1, 1, 12, 0, 0)
        p.medications.append(Medication(
            name="DHEA", dose="100mg daily", status="active",
            field_meta=ProfileField(
                value="DHEA",
                added_at=added_long_ago,
                last_modified=added_long_ago,
                confirmed=True,
                source="user_stated_inline",
                confidence=1.0,
                notes="started for energy",
            ),
        ))
        return p

    def test_modify_medication_status_to_discontinued(self):
        """The literal case from the screenshot."""
        p = self._setup_with_dhea()
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "DHEA", "status": "discontinued"},
        )
        assert apply_proposal(p, prop) is True
        # Same entry, status updated; dose preserved
        assert len(p.medications) == 1
        assert p.medications[0].name == "DHEA"
        assert p.medications[0].status == "discontinued"
        assert p.medications[0].dose == "100mg daily"

    def test_modify_preserves_unmodified_fields(self):
        p = self._setup_with_dhea()
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "DHEA", "status": "discontinued"},
        )
        apply_proposal(p, prop)
        med = p.medications[0]
        # field_meta.added_at preserved; last_modified bumped
        assert med.field_meta.added_at == datetime(2026, 1, 1, 12, 0, 0)
        assert med.field_meta.last_modified > med.field_meta.added_at
        # Original user note preserved
        assert med.field_meta.notes == "started for energy"
        # Source restamped to extracted_confirmed
        assert med.field_meta.source == "extracted_confirmed"

    def test_modify_case_insensitive_identifier_match(self):
        # User says "dhea" lowercase; existing entry is "DHEA" uppercase.
        # Should still match.
        p = self._setup_with_dhea()
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "dhea", "status": "discontinued"},
        )
        assert apply_proposal(p, prop) is True
        assert p.medications[0].status == "discontinued"

    def test_modify_nonexistent_returns_false(self):
        """When there's no matching entry, apply returns False with a
        clear log line — the user gets the 'couldn't apply' warning so
        they know to add it first."""
        p = HealthProfile.empty("self")  # no DHEA at all
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "DHEA", "status": "discontinued"},
        )
        assert apply_proposal(p, prop) is False
        # And nothing was added (we don't accidentally turn modify into add)
        assert p.medications == []

    def test_modify_missing_identifier_returns_false(self):
        # No name → can't find existing → no match → False
        p = self._setup_with_dhea()
        prop = _proposal(
            section="medications", operation="modify",
            value={"status": "discontinued"},  # no name
        )
        assert apply_proposal(p, prop) is False
        # Existing entry untouched
        assert p.medications[0].status == "active"

    def test_modify_first_match_wins_with_duplicates(self):
        # Two entries with the same name — first one gets modified.
        p = self._setup_with_dhea()
        added_recent = datetime(2026, 4, 1, 12, 0, 0)
        p.medications.append(Medication(
            name="DHEA", dose="50mg daily", status="active",
            field_meta=ProfileField(
                value="DHEA", added_at=added_recent,
                last_modified=added_recent, confirmed=True,
                source="user_stated_inline", confidence=1.0,
            ),
        ))
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "DHEA", "status": "discontinued"},
        )
        apply_proposal(p, prop)
        # First entry (older) modified
        assert p.medications[0].status == "discontinued"
        assert p.medications[0].dose == "100mg daily"
        # Second entry untouched
        assert p.medications[1].status == "active"

    def test_modify_condition_status_to_resolved(self):
        # Same pattern for conditions.
        p = HealthProfile.empty("self")
        p.conditions.append(Condition(
            name="Seasonal allergies", status="active",
            field_meta=_provenance(),
        ))
        prop = _proposal(
            section="conditions", operation="modify",
            value={"name": "Seasonal allergies", "status": "resolved"},
        )
        assert apply_proposal(p, prop) is True
        assert p.conditions[0].status == "resolved"

    def test_modify_compound_identifier_family_history(self):
        # family_history uses (relation, condition) as the compound id.
        p = HealthProfile.empty("self")
        from src.health.profile import FamilyHistoryEntry
        p.family_history.append(FamilyHistoryEntry(
            relation="mother", condition="diabetes",
            field_meta=_provenance(),
        ))
        prop = _proposal(
            section="family_history", operation="modify",
            value={
                "relation": "mother", "condition": "diabetes",
                "age_of_onset": "55",
            },
        )
        assert apply_proposal(p, prop) is True
        assert p.family_history[0].age_of_onset == "55"

    def test_modify_invalid_value_returns_false(self):
        # Proposed status not in the Literal → Pydantic rejects → False
        p = self._setup_with_dhea()
        prop = _proposal(
            section="medications", operation="modify",
            value={"name": "DHEA", "status": "not-a-status"},
        )
        assert apply_proposal(p, prop) is False
        # Original entry untouched
        assert p.medications[0].status == "active"


# --- format_confirmation_footer ------------------------------------------


class TestFooter:
    def test_empty_proposals_empty_footer(self):
        assert format_confirmation_footer([]) == ""

    def test_single_proposal_singular_prompt(self):
        prop = _proposal(value={"name": "metformin", "dose": "500mg"})
        text = format_confirmation_footer([prop])
        assert "metformin" in text
        assert "500mg" in text
        assert '"yes" to add it' in text
        # Should NOT have the multi-pick prompt
        assert "specific numbers" not in text

    def test_multiple_proposals_plural_prompt(self):
        a = _proposal(value={"name": "metformin"}, pid="a")
        b = _proposal(
            section="concerns", value={"text": "headache"}, pid="b",
        )
        text = format_confirmation_footer([a, b])
        assert "1." in text
        assert "2." in text
        assert "metformin" in text
        assert "headache" in text
        assert "specific numbers" in text

    def test_long_concern_text_truncates(self):
        long_text = "x" * 200
        prop = _proposal(
            section="concerns", value={"text": long_text}, pid="long",
        )
        text = format_confirmation_footer([prop])
        assert "..." in text
        # And the full 200-char string isn't in the footer
        assert long_text not in text

    def test_reasoning_shown_when_present(self):
        prop = PendingUpdate(
            id="p1",
            proposed_at=_utc_now(),
            target_section="medications",
            operation="add",
            proposed_value={"name": "metformin"},
            source_turn_id="t1",
            extractor_reasoning="User stated they take metformin daily",
            status="pending",
        )
        text = format_confirmation_footer([prop])
        assert "User stated they take metformin daily" in text


# --- mark_proposals + auto_expire_old -----------------------------------


class TestMarkProposals:
    def test_mark_changes_status(self):
        p = HealthProfile.empty("self")
        p.pending_updates.append(_proposal(pid="a"))
        p.pending_updates.append(_proposal(pid="b"))
        updated = mark_proposals(p, ["a"], "rejected")
        assert updated == 1
        statuses = {pp.id: pp.status for pp in p.pending_updates}
        assert statuses == {"a": "rejected", "b": "pending"}

    def test_mark_only_pending(self):
        # Already-rejected can't be re-rejected
        p = HealthProfile.empty("self")
        rejected = _proposal(pid="r").model_copy(update={"status": "rejected"})
        p.pending_updates.append(rejected)
        updated = mark_proposals(p, ["r"], "accepted")
        assert updated == 0
        assert p.pending_updates[0].status == "rejected"

    def test_unknown_status_raises(self):
        p = HealthProfile.empty("self")
        with pytest.raises(ValueError, match="unknown proposal status"):
            mark_proposals(p, [], "bogus")

    def test_empty_ids_noop(self):
        p = HealthProfile.empty("self")
        p.pending_updates.append(_proposal(pid="a"))
        assert mark_proposals(p, [], "accepted") == 0


class TestAutoExpire:
    def test_old_pending_expired(self):
        now = _utc_now()
        old = now - timedelta(hours=DEFAULT_PENDING_TTL_HOURS + 1)
        recent = now - timedelta(hours=1)
        p = HealthProfile.empty("self")
        p.pending_updates.append(_proposal(pid="old", proposed_at=old))
        p.pending_updates.append(_proposal(pid="recent", proposed_at=recent))
        n = auto_expire_old(p, now=now)
        assert n == 1
        by_id = {pp.id: pp.status for pp in p.pending_updates}
        assert by_id == {"old": "auto_expired", "recent": "pending"}

    def test_already_expired_not_touched(self):
        # Idempotent — re-running doesn't double-expire.
        now = _utc_now()
        old = now - timedelta(hours=DEFAULT_PENDING_TTL_HOURS + 1)
        p = HealthProfile.empty("self")
        p.pending_updates.append(_proposal(pid="old", proposed_at=old))
        auto_expire_old(p, now=now)
        n2 = auto_expire_old(p, now=now)
        assert n2 == 0
        assert p.pending_updates[0].status == "auto_expired"

    def test_custom_ttl(self):
        now = _utc_now()
        p = HealthProfile.empty("self")
        p.pending_updates.append(_proposal(
            pid="x", proposed_at=now - timedelta(hours=2),
        ))
        # With 1h TTL, the 2h-old proposal expires
        assert auto_expire_old(p, ttl_hours=1, now=now) == 1


# --- End-to-end: persist → reply → apply --------------------------------


class TestEndToEnd:
    def test_yes_reply_persists_to_real_section(self, conn):
        """Full integration: an extractor proposal lands in pending_updates;
        user replies 'yes'; we load + apply + save; the profile section
        now contains the entry and the pending_update is accepted."""
        from src.health.profile.extractor import persist_proposals

        prop = _proposal(value={"name": "metformin", "status": "active"})
        persist_proposals(conn=conn, proposals=[prop], key_source="env")

        # Simulate user reply parsing
        reply = parse_user_reply("yes", pending_count=1)
        assert reply.intent == ProposalReplyIntent.YES

        loaded = load_profile(conn, key_source="env")
        in_queue = {pp.id: pp for pp in loaded.profile.pending_updates}
        assert prop.id in in_queue

        # Apply + mark
        assert apply_proposal(loaded.profile, in_queue[prop.id]) is True
        mark_proposals(loaded.profile, [prop.id], "accepted")
        save_profile(
            conn, loaded.profile, expected_version=loaded.version,
            key_source="env",
        )

        # Re-read — medication should be in the section, proposal accepted
        re_loaded = load_profile(conn, key_source="env")
        assert len(re_loaded.profile.medications) == 1
        assert re_loaded.profile.medications[0].name == "metformin"
        # The pending_update is still in the list but with status='accepted'
        accepted = [
            pp for pp in re_loaded.profile.pending_updates
            if pp.status == "accepted"
        ]
        assert len(accepted) == 1

    def test_no_reply_rejects_without_applying(self, conn):
        from src.health.profile.extractor import persist_proposals

        prop = _proposal(value={"name": "metformin"})
        persist_proposals(conn=conn, proposals=[prop], key_source="env")

        reply = parse_user_reply("no", pending_count=1)
        assert reply.intent == ProposalReplyIntent.NO

        loaded = load_profile(conn, key_source="env")
        mark_proposals(loaded.profile, [prop.id], "rejected")
        save_profile(
            conn, loaded.profile, expected_version=loaded.version,
            key_source="env",
        )

        re_loaded = load_profile(conn, key_source="env")
        # No medication added
        assert re_loaded.profile.medications == []
        # Proposal status=rejected
        assert re_loaded.profile.pending_updates[0].status == "rejected"

    def test_index_reply_partial_accept(self, conn):
        from src.health.profile.extractor import persist_proposals

        p1 = _proposal(value={"name": "metformin", "status": "active"}, pid="p1")
        p2 = _proposal(
            section="conditions",
            value={"name": "T2D", "status": "active"},
            pid="p2",
        )
        persist_proposals(conn=conn, proposals=[p1, p2], key_source="env")

        # User accepts only #1
        reply = parse_user_reply("1", pending_count=2)
        assert reply.indices == (1,)

        loaded = load_profile(conn, key_source="env")
        in_queue = {pp.id: pp for pp in loaded.profile.pending_updates}

        # Apply just p1, reject p2
        apply_proposal(loaded.profile, in_queue["p1"])
        mark_proposals(loaded.profile, ["p1"], "accepted")
        mark_proposals(loaded.profile, ["p2"], "rejected")
        save_profile(
            conn, loaded.profile, expected_version=loaded.version,
            key_source="env",
        )

        re_loaded = load_profile(conn, key_source="env")
        assert len(re_loaded.profile.medications) == 1
        assert re_loaded.profile.conditions == []
        statuses = {pp.id: pp.status for pp in re_loaded.profile.pending_updates}
        assert statuses == {"p1": "accepted", "p2": "rejected"}
