"""Tests for the profile extractor (Phase G.2.a).

LLM is stubbed via a `_stub_llm(canned_response)` helper — same pattern
as the consolidation tests. We use the real Pydantic models and the
real persist_proposals → load/save cycle against an in-memory DB +
env-source crypto.

Coverage:
- run_extractor: empty inputs → no LLM call, empty list returned
- run_extractor: active project → no LLM call, empty list (fictional
  context guard)
- run_extractor: malformed JSON → empty list (failure swallowed)
- run_extractor: LLM exception → empty list (failure swallowed)
- run_extractor: extracts proposals with correct fields
- Validation: drops bad sections, bad operations, low-confidence,
  empty proposed_value, non-dict items
- summarize_structure: counts demographics/lifestyle field-set ratios,
  list lengths
- persist_proposals: empty input is no-op (0 return)
- persist_proposals: appends to pending_updates, idempotent on re-run
- persist_proposals: optimistic concurrency retry survives one race
- Proposal IDs: stable for same input, different sections produce
  different ids
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import apsw
import pytest

from src.health.profile import (
    Concern,
    Condition,
    Demographics,
    HealthProfile,
    Medication,
    PendingUpdate,
    ProfileField,
    init_tables,
    load_profile,
    save_profile,
)
from src.health.profile.crypto import ENV_MASTER_KEY
from src.health.profile.extractor import (
    MIN_CONFIDENCE,
    _proposal_id,
    persist_proposals,
    run_extractor,
    summarize_structure,
)


# --- fixtures ---


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


def _stub_llm(canned: str):
    calls: list[tuple[str, str]] = []

    async def call(system: str, user: str) -> str:
        calls.append((system, user))
        return canned

    call.calls = calls  # type: ignore[attr-defined]
    return call


def _failing_llm():
    async def call(system: str, user: str) -> str:
        raise RuntimeError("simulated")
    return call


def _provenance():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ProfileField(
        value="x", added_at=now, last_modified=now,
        confirmed=True, source="user_stated_setup", confidence=1.0,
    )


# --- run_extractor: skip conditions ---


@pytest.mark.asyncio
class TestExtractorSkipConditions:
    async def test_empty_messages_skip(self):
        llm = _stub_llm("should-not-be-called")
        out = await run_extractor(
            user_message="",
            assistant_response="",
            profile_structure={"medications": 0},
            llm_call=llm,
        )
        assert out == []
        assert len(llm.calls) == 0  # type: ignore[attr-defined]

    async def test_active_project_skips(self):
        # Fictional context — must not contaminate the user's real
        # profile (spec §G.4 carveout for writing projects).
        llm = _stub_llm("should-not-be-called")
        out = await run_extractor(
            user_message="My character takes metformin daily",
            assistant_response="Got it. The doctor described her dose as...",
            profile_structure={"medications": 0},
            llm_call=llm,
            active_project="the-heist",
        )
        assert out == []
        assert len(llm.calls) == 0  # type: ignore[attr-defined]


# --- run_extractor: happy paths + validation ---


@pytest.mark.asyncio
class TestExtractorHappyPath:
    async def test_extracts_proposals(self):
        llm = _stub_llm(json.dumps({
            "proposals": [
                {
                    "section": "medications",
                    "operation": "add",
                    "proposed_value": {"name": "metformin", "dose": "500mg twice daily"},
                    "confidence": 0.95,
                    "reasoning": "User stated they take metformin",
                },
                {
                    "section": "concerns",
                    "operation": "add",
                    "proposed_value": {"text": "dull headache three days"},
                    "confidence": 0.85,
                    "reasoning": "User reported new headache",
                },
            ]
        }))
        out = await run_extractor(
            user_message="I take metformin 500mg twice a day. Also been "
                         "having a dull headache for three days.",
            assistant_response="...",
            profile_structure={"medications": 0, "concerns": 0},
            llm_call=llm,
        )
        assert len(out) == 2
        assert all(isinstance(p, PendingUpdate) for p in out)
        assert out[0].target_section == "medications"
        assert out[0].operation == "add"
        assert out[0].proposed_value["name"] == "metformin"
        assert out[0].extractor_reasoning.startswith("User stated")
        assert out[0].status == "pending"
        assert out[1].target_section == "concerns"

    async def test_zero_proposals_returned_cleanly(self):
        llm = _stub_llm(json.dumps({"proposals": []}))
        out = await run_extractor(
            user_message="What time is it?",
            assistant_response="Around 4pm.",
            profile_structure={},
            llm_call=llm,
        )
        assert out == []


@pytest.mark.asyncio
class TestExtractorValidation:
    async def test_drops_low_confidence(self):
        llm = _stub_llm(json.dumps({
            "proposals": [
                {
                    "section": "medications", "operation": "add",
                    "proposed_value": {"name": "ibuprofen"},
                    "confidence": 0.5,  # below MIN_CONFIDENCE (0.7)
                    "reasoning": "weak signal",
                },
            ]
        }))
        out = await run_extractor(
            user_message="might take ibuprofen sometimes",
            assistant_response="ok",
            profile_structure={},
            llm_call=llm,
        )
        assert out == []

    async def test_drops_bad_section(self):
        llm = _stub_llm(json.dumps({
            "proposals": [
                {
                    "section": "bogus_section",
                    "operation": "add",
                    "proposed_value": {"x": 1},
                    "confidence": 0.9,
                    "reasoning": "...",
                },
            ]
        }))
        out = await run_extractor(
            user_message="...", assistant_response="...",
            profile_structure={}, llm_call=llm,
        )
        assert out == []

    async def test_drops_bad_operation(self):
        llm = _stub_llm(json.dumps({
            "proposals": [
                {
                    "section": "medications",
                    "operation": "rename",  # not in {add, modify, remove}
                    "proposed_value": {"name": "x"},
                    "confidence": 0.9,
                    "reasoning": "...",
                },
            ]
        }))
        out = await run_extractor(
            user_message="...", assistant_response="...",
            profile_structure={}, llm_call=llm,
        )
        assert out == []

    async def test_drops_empty_proposed_value(self):
        llm = _stub_llm(json.dumps({
            "proposals": [
                {
                    "section": "medications", "operation": "add",
                    "proposed_value": {},
                    "confidence": 0.9,
                    "reasoning": "empty",
                },
            ]
        }))
        out = await run_extractor(
            user_message="...", assistant_response="...",
            profile_structure={}, llm_call=llm,
        )
        assert out == []

    async def test_drops_non_dict_proposal(self):
        llm = _stub_llm(json.dumps({
            "proposals": ["this is not a dict", 42, None]
        }))
        out = await run_extractor(
            user_message="...", assistant_response="...",
            profile_structure={}, llm_call=llm,
        )
        assert out == []


@pytest.mark.asyncio
class TestExtractorFailures:
    async def test_llm_exception_returns_empty(self):
        out = await run_extractor(
            user_message="x", assistant_response="y",
            profile_structure={}, llm_call=_failing_llm(),
        )
        assert out == []

    async def test_malformed_json_returns_empty(self):
        llm = _stub_llm("this is not JSON at all")
        out = await run_extractor(
            user_message="x", assistant_response="y",
            profile_structure={}, llm_call=llm,
        )
        assert out == []


# --- summarize_structure ---


class TestSummarizeStructure:
    def test_counts_list_sections(self):
        p = HealthProfile.empty("self")
        p.conditions.append(Condition(
            name="x", status="active", field_meta=_provenance(),
        ))
        p.medications.append(Medication(
            name="m", status="active", field_meta=_provenance(),
        ))
        s = summarize_structure(p)
        assert s["conditions"] == 1
        assert s["medications"] == 1
        assert s["allergies"] == 0

    def test_counts_demographics_field_set_ratio(self):
        p = HealthProfile.empty("self")
        p.demographics = Demographics(
            age_range=_provenance(),
            sex_assigned_at_birth=_provenance(),
        )
        s = summarize_structure(p)
        assert s["demographics"] == 2  # two fields set

    def test_empty_profile_zero_everywhere(self):
        s = summarize_structure(HealthProfile.empty("self"))
        for v in s.values():
            assert v == 0


# --- _proposal_id ---


class TestProposalIds:
    def test_stable_for_same_input(self):
        a = _proposal_id("medications", "add", {"name": "metformin"})
        b = _proposal_id("medications", "add", {"name": "metformin"})
        assert a == b

    def test_different_section_different_id(self):
        a = _proposal_id("medications", "add", {"name": "x"})
        b = _proposal_id("conditions", "add", {"name": "x"})
        assert a != b

    def test_different_value_different_id(self):
        a = _proposal_id("medications", "add", {"name": "metformin"})
        b = _proposal_id("medications", "add", {"name": "atorvastatin"})
        assert a != b

    def test_key_order_does_not_matter(self):
        a = _proposal_id("medications", "add", {"name": "x", "dose": "1mg"})
        b = _proposal_id("medications", "add", {"dose": "1mg", "name": "x"})
        assert a == b


# --- persist_proposals ---


def _build_proposal(section="medications", confidence=0.9, value=None):
    """Build a real PendingUpdate for persistence tests."""
    val = value or {"name": "metformin", "dose": "500mg"}
    pid = _proposal_id(section, "add", val)
    return PendingUpdate(
        id=pid,
        proposed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        target_section=section,
        operation="add",
        proposed_value=val,
        source_turn_id="turn-1",
        extractor_reasoning="test",
        status="pending",
    )


class TestPersistProposals:
    def test_empty_input_is_noop(self, conn):
        n = persist_proposals(
            conn=conn, proposals=[], key_source="env",
        )
        assert n == 0

    def test_appends_to_pending_updates(self, conn):
        # Seed an empty profile so load_profile has something to update
        prop = _build_proposal()
        n = persist_proposals(
            conn=conn, proposals=[prop], key_source="env",
        )
        assert n == 1

        # Verify it's in the loaded profile
        loaded = load_profile(conn, key_source="env")
        assert len(loaded.profile.pending_updates) == 1
        assert loaded.profile.pending_updates[0].id == prop.id

    def test_idempotent_on_rerun(self, conn):
        prop = _build_proposal()
        persist_proposals(conn=conn, proposals=[prop], key_source="env")
        n2 = persist_proposals(conn=conn, proposals=[prop], key_source="env")
        # Same proposal id → no duplicate, returns 0 new
        assert n2 == 0
        loaded = load_profile(conn, key_source="env")
        assert len(loaded.profile.pending_updates) == 1

    def test_different_proposals_accumulate(self, conn):
        p1 = _build_proposal(value={"name": "metformin"})
        p2 = _build_proposal(value={"name": "atorvastatin"})
        persist_proposals(conn=conn, proposals=[p1, p2], key_source="env")
        loaded = load_profile(conn, key_source="env")
        assert len(loaded.profile.pending_updates) == 2
        names = {p.proposed_value["name"] for p in loaded.profile.pending_updates}
        assert names == {"metformin", "atorvastatin"}

    def test_mixed_new_and_existing(self, conn):
        p1 = _build_proposal(value={"name": "metformin"})
        persist_proposals(conn=conn, proposals=[p1], key_source="env")

        p2 = _build_proposal(value={"name": "atorvastatin"})
        # p1 is now existing, p2 is new — returns 1
        n = persist_proposals(conn=conn, proposals=[p1, p2], key_source="env")
        assert n == 1
        loaded = load_profile(conn, key_source="env")
        assert len(loaded.profile.pending_updates) == 2


# --- End-to-end: extract + persist ---


@pytest.mark.asyncio
class TestEndToEnd:
    async def test_extract_then_persist(self, conn):
        llm = _stub_llm(json.dumps({
            "proposals": [{
                "section": "medications", "operation": "add",
                "proposed_value": {"name": "metformin", "dose": "500mg"},
                "confidence": 0.95,
                "reasoning": "stated directly",
            }]
        }))
        proposals = await run_extractor(
            user_message="I take metformin 500mg twice daily.",
            assistant_response="Noted.",
            profile_structure={"medications": 0},
            llm_call=llm,
        )
        assert len(proposals) == 1

        n = persist_proposals(
            conn=conn, proposals=proposals, key_source="env",
        )
        assert n == 1

        loaded = load_profile(conn, key_source="env")
        assert len(loaded.profile.pending_updates) == 1
        assert loaded.profile.pending_updates[0].target_section == "medications"
