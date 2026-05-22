"""Tests for the memory consolidation pipeline (Phase F.2).

We stub the LLM callable rather than hitting NEAR — the stages are
about persistence + JSON parsing + pipeline shape, not about Haiku/
Sonnet behavior. Real-model tests live outside CI.

Coverage:
- Schema: init_tables idempotent, all three tables created
- Extract: happy path, idempotency on re-run, malformed/low-confidence
  filtering, empty inputs, LLM failure swallowed
- Link: edges persisted, supersedes updates the dst's superseded_by
  pointer, contradictions go to the pending queue, unknown IDs dropped,
  empty new_facts is a no-op
- Brief: writes file atomically, omits empty sections, no-op on
  empty graph
- Pipeline orchestration: end-to-end with stub LLMs across all three
  stages, counts in ConsolidationResult
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import apsw
import pytest

from src.memory.consolidation import (
    ConsolidationResult,
    ExtractedFact,
    init_tables,
    run_brief,
    run_consolidation,
    run_extract,
    run_link,
)
from src.memory.consolidation.extract import _fact_id


# --- fixtures ---


@pytest.fixture
def conn() -> apsw.Connection:
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


def _stub_llm(canned_response: str):
    """Build an async (system, user) -> str callable that returns a fixed
    response, regardless of input. Captures call args for assertions."""

    captured: list[tuple[str, str]] = []

    async def call(system: str, user: str) -> str:
        captured.append((system, user))
        return canned_response

    call.captured = captured  # type: ignore[attr-defined]
    return call


def _failing_llm():
    async def call(system: str, user: str) -> str:
        raise RuntimeError("simulated LLM failure")
    return call


# --- Schema ---


class TestSchema:
    def test_init_tables_creates_all_three(self, conn):
        names = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for t in ("consolidated_facts", "fact_edges", "pending_contradictions"):
            assert t in names

    def test_init_tables_idempotent(self, conn):
        init_tables(conn)
        init_tables(conn)
        # No error, no duplicate tables (sqlite would complain on the
        # second call without IF NOT EXISTS).


# --- Extract ---


@pytest.mark.asyncio
class TestExtract:
    async def test_happy_path_persists_facts(self, conn, tmp_path):
        llm = _stub_llm(json.dumps({
            "facts": [
                {"fact_type": "decision", "content": "Use Bedrock for BAA",
                 "confidence": 0.95, "source_excerpt": "we decided..."},
                {"fact_type": "preference",
                 "content": "Joe prefers terse responses",
                 "confidence": 0.85, "source_excerpt": "don't pad..."},
            ]
        }))
        facts = await run_extract(
            conn=conn, llm_call=llm,
            recent_turns=[{"role": "user", "content": "let's use bedrock"}],
            now=1_700_000_000,
        )
        assert len(facts) == 2
        # Both rows persisted to the table
        rows = list(conn.execute("SELECT * FROM consolidated_facts"))
        assert len(rows) == 2

    async def test_idempotent_on_rerun(self, conn):
        llm = _stub_llm(json.dumps({
            "facts": [{"fact_type": "decision", "content": "X",
                       "confidence": 0.9, "source_excerpt": "..."}],
        }))
        # First run inserts; second run skips (same content → same ID)
        await run_extract(conn=conn, llm_call=llm,
                          recent_turns=[{"role": "u", "content": "x"}])
        await run_extract(conn=conn, llm_call=llm,
                          recent_turns=[{"role": "u", "content": "x"}])
        rows = list(conn.execute("SELECT COUNT(*) AS n FROM consolidated_facts"))
        assert rows[0]["n"] == 1

    async def test_drops_low_confidence(self, conn):
        llm = _stub_llm(json.dumps({
            "facts": [
                {"fact_type": "decision", "content": "Solid", "confidence": 0.9,
                 "source_excerpt": "stated directly"},
                {"fact_type": "learning", "content": "Maybe", "confidence": 0.3,
                 "source_excerpt": "implied"},
            ]
        }))
        facts = await run_extract(
            conn=conn, llm_call=llm,
            recent_turns=[{"role": "user", "content": "something"}],
        )
        assert len(facts) == 1
        assert facts[0].content == "Solid"

    async def test_drops_unknown_type(self, conn):
        llm = _stub_llm(json.dumps({
            "facts": [
                {"fact_type": "bogus", "content": "X", "confidence": 0.9,
                 "source_excerpt": "..."},
            ]
        }))
        facts = await run_extract(
            conn=conn, llm_call=llm,
            recent_turns=[{"role": "user", "content": "x"}],
        )
        assert facts == []

    async def test_empty_inputs_skips_llm_call(self, conn):
        llm = _stub_llm("should not be called")
        facts = await run_extract(conn=conn, llm_call=llm)
        assert facts == []
        # Confirm the LLM was never called
        assert len(llm.captured) == 0  # type: ignore[attr-defined]

    async def test_llm_failure_returns_empty(self, conn):
        facts = await run_extract(
            conn=conn, llm_call=_failing_llm(),
            recent_turns=[{"role": "u", "content": "x"}],
        )
        assert facts == []

    async def test_malformed_json_returns_empty(self, conn):
        llm = _stub_llm("this is not json at all")
        facts = await run_extract(
            conn=conn, llm_call=llm,
            recent_turns=[{"role": "u", "content": "x"}],
        )
        assert facts == []

    async def test_reads_daily_notes_within_window(self, conn, tmp_path):
        # Recent note: included
        recent = tmp_path / "2026-05-22.md"
        recent.write_text("today's note content")
        # Old note: predates the lookback window — skipped
        old = tmp_path / "old.md"
        old.write_text("old note content")
        # backdate the old file's mtime to a week ago
        import os
        old_mtime = time.time() - 7 * 86400
        os.utime(old, (old_mtime, old_mtime))

        captured_user_messages: list[str] = []

        async def llm(system: str, user: str) -> str:
            captured_user_messages.append(user)
            return json.dumps({"facts": []})

        await run_extract(
            conn=conn,
            llm_call=llm,
            daily_notes_dir=tmp_path,
            lookback_hours=24,
            now=int(time.time()),
        )
        msg = captured_user_messages[0]
        assert "today's note content" in msg
        assert "old note content" not in msg


# --- Link ---


def _make_fact(content: str, fact_type: str = "decision",
               confidence: float = 0.9) -> ExtractedFact:
    return ExtractedFact(
        id=_fact_id(fact_type, content),  # type: ignore[arg-type]
        extracted_at=1_700_000_000,
        fact_type=fact_type,  # type: ignore[arg-type]
        content=content,
        confidence=confidence,
        source_note="",
        source_excerpt="",
        superseded_by=None,
    )


@pytest.mark.asyncio
class TestLink:
    async def test_empty_new_facts_is_noop(self, conn):
        llm = _stub_llm("should not be called")
        edges, contras = await run_link(
            conn=conn, new_facts=[], llm_call=llm,
        )
        assert edges == []
        assert contras == []
        assert len(llm.captured) == 0  # type: ignore[attr-defined]

    async def test_edges_persisted(self, conn):
        # Set up two existing facts in the DB
        old = _make_fact("Use Anthropic API directly")
        new = _make_fact("Use Bedrock for BAA path", fact_type="decision")
        for f in (old, new):
            conn.execute(
                "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
                "content, confidence, source_note, source_excerpt) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f.id, f.extracted_at, f.fact_type, f.content,
                 f.confidence, f.source_note, f.source_excerpt),
            )

        llm = _stub_llm(json.dumps({
            "edges": [{
                "src_id": new.id, "dst_id": old.id,
                "relation": "supersedes", "weight": 0.95,
            }],
            "contradictions": [],
        }))
        edges, contras = await run_link(
            conn=conn, new_facts=[new], llm_call=llm,
        )
        assert len(edges) == 1
        assert edges[0].relation == "supersedes"
        # Persisted to fact_edges
        rows = list(conn.execute("SELECT * FROM fact_edges"))
        assert len(rows) == 1
        # superseded_by pointer was updated on the older fact
        old_row = list(conn.execute(
            "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
            (old.id,),
        ))[0]
        assert old_row["superseded_by"] == new.id

    async def test_contradictions_go_to_pending_queue(self, conn):
        a = _make_fact("Joe prefers Haiku for triage", fact_type="preference")
        b = _make_fact("Joe prefers Sonnet for triage", fact_type="preference")
        for f in (a, b):
            conn.execute(
                "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
                "content, confidence, source_note, source_excerpt) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f.id, f.extracted_at, f.fact_type, f.content,
                 f.confidence, f.source_note, f.source_excerpt),
            )

        llm = _stub_llm(json.dumps({
            "edges": [],
            "contradictions": [{
                "fact_a_id": a.id, "fact_b_id": b.id,
                "explanation": "Conflicting model preference for triage.",
            }],
        }))
        edges, contras = await run_link(
            conn=conn, new_facts=[b], llm_call=llm,
        )
        assert len(contras) == 1
        # Pending — explicitly NOT auto-resolved
        row = list(conn.execute(
            "SELECT status, explanation FROM pending_contradictions"
        ))[0]
        assert row["status"] == "pending"
        assert "model preference" in row["explanation"]

    async def test_unknown_id_dropped(self, conn):
        f = _make_fact("Real fact")
        conn.execute(
            "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
            "content, confidence, source_note, source_excerpt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.id, f.extracted_at, f.fact_type, f.content,
             f.confidence, f.source_note, f.source_excerpt),
        )
        llm = _stub_llm(json.dumps({
            "edges": [{
                "src_id": f.id, "dst_id": "fact_does_not_exist",
                "relation": "relates_to", "weight": 0.5,
            }],
            "contradictions": [],
        }))
        edges, _ = await run_link(
            conn=conn, new_facts=[f], llm_call=llm,
        )
        assert edges == []
        rows = list(conn.execute("SELECT COUNT(*) AS n FROM fact_edges"))
        assert rows[0]["n"] == 0

    async def test_self_edge_dropped(self, conn):
        f = _make_fact("Solo fact")
        conn.execute(
            "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
            "content, confidence, source_note, source_excerpt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.id, f.extracted_at, f.fact_type, f.content,
             f.confidence, f.source_note, f.source_excerpt),
        )
        llm = _stub_llm(json.dumps({
            "edges": [{
                "src_id": f.id, "dst_id": f.id,
                "relation": "relates_to", "weight": 0.5,
            }],
            "contradictions": [],
        }))
        edges, _ = await run_link(
            conn=conn, new_facts=[f], llm_call=llm,
        )
        assert edges == []


# --- Brief ---


@pytest.mark.asyncio
class TestBrief:
    async def test_writes_file_when_facts_exist(self, conn, tmp_path):
        # Seed a project_state fact so brief has content
        f = _make_fact("microwave-os: Phase F.2 in progress",
                       fact_type="project_state")
        conn.execute(
            "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
            "content, confidence, source_note, source_excerpt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.id, int(time.time()), f.fact_type, f.content,
             f.confidence, f.source_note, f.source_excerpt),
        )
        # Also need a breadcrumbs table to exist or load_recent_breadcrumbs
        # crashes — install it via the breadcrumb module's helper.
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)

        llm = _stub_llm("# Today\n\n## Active projects\n- microwave-os: F.2\n")
        out = tmp_path / "BRIEFING.md"
        result = await run_brief(
            conn=conn, llm_call=llm, output_path=out,
            now=int(time.time()),
        )
        assert result == out
        assert out.exists()
        content = out.read_text()
        assert "microwave-os" in content

    async def test_no_facts_means_no_write(self, conn, tmp_path):
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)
        llm = _stub_llm("should not be called")
        out = tmp_path / "BRIEFING.md"
        result = await run_brief(
            conn=conn, llm_call=llm, output_path=out,
            now=int(time.time()),
        )
        assert result is None
        assert not out.exists()
        assert len(llm.captured) == 0  # type: ignore[attr-defined]

    async def test_atomic_write(self, conn, tmp_path):
        """Brief writes through a .tmp file then renames — a write error
        mid-flight shouldn't leave a half-baked BRIEFING.md."""
        f = _make_fact("p", fact_type="project_state")
        conn.execute(
            "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
            "content, confidence, source_note, source_excerpt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.id, int(time.time()), f.fact_type, f.content,
             f.confidence, f.source_note, f.source_excerpt),
        )
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)

        llm = _stub_llm("brief content")
        out = tmp_path / "BRIEFING.md"
        await run_brief(
            conn=conn, llm_call=llm, output_path=out,
            now=int(time.time()),
        )
        # No stray .tmp file left behind
        assert not (tmp_path / "BRIEFING.md.tmp").exists()
        assert out.read_text() == "brief content\n"

    async def test_llm_failure_leaves_no_file(self, conn, tmp_path):
        f = _make_fact("p", fact_type="project_state")
        conn.execute(
            "INSERT INTO consolidated_facts (id, extracted_at, fact_type, "
            "content, confidence, source_note, source_excerpt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.id, int(time.time()), f.fact_type, f.content,
             f.confidence, f.source_note, f.source_excerpt),
        )
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)

        out = tmp_path / "BRIEFING.md"
        result = await run_brief(
            conn=conn, llm_call=_failing_llm(), output_path=out,
            now=int(time.time()),
        )
        assert result is None
        assert not out.exists()


# --- Pipeline ---


@pytest.mark.asyncio
class TestPipeline:
    async def test_end_to_end_with_stubs(self, conn, tmp_path, monkeypatch):
        # Provision env so the selector's NEAR-or-legacy choice doesn't
        # try to construct real clients; the test patches the callable
        # builder anyway.
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)
        # Seed a turn so Extract has input
        conn.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO turns (role, content, created_at) VALUES (?, ?, ?)",
            ("user", "let's use Bedrock for the BAA path", "2026-05-22"),
        )

        # Patch get_stage_callable so each stage gets its own stub.
        from src.memory.consolidation import pipeline as pipe_mod
        from src.memory.consolidation.pipeline import (
            STAGE_BRIEF, STAGE_EXTRACT, STAGE_LINK,
        )
        canned = {
            STAGE_EXTRACT: json.dumps({
                "facts": [{
                    "fact_type": "decision",
                    "content": "Use Bedrock for BAA",
                    "confidence": 0.95,
                    "source_excerpt": "let's use Bedrock",
                }]
            }),
            STAGE_LINK: json.dumps({"edges": [], "contradictions": []}),
            STAGE_BRIEF: "# Today\n\n## Recent decisions\n- Bedrock for BAA\n",
        }

        def fake_get_stage_callable(stage, **kw):
            response = canned[stage]
            async def call(s, u):
                return response
            return call

        monkeypatch.setattr(
            "src.llm.selector.get_stage_callable", fake_get_stage_callable
        )

        class _C:
            auth_mode = "max"
            anthropic_api_key = ""
            cli_path = ""
            workspace_dir = tmp_path

        out_path = tmp_path / "BRIEFING.md"
        result = await run_consolidation(
            conn=conn,
            config=_C(),
            briefing_path=out_path,
            now=int(time.time()),
        )

        assert isinstance(result, ConsolidationResult)
        assert result.new_facts == 1
        assert result.edges == 0
        assert result.contradictions == 0
        assert result.briefing_path == str(out_path)
        assert out_path.exists()
        assert result.duration_ms >= 0

    async def test_skips_brief_when_no_path_given(self, conn, monkeypatch, tmp_path):
        from src.memory.breadcrumbs import init_tables as init_bc
        init_bc(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TEXT)"
        )

        def fake_get_stage_callable(stage, **kw):
            async def call(s, u):
                return json.dumps({"facts": []})
            return call

        monkeypatch.setattr(
            "src.llm.selector.get_stage_callable", fake_get_stage_callable
        )

        class _C:
            auth_mode = "max"
            anthropic_api_key = ""
            cli_path = ""
            workspace_dir = tmp_path

        result = await run_consolidation(
            conn=conn,
            config=_C(),
            briefing_path=None,
            now=int(time.time()),
        )
        assert result.briefing_path is None
