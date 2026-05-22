"""Schema + shared types for the consolidation pipeline (Phase F.2).

Three tables, all additive — no changes to existing schema:

- `consolidated_facts` — structured facts extracted from daily notes
  and conversation transcripts. The Extract stage writes here.
- `fact_edges` — directed relationships between facts. The Link stage
  writes here (relates_to / supersedes / contradicts / follows_from).
- `pending_contradictions` — flagged conflicts awaiting user review.
  Link writes here when it detects contradictions; the user resolves
  via the CLI (Phase F.3). Resolution does NOT auto-fire — silence is
  not consent, per the discipline-paper framing.

Why dataclasses (not Pydantic models): the rest of `src/memory/` uses
plain dataclasses or apsw rows. Consistency wins; we're not doing
extensive validation here (the JSON shapes are validated when parsed
out of LLM responses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import apsw


FactType = Literal[
    "decision",        # "we decided to use Bedrock for BAA"
    "preference",      # "Joe prefers terse responses"
    "commitment",      # "ship Phase B by 2026-06-15"
    "learning",        # "Bedrock streams events differently from SSE"
    "person",          # "Sarah from Acme Corp, met 2026-05-10"
    "project_state",   # "microwave-os: Phase A complete"
]
FACT_TYPES = ("decision", "preference", "commitment", "learning", "person", "project_state")


EdgeRelation = Literal[
    "relates_to",      # facts share a topic but neither subsumes the other
    "supersedes",      # new fact overrides an older one
    "contradicts",     # facts directly conflict — user resolution required
    "follows_from",    # one fact is a logical consequence of another
]
EDGE_RELATIONS = ("relates_to", "supersedes", "contradicts", "follows_from")


ContradictionStatus = Literal[
    "pending", "accepted_a", "accepted_b", "both_kept", "dismissed",
]


@dataclass(frozen=True)
class ExtractedFact:
    """One structured fact extracted from a turn or note.

    `id` is set by the database on insert; pre-insert it's "".
    `extracted_at` and `confidence` originate from the Extract stage's
    LLM response. `source_note` and `source_excerpt` provide
    provenance — every fact must be traceable to its origin.
    """

    id: str
    extracted_at: int               # epoch seconds
    fact_type: FactType
    content: str
    confidence: float               # 0–1
    source_note: str                # path or session-id reference
    source_excerpt: str             # the sentences the fact came from
    superseded_by: str | None       # set by Link when a newer fact replaces this


@dataclass(frozen=True)
class FactEdge:
    """One edge in the knowledge graph between two facts."""

    src_id: str
    dst_id: str
    relation: EdgeRelation
    weight: float                   # 0–1, model's confidence in the relation
    created_at: int


@dataclass(frozen=True)
class PendingContradiction:
    """A contradiction the Link stage flagged for user review."""

    id: int
    detected_at: int
    fact_a_id: str
    fact_b_id: str
    explanation: str                # short, plain-language reason
    status: ContradictionStatus


def init_tables(conn: apsw.Connection) -> None:
    """Create the three consolidation tables. Idempotent.

    Wired into the orchestrator's startup path next to the breadcrumb
    init so all Phase-F tables come up in one place.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidated_facts (
            id TEXT PRIMARY KEY,
            extracted_at INTEGER NOT NULL,
            fact_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_note TEXT NOT NULL,
            source_excerpt TEXT NOT NULL,
            superseded_by TEXT REFERENCES consolidated_facts(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_type_extracted
        ON consolidated_facts(fact_type, extracted_at DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_edges (
            src_id TEXT NOT NULL REFERENCES consolidated_facts(id),
            dst_id TEXT NOT NULL REFERENCES consolidated_facts(id),
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (src_id, dst_id, relation)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edges_src ON fact_edges(src_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON fact_edges(dst_id)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at INTEGER NOT NULL,
            fact_a_id TEXT NOT NULL REFERENCES consolidated_facts(id),
            fact_b_id TEXT NOT NULL REFERENCES consolidated_facts(id),
            explanation TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_contradictions_status
        ON pending_contradictions(status, detected_at DESC)
    """)
