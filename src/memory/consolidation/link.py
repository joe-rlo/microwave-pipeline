"""Link stage — new facts + existing graph → edges + contradictions.

Reads the facts the Extract stage just inserted (or a caller-supplied
set), plus the existing fact graph, and asks Sonnet to identify
relationships between them. The model returns a list of typed edges
and a list of flagged contradictions; we persist both.

Why Sonnet (not Haiku): multi-fact reasoning across a graph needs more
working memory than Haiku reliably handles. The discipline draft is
explicit about this — "Sonnet for this stage because multi-fact
reasoning is the whole point; Haiku misses cross-references."

Contradictions are NOT auto-resolved. They land in
`pending_contradictions` with status='pending'. The CLI (Phase F.3)
lets the user resolve them — accept A, accept B, keep both, or
dismiss. The discipline framing: silence is not consent, so the
system doesn't decide for the user.

Existing-facts cap: for personal-use volume the entire graph fits in
context. We cap at MAX_EXISTING_FACTS to avoid runaway token cost
once the graph grows. When the cap is hit, we feed only the most
recent facts (rolling window). A future version can pre-filter by
embedding similarity if this becomes a real bottleneck.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable, Iterable, Iterator

import apsw

from src.memory.consolidation.schema import (
    EDGE_RELATIONS,
    ExtractedFact,
    FactEdge,
)

log = logging.getLogger(__name__)


# Personal-use volume rarely exceeds this; once it does, we feed the
# rolling window of newest facts so the model still has context.
MAX_EXISTING_FACTS = 500


LINK_SCHEMA_HINT = """{
  "edges": [
    {
      "src_id": "<fact id from inputs>",
      "dst_id": "<fact id from inputs>",
      "relation": "relates_to" | "supersedes" | "contradicts" | "follows_from",
      "weight": <float 0.0-1.0>
    }
  ],
  "contradictions": [
    {
      "fact_a_id": "<fact id>",
      "fact_b_id": "<fact id>",
      "explanation": "<one-sentence plain-language why these conflict>"
    }
  ]
}"""


LINK_SYSTEM_PROMPT = """\
You are the Link stage of a memory consolidation pipeline. You read a
set of newly-extracted facts and a set of existing facts, and you
identify relationships between them.

For each related pair, output an edge with one of these relations:
- relates_to: facts share a topic but neither subsumes the other
- supersedes: the new fact replaces or updates an older one
  (e.g., "Joe prefers Sonnet" supersedes "Joe prefers Haiku")
- contradicts: the facts directly conflict — both can't be true
- follows_from: one fact is a logical consequence of another

For contradictions specifically, also output a contradiction record
with a one-sentence explanation the user will read. Do NOT
auto-decide which fact is correct; flag it for review.

Rules:
- Only produce edges between facts in the provided inputs. Don't
  invent IDs.
- Weight reflects how strong the relationship is (1.0 = certain,
  0.5 = plausible).
- It's fine to return zero edges. Most fact pairs ARE unrelated.
- Don't relate two facts of type "person" unless they reference the
  same person.
- Don't relate a fact to itself.

Return ONLY valid JSON matching this schema:
""" + LINK_SCHEMA_HINT


LLMCall = Callable[[str, str], Awaitable[str]]


async def run_link(
    *,
    conn: apsw.Connection,
    new_facts: list[ExtractedFact],
    llm_call: LLMCall,
    max_existing_facts: int = MAX_EXISTING_FACTS,
    now: int | None = None,
) -> tuple[list[FactEdge], list[int]]:
    """Run the Link stage.

    Returns `(edges_written, contradiction_ids)`. `contradiction_ids`
    is the list of `pending_contradictions.id` values for any
    contradictions written this run — the CLI uses these to surface
    pending review items.

    `new_facts` is what the Extract stage just produced. The Link
    stage compares them against existing facts in the database, NOT
    against each other only — supersedes/contradicts most often
    happen between new and existing entries.
    """
    if not new_facts:
        log.info("Link: no new facts; skipping")
        return [], []

    now_ts = now if now is not None else int(time.time())

    existing = _load_existing_facts(conn, limit=max_existing_facts)
    new_ids = {f.id for f in new_facts}
    # Combined ID set used for validating LLM-returned edges
    valid_ids = new_ids | {f.id for f in existing}

    user_message = _compose_user_message(new_facts, existing)
    if not user_message.strip():
        return [], []

    raw = ""
    try:
        raw = await llm_call(LINK_SYSTEM_PROMPT, user_message)
    except Exception as e:
        log.warning("Link LLM call failed: %s", e)
        return [], []

    data = _parse_link_response(raw)
    if data is None:
        log.warning("Link: malformed JSON")
        return [], []

    edges_written: list[FactEdge] = []
    for raw_edge in data.get("edges", []):
        edge = _validate_edge(raw_edge, valid_ids=valid_ids, now_ts=now_ts)
        if edge is None:
            continue
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO fact_edges
                    (src_id, dst_id, relation, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (edge.src_id, edge.dst_id, edge.relation, edge.weight, edge.created_at),
            )
        except Exception as e:
            log.warning("Link: edge insert failed: %s", e)
            continue
        edges_written.append(edge)

        # When the model marks an edge as `supersedes`, set the
        # destination's `superseded_by` pointer so callers can find the
        # active version. We don't delete superseded facts — provenance
        # is preserved; queries can filter by `superseded_by IS NULL`.
        if edge.relation == "supersedes":
            try:
                conn.execute(
                    "UPDATE consolidated_facts SET superseded_by = ? WHERE id = ?",
                    (edge.src_id, edge.dst_id),
                )
            except Exception as e:
                log.warning("Link: superseded_by update failed: %s", e)

    contradiction_ids: list[int] = []
    for raw_c in data.get("contradictions", []):
        cid = _persist_contradiction(conn, raw_c, valid_ids=valid_ids, now_ts=now_ts)
        if cid is not None:
            contradiction_ids.append(cid)

    log.info(
        "Link: %d edges written, %d contradictions flagged",
        len(edges_written), len(contradiction_ids),
    )
    return edges_written, contradiction_ids


# --- Inputs -----------------------------------------------------------------


def _load_existing_facts(
    conn: apsw.Connection, *, limit: int,
) -> list[ExtractedFact]:
    """Pull the most recent existing facts to use as context for Link.

    Excludes facts already superseded (no point linking to a stale
    version when a current one exists).
    """
    rows = list(conn.execute(
        """
        SELECT id, extracted_at, fact_type, content, confidence,
               source_note, source_excerpt, superseded_by
        FROM consolidated_facts
        WHERE superseded_by IS NULL
        ORDER BY extracted_at DESC
        LIMIT ?
        """,
        (limit,),
    ))
    return [
        ExtractedFact(
            id=r["id"],
            extracted_at=int(r["extracted_at"]),
            fact_type=r["fact_type"],
            content=r["content"],
            confidence=float(r["confidence"]),
            source_note=r["source_note"],
            source_excerpt=r["source_excerpt"],
            superseded_by=r["superseded_by"],
        )
        for r in rows
    ]


def _compose_user_message(
    new_facts: list[ExtractedFact],
    existing_facts: list[ExtractedFact],
) -> str:
    """Build the message the model sees. Keeps the new vs. existing
    split visible — most cross-references happen between the two sets."""
    parts: list[str] = ["[New facts to link]"]
    for f in new_facts:
        parts.append(f"- id={f.id} type={f.fact_type} :: {f.content}")
    parts.append("")
    parts.append("[Existing facts (rolling window)]")
    for f in existing_facts:
        parts.append(f"- id={f.id} type={f.fact_type} :: {f.content}")
    parts.append("")
    parts.append(
        "Identify edges (relates_to / supersedes / contradicts / "
        "follows_from) and contradictions. Return JSON per the schema."
    )
    return "\n".join(parts)


# --- LLM response parsing ---------------------------------------------------


def _parse_link_response(raw: str) -> dict | None:
    from src.pipeline.json_utils import extract_json
    return extract_json(raw)


def _validate_edge(
    raw_edge: dict, *, valid_ids: set[str], now_ts: int,
) -> FactEdge | None:
    if not isinstance(raw_edge, dict):
        return None

    src = (raw_edge.get("src_id") or "").strip()
    dst = (raw_edge.get("dst_id") or "").strip()
    rel = raw_edge.get("relation")

    if not src or not dst or src == dst:
        return None
    if src not in valid_ids or dst not in valid_ids:
        log.debug("Link: dropping edge with unknown id (%s -> %s)", src, dst)
        return None
    if rel not in EDGE_RELATIONS:
        log.debug("Link: dropping edge with bad relation %r", rel)
        return None

    try:
        weight = float(raw_edge.get("weight", 0.5))
    except (TypeError, ValueError):
        weight = 0.5
    weight = max(0.0, min(1.0, weight))

    return FactEdge(
        src_id=src,
        dst_id=dst,
        relation=rel,  # type: ignore[arg-type]
        weight=weight,
        created_at=now_ts,
    )


def _persist_contradiction(
    conn: apsw.Connection,
    raw_c: dict,
    *,
    valid_ids: set[str],
    now_ts: int,
) -> int | None:
    if not isinstance(raw_c, dict):
        return None
    a = (raw_c.get("fact_a_id") or "").strip()
    b = (raw_c.get("fact_b_id") or "").strip()
    explanation = (raw_c.get("explanation") or "").strip()
    if not a or not b or a == b:
        return None
    if a not in valid_ids or b not in valid_ids:
        log.debug("Link: dropping contradiction with unknown id (%s, %s)", a, b)
        return None
    if not explanation:
        explanation = "Facts conflict; details not provided by extractor."

    try:
        conn.execute(
            """
            INSERT INTO pending_contradictions
                (detected_at, fact_a_id, fact_b_id, explanation, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (now_ts, a, b, explanation),
        )
    except Exception as e:
        log.warning("Link: contradiction insert failed: %s", e)
        return None
    return conn.last_insert_rowid()
