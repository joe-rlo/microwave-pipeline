"""Pipeline 3.4 — embedding-similarity contradiction queue (Phase A).

Companion to `memory.health` (LLM-based detection): a fast,
deterministic pass that flags fragment pairs whose embeddings are
similar enough to be candidates for supersession review. The output
lands in `workspace/memory/contradictions.md` as a triage queue —
the user reviews each pair and decides keep / supersede / merge.

By design this only flags. Auto-supersession is the deferred Phase B,
gated behind `MEMORY_AUTO_SUPERSEDE=true`, and only enabled after
Phase A produces calibration data on real corpora.

Sovereignty over memory is the design value (same as `memory.health`):
the agent flags, the user resolves. Even a 0.95-similar pair might be
two legitimate, context-dependent facts — only the user knows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.memory.index import MemoryIndex, _deserialize_vector

log = logging.getLogger(__name__)


# Phase A threshold from the resolved Q4 in the spec. Calibrated by
# eye for "looks suspiciously similar" rather than ground-truth data —
# the whole point of Phase A is to *produce* that ground-truth data.
# Phase B's stricter 0.95 auto-supersede threshold will be tuned from
# queue results once we have real flagged pairs.
DEFAULT_CONTRADICTION_THRESHOLD = 0.80


@dataclass
class SimilarFragmentPair:
    """Two fragments whose embeddings sit above the similarity threshold.

    Sorted output convention: `a` is the older fragment (lower id),
    `b` is the newer. That lets queue readers (and the future
    `/memory review` triage UX) default-suggest "supersede a with b"
    without re-checking timestamps.
    """
    id_a: int
    id_b: int
    content_a: str
    content_b: str
    source_a: str
    source_b: str
    similarity: float


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two same-length float vectors.

    Hand-rolled to avoid pulling numpy as a dependency for one
    function. Vectors from OpenAI embeddings are unit-normalized in
    practice, so a dot product is usually enough — but normalize
    explicitly here so callers don't have to care about embedding
    provider invariants.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def find_similar_pairs(
    index: MemoryIndex,
    source_filter: str | None = None,
    threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
) -> list[SimilarFragmentPair]:
    """Pairwise cosine scan; return pairs above `threshold`, newest first.

    `source_filter` is a SQL LIKE pattern against `fragments.source`.
    Pass `"%MEMORY.md"` to scan MEMORY.md fragments only — the typical
    Phase A use case. Pass `None` to scan everything (slower, more
    false positives across daily-notes / project files).

    Returns up to one entry per (id_a, id_b) pair with id_a < id_b so
    duplicates aren't surfaced; sorted by similarity descending so the
    most suspicious pairs appear first in the queue.
    """
    if not index.conn or not index._has_vec:
        log.warning("Contradiction scan needs vec extension; skipping")
        return []

    query = (
        "SELECT f.id, f.content, f.source, f.timestamp, v.embedding "
        "FROM fragments f JOIN fragments_vec v ON f.id = v.id"
    )
    params: tuple = ()
    if source_filter:
        query += " WHERE f.source LIKE ?"
        params = (source_filter,)
    query += " ORDER BY f.id"

    rows = list(index.conn.execute(query, params))
    if len(rows) < 2:
        return []

    # Deserialize once, then n² compare. n is small (memory file
    # rarely exceeds a few hundred fragments), so the quadratic
    # cost stays well under the LLM-based detector's latency.
    frags = []
    for row in rows:
        try:
            vec = _deserialize_vector(row["embedding"])
        except Exception as e:
            log.debug(f"Could not deserialize vector for id={row['id']}: {e}")
            continue
        frags.append({
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "vec": vec,
        })

    pairs: list[SimilarFragmentPair] = []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            sim = _cosine(frags[i]["vec"], frags[j]["vec"])
            if sim < threshold:
                continue
            # Skip exact-duplicate text — that's an indexing artifact
            # (e.g. MEMORY.md re-indexed without delete-source), not
            # a contradiction. Queue noise we don't want the user to
            # triage every week.
            if frags[i]["content"].strip() == frags[j]["content"].strip():
                continue
            pairs.append(SimilarFragmentPair(
                id_a=frags[i]["id"],
                id_b=frags[j]["id"],
                content_a=frags[i]["content"],
                content_b=frags[j]["content"],
                source_a=frags[i]["source"],
                source_b=frags[j]["source"],
                similarity=sim,
            ))

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


def render_queue(pairs: list[SimilarFragmentPair]) -> str:
    """Render flagged pairs as the `contradictions.md` queue file body.

    Format is markdown so the user can read it directly without a
    custom viewer. Each pair becomes a heading + the two contents
    quoted as blockquotes + a similarity score line. A trailing
    "how to resolve" footer explains the (still-manual) next steps
    so users running `memory scan` for the first time know what to
    do with the output.
    """
    if not pairs:
        return (
            "# Contradiction queue\n\n"
            f"_Scanned at {datetime.now().isoformat(timespec='minutes')}_\n\n"
            "✓ No fragment pairs above the similarity threshold. "
            "MEMORY.md looks clean.\n"
        )

    lines = [
        "# Contradiction queue",
        "",
        f"_Scanned at {datetime.now().isoformat(timespec='minutes')}_",
        "",
        f"{len(pairs)} pair{'s' if len(pairs) != 1 else ''} flagged at "
        f"similarity ≥ {DEFAULT_CONTRADICTION_THRESHOLD:.2f}. ",
        "Triage each entry below: keep both (different referents / time "
        "periods), supersede one with the other, or merge into a single "
        "clearer line. Resolution is manual — edit MEMORY.md directly.",
        "",
    ]
    for i, p in enumerate(pairs, 1):
        lines.append(f"## {i}. Similarity {p.similarity:.3f}")
        lines.append("")
        lines.append(f"**A** (fragment #{p.id_a}):")
        for ln in p.content_a.splitlines() or [""]:
            lines.append(f"> {ln}")
        lines.append("")
        lines.append(f"**B** (fragment #{p.id_b}):")
        for ln in p.content_b.splitlines() or [""]:
            lines.append(f"> {ln}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_queue(
    pairs: list[SimilarFragmentPair],
    queue_path: Path,
) -> Path:
    """Persist the queue to `queue_path` (typically `workspace/memory/contradictions.md`).

    Overwrites on each scan so the queue is always current — there's
    no append behavior, because last-week's flagged pair that's still
    flagged should appear in this week's queue too. Callers that need
    history should use `supersession-log.md` (Phase B feature, future).
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(render_queue(pairs), encoding="utf-8")
    return queue_path
