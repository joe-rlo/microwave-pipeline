"""Hybrid retrieval: vector + BM25 keyword search, merged via RRF.

Applies temporal decay and MMR diversity per triage parameters.

Also queries the session `turns` table directly (if a SessionEngine is
wired in), so recent conversation is recoverable even when it hasn't
been promoted to MEMORY.md yet. Results from that path are marked with
`source_type="turn"` so assembly can render them under a distinct label.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from src.memory.embeddings import EmbeddingClient
from src.memory.index import MemoryIndex, _serialize_vector
from src.session.models import MemoryFragment, SearchResult, TriageResult

if TYPE_CHECKING:
    from src.session.engine import SessionEngine

log = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant
RECENT_TURN_HOURS = 48.0
RECENT_TURN_LIMIT = 4


class MemorySearcher:
    def __init__(
        self,
        index: MemoryIndex,
        embedder: EmbeddingClient,
        session_engine: "SessionEngine | None" = None,
    ):
        self.index = index
        self.embedder = embedder
        self.session_engine = session_engine

    def search(
        self,
        query: str,
        triage: TriageResult,
        max_results: int | None = None,
    ) -> SearchResult:
        """Run hybrid search shaped by triage parameters."""
        start = time.time()

        params = triage.search_params
        max_results = max_results or params.get("result_count", 5)
        decay_half_life = params.get("decay_half_life", 30.0)
        weight_recency = params.get("weight_recency", 0.5)
        mmr_lambda = params.get("mmr_lambda", 0.7)

        # Run vector and BM25 in parallel (conceptually — SQLite is single-threaded)
        vec_results = self._vector_search(query, max_results * 3)
        bm25_results = self._bm25_search(query, max_results * 3)

        # Merge via Reciprocal Rank Fusion
        merged = self._rrf_merge(vec_results, bm25_results)

        # Apply temporal decay
        now = datetime.now()
        for frag in merged:
            age_days = (now - frag.timestamp).total_seconds() / 86400
            decay = 0.5 ** (age_days / decay_half_life)
            frag.score = frag.score * (1 - weight_recency) + decay * weight_recency

        # Sort by adjusted score
        merged.sort(key=lambda f: f.score, reverse=True)

        # MMR diversity filtering
        selected = self._mmr_select(merged, max_results, mmr_lambda)

        # Increment retrieval counts (only for durable fragments, not live turns)
        for frag in selected:
            self.index.increment_retrieval(frag.id)

        # Recent-turn recall: live query the turns table so conversation from
        # the last ~48h is retrievable even before compaction indexes it.
        # These are NOT ranked against fragments — they occupy a separate slot
        # and get rendered under their own label by the assembly stage.
        turn_fragments = self._recent_turn_fragments(query)

        elapsed_ms = int((time.time() - start) * 1000)
        return SearchResult(
            fragments=selected + turn_fragments,
            strategy_used=f"hybrid(decay={decay_half_life}, recency={weight_recency})"
            + (f"+turns({len(turn_fragments)})" if turn_fragments else ""),
            search_time_ms=elapsed_ms,
        )

    def _recent_turn_fragments(self, query: str) -> list[MemoryFragment]:
        """Pull recent conversation turns matching the query. Returns them as
        MemoryFragments with source_type='turn' so assembly can label them."""
        if self.session_engine is None:
            return []
        try:
            turns = self.session_engine.search_recent_turns(
                query, hours=RECENT_TURN_HOURS, limit=RECENT_TURN_LIMIT
            )
        except Exception as e:
            log.debug(f"Recent-turn search failed: {e}")
            return []

        out: list[MemoryFragment] = []
        for t in turns:
            speaker = "User" if t.role == "user" else (
                "Summary" if t.metadata.get("type") == "compaction_summary" else "Microwave"
            )
            content = f"{speaker}: {t.content}"
            out.append(
                MemoryFragment(
                    id=-(t.id or 0),  # negative to avoid collision with fragment IDs
                    content=content,
                    source=f"session:{t.session_id}",
                    timestamp=t.timestamp,
                    score=0.0,
                    source_type="turn",
                )
            )
        return out

    def _vector_search(self, query: str, limit: int) -> list[MemoryFragment]:
        """Search by vector similarity using sqlite-vec."""
        try:
            embedding = self.embedder.embed(query)
            rows = list(self.index.conn.execute(
                "SELECT v.id, v.distance, f.content, f.source, f.timestamp, f.retrieval_count "
                "FROM fragments_vec v "
                "JOIN fragments f ON f.id = v.id "
                "WHERE v.embedding MATCH ? AND k = ?",
                (_serialize_vector(embedding), limit),
            ))

            results = []
            for row in rows:
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1.0 - row["distance"]
                results.append(
                    MemoryFragment(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        score=max(0, similarity),
                        retrieval_count=row["retrieval_count"],
                    )
                )
            return results
        except Exception as e:
            log.debug(f"Vector search unavailable: {e}")
            return []

    @staticmethod
    def _sanitize_fts5(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH syntax.

        Wraps each word in double quotes so punctuation and special
        characters (. - : / etc.) are treated as literals, not operators.
        """
        import re
        # Extract alphanumeric words, discard pure punctuation
        words = re.findall(r"[a-zA-Z0-9]+", query)
        if not words:
            return ""
        # Quote each word and join with implicit AND
        return " ".join(f'"{w}"' for w in words)

    def _bm25_search(self, query: str, limit: int) -> list[MemoryFragment]:
        """Search by BM25 keyword matching via FTS5."""
        try:
            safe_query = self._sanitize_fts5(query)
            if not safe_query:
                return []
            rows = list(self.index.conn.execute(
                "SELECT f.id, f.content, f.source, f.timestamp, f.retrieval_count, "
                "bm25(fragments_fts) AS rank "
                "FROM fragments_fts fts "
                "JOIN fragments f ON f.id = fts.rowid "
                "WHERE fragments_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (safe_query, limit),
            ))

            results = []
            for row in rows:
                # BM25 returns negative scores (lower = better), normalize
                score = 1.0 / (1.0 + abs(row["rank"]))
                results.append(
                    MemoryFragment(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        score=score,
                        retrieval_count=row["retrieval_count"],
                    )
                )
            return results
        except Exception as e:
            log.debug(f"BM25 search error: {e}")
            return []

    def _rrf_merge(
        self,
        vec_results: list[MemoryFragment],
        bm25_results: list[MemoryFragment],
    ) -> list[MemoryFragment]:
        """Merge two ranked lists via Reciprocal Rank Fusion."""
        scores: dict[int, float] = {}
        fragments: dict[int, MemoryFragment] = {}

        for rank, frag in enumerate(vec_results):
            scores[frag.id] = scores.get(frag.id, 0) + 1.0 / (RRF_K + rank + 1)
            fragments[frag.id] = frag

        for rank, frag in enumerate(bm25_results):
            scores[frag.id] = scores.get(frag.id, 0) + 1.0 / (RRF_K + rank + 1)
            if frag.id not in fragments:
                fragments[frag.id] = frag

        for fid, score in scores.items():
            fragments[fid].score = score

        return list(fragments.values())

    def _mmr_select(
        self,
        candidates: list[MemoryFragment],
        k: int,
        lambda_param: float,
    ) -> list[MemoryFragment]:
        """Maximal Marginal Relevance — diversify selected fragments.

        Penalizes candidates similar to already-selected ones.
        Uses simple text overlap as a proxy for similarity (avoids extra embedding calls).
        """
        if len(candidates) <= k:
            return candidates

        selected = [candidates[0]]
        remaining = candidates[1:]

        while len(selected) < k and remaining:
            best_score = -1.0
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance component
                relevance = candidate.score

                # Diversity component: max similarity to any selected fragment
                max_sim = max(
                    self._text_similarity(candidate.content, s.content) for s in selected
                )

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple word overlap similarity (Jaccard). Cheap proxy for cosine similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
