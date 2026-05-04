"""Evidence retrieval primitives.

`Evidence` is the lingua franca every source returns. Source-specific
fields live in `raw` so the model can reach for them when the summary
isn't enough, without bloating the per-citation prompt footprint.

`EvidenceSource` is the abstract base every concrete source implements.
Two contracts matter:

1. `search(query, topic) -> list[Evidence]` is an async coroutine. The
   orchestrator awaits all registered sources concurrently; failure of
   one source must NOT take down the whole retrieval — that's enforced
   at the orchestrator layer via `asyncio.gather(return_exceptions=True)`.
2. `name` and `authority_weight` are class-level so the orchestrator can
   identify and rank without instantiating. Authority is a coarse
   estimate of how seriously to weight this source's results when
   ranking against others (NEJM/JAMA > FDA > MedlinePlus > CDC >
   ClinicalTrials per the spec).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Evidence:
    """One retrieved citation.

    Fields are intentionally narrow — the LLM doesn't need the full
    record, just enough to summarize and cite. Source-specific extras
    (e.g., PubMed's MeSH terms, FDA's adverse event counts) live in
    `raw` for assembly to splice in when relevant.
    """
    source: str            # "pubmed", "openfda", "medlineplus", "cdc", "clinicaltrials"
    title: str
    snippet: str           # 1-3 sentence summary or excerpt — keep short
    url: str               # citable canonical link
    published: date | None = None  # None when source doesn't expose a date
    quality_score: float = 0.5     # 0.0-1.0, source-specific heuristic
    raw: dict = field(default_factory=dict)  # full response for the model


class EvidenceSource(ABC):
    """Abstract base for a single authoritative health source.

    Concrete subclasses set `name` and `authority_weight` as class attrs
    and implement `search`. The constructor takes whatever config the
    source needs (API keys, custom timeouts, base URLs) so the
    orchestrator can register pre-configured instances.
    """

    # Override in subclass — used in audit logs, citation labels, dedup.
    name: str = "unknown"

    # Coarse "trust" weight for ranking. The spec's authority order:
    # PubMed (NEJM/JAMA via abstracts) > openFDA > MedlinePlus > CDC > ClinicalTrials.
    # These numbers express that ordering; absolute values don't matter
    # as long as the ordering does. Tune from real usage in Phase 4.
    authority_weight: float = 0.5

    # Per-source timeout. Source orchestrator uses this to cap each
    # leg of the fan-out so one slow source can't stall the whole turn.
    # Override in subclass when a source is known to be slow (CDC's
    # Socrata API can be sluggish).
    timeout_seconds: float = 5.0

    @abstractmethod
    async def search(self, query: str, topic: str | None = None) -> list[Evidence]:
        """Fetch evidence for `query`, optionally filtered by `topic`.

        `topic` comes from triage's health_topic field — sources can
        use it to bias their query (e.g., adding MeSH filters for
        PubMed) or ignore it. Returning an empty list is normal when
        the source has nothing for this query; raising is reserved for
        actual failures (network, parse errors).
        """
        ...
