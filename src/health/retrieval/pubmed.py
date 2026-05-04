"""PubMed evidence source via NCBI E-utilities.

Two-step flow:

1. esearch.fcgi — query → list of PMIDs (PubMed IDs)
2. esummary.fcgi — PMIDs → titles, journal, authors, pub date

We use esummary (not efetch) because the abstract-summary metadata is
enough for citation rendering, and esummary's JSON shape is way easier
to parse than efetch's XML. If the LLM needs the full abstract for a
specific result it can fetch via the URL.

Auth: anonymous calls are capped at ~3 req/s. Personal-use volume is
well under that, but `NCBI_API_KEY` env var (when set) raises the cap
to ~10 req/s. The key, when present, is appended as `&api_key=...`.

Rate-limiting: not implemented in Phase 1. The orchestrator's per-source
timeout is the only guard against runaway latency; a future phase can
add a token-bucket if usage scales beyond personal.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from urllib.parse import quote_plus

import aiohttp

from src.channels._http import make_session
from src.health.retrieval.base import Evidence, EvidenceSource

log = logging.getLogger(__name__)


_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH_URL = f"{_BASE_URL}/esearch.fcgi"
_ESUMMARY_URL = f"{_BASE_URL}/esummary.fcgi"

# Default page size: more is wasted in Phase 1 (the orchestrator caps
# results at 8 anyway and applies cross-source ranking on top), and
# each PMID costs an esummary lookup. 5 hits a sensible balance.
_DEFAULT_RETMAX = 5

# A handful of journals get a bump in quality_score because their
# editorial standards are exceptional even within PubMed. The list is
# deliberately short — the goal is to nudge ranking, not to legislate
# medical orthodoxy. Phase 4 may broaden this from real usage data.
_PRESTIGIOUS_JOURNALS = {
    "n engl j med",
    "lancet",
    "jama",
    "bmj",
    "ann intern med",
    "nat med",
    "cell",
    "nature",
    "science",
}


class PubMedSource(EvidenceSource):
    """Evidence source backed by NCBI's PubMed E-utilities."""

    name = "pubmed"
    # Highest authority among the Phase 1 sources — peer-reviewed
    # primary literature.
    authority_weight = 1.0
    timeout_seconds = 5.0

    def __init__(self, api_key: str = "", retmax: int = _DEFAULT_RETMAX):
        self.api_key = api_key
        self.retmax = retmax

    async def search(self, query: str, topic: str | None = None) -> list[Evidence]:
        """Search PubMed for `query`. Returns up to `retmax` Evidence."""
        if not query or not query.strip():
            return []

        async with make_session() as session:
            pmids = await self._esearch(session, query)
            if not pmids:
                return []
            summaries = await self._esummary(session, pmids)

        evidence: list[Evidence] = []
        for pmid in pmids:
            summary = summaries.get(pmid)
            if summary is None:
                continue
            ev = _summary_to_evidence(pmid, summary)
            if ev is not None:
                evidence.append(ev)
        return evidence

    async def _esearch(self, session: aiohttp.ClientSession, query: str) -> list[str]:
        """Call esearch.fcgi to turn the query into a list of PMIDs."""
        params: dict[str, str] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(self.retmax),
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with session.get(_ESEARCH_URL, params=params) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"PubMed esearch {resp.status}: {body[:200]}")
            data = await resp.json()

        return list(data.get("esearchresult", {}).get("idlist", []))

    async def _esummary(
        self, session: aiohttp.ClientSession, pmids: list[str]
    ) -> dict[str, dict]:
        """Fetch summaries for a batch of PMIDs. Returns pmid -> summary dict."""
        params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with session.get(_ESUMMARY_URL, params=params) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"PubMed esummary {resp.status}: {body[:200]}")
            data = await resp.json()

        # esummary's JSON shape: {"result": {"uids": [...], "<pmid>": {...}}}
        result = data.get("result") or {}
        return {k: v for k, v in result.items() if k != "uids" and isinstance(v, dict)}


def _summary_to_evidence(pmid: str, summary: dict) -> Evidence | None:
    """Project an esummary record into an Evidence row.

    Returns None for malformed records (no title) — better to drop one
    citation than to surface a blank entry to the LLM.
    """
    title = (summary.get("title") or "").strip()
    if not title:
        return None

    journal = (summary.get("fulljournalname") or summary.get("source") or "").strip()
    pub_date = _parse_pub_date(summary.get("pubdate") or summary.get("epubdate") or "")

    # Authors — first author + "et al." when there are several.
    authors_raw = summary.get("authors") or []
    author_names = [
        (a.get("name") or "").strip()
        for a in authors_raw
        if isinstance(a, dict) and a.get("name")
    ]
    authors = _format_authors(author_names)

    snippet_parts = []
    if authors:
        snippet_parts.append(authors)
    if journal:
        snippet_parts.append(journal)
    if pub_date:
        snippet_parts.append(pub_date.isoformat())
    snippet = " — ".join(snippet_parts) if snippet_parts else "(no source metadata)"

    return Evidence(
        source="pubmed",
        title=title,
        snippet=snippet,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        published=pub_date,
        quality_score=_journal_quality_score(journal),
        raw=summary,
    )


def _format_authors(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{names[0]} et al."


def _journal_quality_score(journal: str) -> float:
    """Map journal name to a 0..1 quality_score.

    PubMed's blanket authority_weight already covers "this is
    peer-reviewed primary literature." This adds a small additional
    bump for prestigious venues, capped so it never dominates the
    ranking — authority + recency are still the load-bearing signals.
    """
    if not journal:
        return 0.5
    norm = journal.strip().lower().rstrip(".")
    if any(p in norm for p in _PRESTIGIOUS_JOURNALS):
        return 0.9
    return 0.6


def _parse_pub_date(s: str) -> date | None:
    """Parse PubMed's pubdate strings into `date`.

    PubMed pubdates are unhelpfully varied: "2024", "2024 Mar", "2024
    Mar 15", sometimes seasonal ("2024 Spring"). We accept the common
    shapes and return None when the format isn't predictable enough to
    risk a misparse — the ranker handles `None` cleanly with the
    neutral 0.5 recency score.
    """
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split()
    try:
        year = int(parts[0])
    except (ValueError, IndexError):
        return None

    if len(parts) == 1:
        return date(year, 1, 1)

    try:
        month = datetime.strptime(parts[1][:3], "%b").month
    except ValueError:
        return date(year, 1, 1)

    if len(parts) >= 3:
        try:
            day = int(parts[2])
            return date(year, month, day)
        except ValueError:
            return date(year, month, 1)
    return date(year, month, 1)
