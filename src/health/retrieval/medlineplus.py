"""MedlinePlus evidence source via the NLM Web Service.

MedlinePlus is the National Library of Medicine's consumer-facing
health information site. The Web Service at wsearch.nlm.nih.gov
exposes a query interface that returns curated, plain-language
content — exactly the right register for "what is metformin"-style
questions where a NEJM abstract would over-shoot.

Why prefer it over PubMed for some queries: MedlinePlus distills
clinical literature into reading-grade-8 prose, which the LLM can
summarize cleanly. PubMed gives jargon-dense abstracts; MedlinePlus
gives the editorial overview written for patients. They complement —
the orchestrator's ranking puts both forward and lets the model pick.

API shape: returns XML (the JSON option is undocumented and unstable,
so we parse XML defensively). Each result has `title`, `summary`
(HTML), and `url`. We strip the HTML to a plain snippet.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from src.channels._http import make_session
from src.health.retrieval.base import Evidence, EvidenceSource

log = logging.getLogger(__name__)


_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
_DEFAULT_DB = "healthTopics"  # consumer-friendly; "drugs" exists but is shallower
_DEFAULT_RETMAX = 5
# MedlinePlus summary snippets vary wildly; cap so one verbose entry
# doesn't blow the prompt budget. The full URL is in the citation if
# the model needs more.
_MAX_SNIPPET_CHARS = 400


class MedlinePlusSource(EvidenceSource):
    """Evidence source backed by NLM MedlinePlus Web Service."""

    name = "medlineplus"
    # Authority sits between PubMed (peer-reviewed primary) and the
    # general-data sources (CDC, ClinicalTrials) — MedlinePlus is
    # editorial, vetted, and authoritative for consumer queries.
    authority_weight = 0.7
    timeout_seconds = 5.0

    def __init__(self, db: str = _DEFAULT_DB, retmax: int = _DEFAULT_RETMAX):
        self.db = db
        self.retmax = retmax

    async def search(self, query: str, topic: str | None = None) -> list[Evidence]:
        if not query or not query.strip():
            return []

        params = {
            "db": self.db,
            "term": query,
            "retmax": str(self.retmax),
        }

        async with make_session() as session:
            async with session.get(_BASE_URL, params=params) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"MedlinePlus {resp.status}: {body[:200]}"
                    )
                text = await resp.text()

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            log.warning(f"MedlinePlus XML parse failed: {e}")
            return []

        return _parse_results(root)


def _parse_results(root: ET.Element) -> list[Evidence]:
    """Walk the wsearch response tree into Evidence rows.

    Response shape (abbreviated):

      <nlmSearchResult>
        <list>
          <document rank="0" url="...">
            <content name="title">...</content>
            <content name="FullSummary">...</content>
            ...
          </document>
        </list>
      </nlmSearchResult>

    Source notes worth knowing: the `summary` field contains HTML with
    bolded query terms (`<span class="qt0">...</span>`) — we strip
    those plus all other tags. The `url` attribute on `<document>` is
    the canonical MedlinePlus topic URL.
    """
    out: list[Evidence] = []
    documents = root.findall(".//document")
    for doc in documents:
        url = (doc.attrib.get("url") or "").strip()
        title = _content_field(doc, "title")
        summary = _content_field(doc, "FullSummary") or _content_field(doc, "snippet")

        if not title or not url:
            continue

        snippet = _strip_html(summary)[:_MAX_SNIPPET_CHARS].strip()
        if not snippet:
            snippet = "(MedlinePlus summary; see URL for full content)"

        out.append(
            Evidence(
                source="medlineplus",
                title=_strip_html(title).strip(),
                snippet=snippet,
                url=url,
                # MedlinePlus pages don't carry per-page publication
                # dates we can rely on — leave as None and let the
                # ranker score them with the neutral 0.5 recency.
                published=None,
                quality_score=0.7,
                raw={"xml": ET.tostring(doc, encoding="unicode")},
            )
        )
    return out


def _content_field(doc: ET.Element, name: str) -> str:
    """Extract a `<content name="X">value</content>` field by name."""
    for child in doc.findall("content"):
        if child.attrib.get("name") == name:
            return child.text or ""
    return ""


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags + collapse whitespace.

    MedlinePlus wraps query-term hits in `<span class="qt0">` and uses
    `<p>` for paragraph breaks. A regex strip is fine here — the input
    is tightly scoped and never hostile.
    """
    no_tags = _TAG_RE.sub("", text or "")
    return _WHITESPACE_RE.sub(" ", no_tags).strip()
