"""Medical / scientific literature tools — legal full-text retrieval.

Two tools live here:

- **medical_literature_search** — search the biomedical literature and
  return structured hits (title, authors, journal, year, identifiers,
  open-access flag, abstract). Default backend is **Europe PMC**, which
  indexes PubMed/MEDLINE *plus* PMC full text *plus* preprints behind one
  REST API. A **PubMed** (NCBI E-utilities) backend is also wired in for
  callers who want to hit NCBI directly — switch with `MEDICAL_SEARCH_BACKEND`.

- **medical_article_fetch** — given a DOI, PMID, or PMCID, return the
  abstract always, the *open-access* full text when one exists (Europe PMC
  fullTextXML), and a legally-posted free PDF/link resolved via **Unpaywall**
  when the publisher version is paywalled. When nothing is openly available
  it says so and hands back the publisher link — it never reaches for a
  pirate mirror.

Why this shape: the user's underlying need is cited, readable medical
literature for the health profile. Every source here is a sanctioned API
(Europe PMC and NCBI are public-funded; Unpaywall only indexes versions
the authors/publishers themselves posted legally). No Sci-Hub, no
paywall-bypassing — see the README "No vision (except PDFs)" sibling note
on staying on the legal side of content access.

No API key required for any of this. Unpaywall *requires* a contact email
as a query param (it's how they rate-identify callers); NCBI *recommends*
one. Both come from `Config.research_contact_email` (env
`RESEARCH_CONTACT_EMAIL`); a generic fallback is used if unset so the
tools still work out of the box.

Set `MEDICAL_TOOLS_DISABLED=true` to opt out entirely.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


# --- Endpoints ---

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# NCBI's BioC service returns OA full text as structured JSON passages. We
# prefer it over Europe PMC's fullTextXML endpoint, which was deprecated /
# now 404s for single-article requests (verified 2026-06 against EBI's own
# documented example). BioC is actively maintained and parses cleanly.
NCBI_BIOC_FULLTEXT = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

USER_AGENT = "MicrowaveOS-Research/0.1 (+https://0x4a6f65.com)"
# Used when the operator hasn't set RESEARCH_CONTACT_EMAIL. Valid format so
# Unpaywall accepts it; deliberately generic (not a real inbox).
FALLBACK_EMAIL = "microwaveos@0x4a6f65.com"

DEFAULT_MAX_RESULTS = 8
HARD_MAX_RESULTS = 25
DEFAULT_TIMEOUT_SECONDS = 20.0
# Full-text bodies can be book-length; clip so one fetch can't blow the
# model's context. Generous enough for a full research article.
FULLTEXT_MAX_CHARS = 40_000
ABSTRACT_MAX_CHARS = 4_000

DEFAULT_BACKEND = "europepmc"


class ResearchToolError(RuntimeError):
    """Raised on any research-tool failure that should surface to the model."""


# --- Article model ---


@dataclass(frozen=True)
class Article:
    """One literature hit. Fields are best-effort — not every source
    populates every field."""

    title: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    is_open_access: bool = False
    abstract: str = ""
    url: str = ""
    source: str = ""
    # True for preprints (medRxiv / bioRxiv / other PPR-source records).
    # Preprints are NOT peer-reviewed — kept in results but flagged so the
    # model can caveat them rather than present them as established.
    is_preprint: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "is_open_access": self.is_open_access,
            "is_preprint": self.is_preprint,
            "abstract": _clip(self.abstract, ABSTRACT_MAX_CHARS),
            "url": self.url,
            "source": self.source,
        }


# --- helpers ---


def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated, {len(text)} chars total]"


def _contact_email(config) -> str:
    return (getattr(config, "research_contact_email", "") or "").strip() or FALLBACK_EMAIL


def _make_client(client: httpx.AsyncClient | None) -> tuple[httpx.AsyncClient, bool]:
    """Return (client, owned). When the caller injects a client (tests),
    we don't close it; otherwise we build one and own its lifecycle."""
    if client is not None:
        return client, False
    return (
        httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=DEFAULT_TIMEOUT_SECONDS / 3,
                read=DEFAULT_TIMEOUT_SECONDS,
                write=5.0,
                pool=5.0,
            ),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ),
        True,
    )


def _normalize_id(identifier: str) -> tuple[str, str]:
    """Classify a raw identifier into (kind, value).

    kind ∈ {"doi", "pmcid", "pmid"}. We sniff by shape:
      - contains "/" or starts with "10." → DOI
      - starts with "PMC" → PMCID
      - all digits → PMID
    """
    s = (identifier or "").strip()
    if not s:
        raise ResearchToolError("identifier must be a non-empty string")
    low = s.lower()
    if low.startswith("doi:"):
        s = s[4:].strip()
        low = s.lower()
    if low.startswith("10.") or "/" in s:
        return "doi", s
    if low.startswith("pmc"):
        return "pmcid", s.upper()
    if s.isdigit():
        return "pmid", s
    # Fall back to treating it as a DOI — Europe PMC is forgiving.
    return "doi", s


# --- Europe PMC ---


def _epmc_article(rec: dict) -> Article:
    """Map one Europe PMC `result` record to an Article."""
    authors = []
    author_string = rec.get("authorString") or ""
    if author_string:
        authors = [a.strip() for a in author_string.split(",") if a.strip()]
    pmcid = rec.get("pmcid") or ""
    doi = rec.get("doi") or ""
    pmid = rec.get("pmid") or ""
    is_oa = (rec.get("isOpenAccess") or "").upper() == "Y" or bool(rec.get("inEPMC") == "Y")
    # Europe PMC flags preprints with source "PPR" (medRxiv/bioRxiv/etc.);
    # some also carry "preprint" in pubType. Either marks it not-peer-reviewed.
    is_preprint = (rec.get("source") or "").upper() == "PPR" or \
        "preprint" in (rec.get("pubType") or "").lower()
    if doi:
        url = f"https://doi.org/{doi}"
    elif pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid}"
    elif pmid:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    else:
        url = ""
    return Article(
        title=(rec.get("title") or "").strip().rstrip("."),
        authors=authors,
        journal=(rec.get("journalTitle") or rec.get("bookOrReportDetails", {}).get("publisher", "") or "").strip(),
        year=str(rec.get("pubYear") or ""),
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        is_open_access=is_oa,
        abstract=(rec.get("abstractText") or "").strip(),
        url=url,
        source="europepmc",
        is_preprint=is_preprint,
    )


async def _search_europepmc(
    query: str, *, max_results: int, client: httpx.AsyncClient,
) -> list[Article]:
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",  # includes abstractText
        "pageSize": str(max_results),
    }
    try:
        resp = await client.get(EUROPEPMC_SEARCH, params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        raise ResearchToolError(f"Europe PMC search failed: {e}") from e
    except ValueError as e:
        raise ResearchToolError(f"Europe PMC returned non-JSON: {e}") from e

    results = (data.get("resultList") or {}).get("result") or []
    return [_epmc_article(r) for r in results[:max_results]]


# --- PubMed (NCBI E-utilities) ---


async def _search_pubmed(
    query: str, *, max_results: int, client: httpx.AsyncClient, email: str,
) -> list[Article]:
    """esearch → idlist → esummary. Abstracts aren't in esummary, so PubMed
    hits carry metadata only; use medical_article_fetch for the abstract."""
    common = {"db": "pubmed", "retmode": "json", "tool": "MicrowaveOS", "email": email}
    try:
        r1 = await client.get(NCBI_ESEARCH, params={
            **common, "term": query, "retmax": str(max_results), "sort": "relevance",
        })
        r1.raise_for_status()
        ids = ((r1.json().get("esearchresult") or {}).get("idlist")) or []
        if not ids:
            return []
        r2 = await client.get(NCBI_ESUMMARY, params={**common, "id": ",".join(ids)})
        r2.raise_for_status()
        summary = (r2.json().get("result") or {})
    except httpx.HTTPError as e:
        raise ResearchToolError(f"PubMed search failed: {e}") from e
    except ValueError as e:
        raise ResearchToolError(f"PubMed returned non-JSON: {e}") from e

    out: list[Article] = []
    for uid in ids:
        rec = summary.get(uid)
        if not isinstance(rec, dict):
            continue
        authors = [a.get("name", "") for a in rec.get("authors", []) if a.get("name")]
        doi = ""
        for aid in rec.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
                break
        # PubMed is largely peer-reviewed, but the NIH Preprint Pilot indexes
        # some preprints — esummary lists them in pubtype. Flag if present.
        is_preprint = any(
            "preprint" in str(pt).lower() for pt in (rec.get("pubtype") or [])
        )
        out.append(Article(
            title=(rec.get("title") or "").strip().rstrip("."),
            authors=authors,
            journal=(rec.get("fulljournalname") or rec.get("source") or "").strip(),
            year=(rec.get("pubdate") or "")[:4],
            doi=doi,
            pmid=uid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            source="pubmed",
            is_preprint=is_preprint,
        ))
    return out


# --- public: search ---


def get_backend_name(name: str | None = None) -> str:
    selected = (name or os.environ.get("MEDICAL_SEARCH_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if selected not in ("europepmc", "pubmed"):
        raise ResearchToolError(
            f"Unknown MEDICAL_SEARCH_BACKEND={selected!r}. Valid: europepmc | pubmed"
        )
    return selected


async def search_literature(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    backend: str | None = None,
    email: str = FALLBACK_EMAIL,
    client: httpx.AsyncClient | None = None,
) -> list[Article]:
    if not isinstance(query, str) or not query.strip():
        raise ResearchToolError("query must be a non-empty string")
    max_results = min(max(1, int(max_results)), HARD_MAX_RESULTS)
    be = get_backend_name(backend)

    cli, owned = _make_client(client)
    try:
        if be == "pubmed":
            return await _search_pubmed(
                query.strip(), max_results=max_results, client=cli, email=email,
            )
        return await _search_europepmc(
            query.strip(), max_results=max_results, client=cli,
        )
    finally:
        if owned:
            await cli.aclose()


# --- public: fetch ---


def _bioc_to_text(data: Any) -> str:
    """Flatten a BioC JSON document into plain text.

    BioC shape: a top-level collection (list or dict) → `documents` →
    `passages` → each with a `text` field (a title, paragraph, table
    caption, etc., in reading order). We join the passage texts; that's
    exactly the full-text body, minus the XML scaffolding."""
    collections = data if isinstance(data, list) else [data]
    parts: list[str] = []
    for col in collections:
        if not isinstance(col, dict):
            continue
        for doc in col.get("documents", []):
            for psg in doc.get("passages", []):
                t = (psg.get("text") or "").strip()
                if t:
                    parts.append(t)
    return "\n\n".join(parts)


async def _pmc_fulltext(
    pmcid: str, *, client: httpx.AsyncClient,
) -> str:
    """Fetch OA full text for a PMCID via NCBI BioC and reduce to plain
    text. Returns "" when not available (non-OA, or service error). The
    service answers 200 with a plain-text error string for non-OA ids, so
    a JSON-parse failure is treated as "no full text", not an error."""
    url = NCBI_BIOC_FULLTEXT.format(pmcid=pmcid)
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.info("BioC fulltext fetch failed for %s: %s", pmcid, e)
        return ""
    except ValueError:
        # Non-JSON body → not an OA article with retrievable full text.
        return ""
    return _bioc_to_text(data)


async def _unpaywall_lookup(
    doi: str, *, email: str, client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Resolve a DOI to a legally-free version via Unpaywall. Returns a small
    dict {is_oa, oa_url, oa_pdf, oa_status, host} — empty-ish when none."""
    try:
        resp = await client.get(UNPAYWALL.format(doi=doi), params={"email": email})
        if resp.status_code == 404:
            return {"is_oa": False}
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.info("Unpaywall lookup failed for %s: %s", doi, e)
        return {"is_oa": False}
    except ValueError:
        return {"is_oa": False}

    best = data.get("best_oa_location") or {}
    return {
        "is_oa": bool(data.get("is_oa")),
        "oa_status": data.get("oa_status") or "",
        "oa_url": best.get("url") or "",
        "oa_pdf": best.get("url_for_pdf") or "",
        "host": best.get("host_type") or "",  # e.g. "publisher", "repository"
        "version": best.get("version") or "",  # e.g. "publishedVersion"
    }


async def fetch_article(
    identifier: str,
    *,
    include_full_text: bool = True,
    email: str = FALLBACK_EMAIL,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch metadata + abstract for one article, plus OA full text / a legal
    free link when available. Never raises for a missing article — returns a
    payload with `found: False` instead."""
    kind, value = _normalize_id(identifier)

    # Europe PMC is the metadata + abstract source regardless of id kind.
    if kind == "doi":
        epmc_query = f'DOI:"{value}"'
    elif kind == "pmcid":
        epmc_query = f'PMCID:{value}'
    else:
        epmc_query = f'EXT_ID:{value} AND SRC:MED'

    cli, owned = _make_client(client)
    try:
        hits = await _search_europepmc(epmc_query, max_results=1, client=cli)
        article = hits[0] if hits else None

        doi = (article.doi if article else "") or (value if kind == "doi" else "")
        pmcid = (article.pmcid if article else "") or (value if kind == "pmcid" else "")

        full_text = ""
        if include_full_text and pmcid:
            full_text = await _pmc_fulltext(pmcid, client=cli)

        unpaywall = {}
        if doi:
            unpaywall = await _unpaywall_lookup(doi, email=email, client=cli)
    finally:
        if owned:
            await cli.aclose()

    if article is None and not unpaywall.get("is_oa"):
        return {
            "found": False,
            "identifier": identifier,
            "message": (
                "No record found in Europe PMC for that identifier. "
                "Double-check the DOI/PMID/PMCID."
            ),
        }

    payload: dict[str, Any] = {
        "found": True,
        "identifier": identifier,
    }
    if article is not None:
        payload.update(article.to_payload())

    # Legal free-access resolution. Priority: OA full text we already pulled,
    # then an Unpaywall-resolved free copy, then the publisher link only.
    access: dict[str, Any] = {"open_access_full_text": False}
    if full_text:
        access["open_access_full_text"] = True
        payload["full_text"] = _clip(full_text, FULLTEXT_MAX_CHARS)
    if unpaywall.get("is_oa"):
        access["free_version_available"] = True
        access["free_url"] = unpaywall.get("oa_url", "")
        access["free_pdf"] = unpaywall.get("oa_pdf", "")
        access["free_version_type"] = unpaywall.get("version", "")
        access["free_host"] = unpaywall.get("host", "")
    else:
        access["free_version_available"] = False
        if not full_text:
            access["note"] = (
                "No openly licensed full text is available for this article. "
                "Only the abstract is provided; the link goes to the publisher's "
                "page. Do not attempt to bypass the paywall."
            )
    payload["access"] = access
    return payload


# --- Tool docs + schemas ---


MEDICAL_TOOL_DOCS = """\
**medical_literature_search** — Search the biomedical literature (Europe PMC
by default: PubMed/MEDLINE + PMC full text + preprints in one index).

When to use:
- The user asks about medical research, a condition, drug, biomarker, or
  "what does the science say about X" — especially for their health profile.
- You need citable primary literature rather than general web pages.

How to use:
- `query`: a literature query. Europe PMC supports field syntax, e.g.
  `metformin AND cardiovascular outcomes`, `AUTH:"Smith" AND vitamin D`.
- `max_results`: defaults to 8, max 25.
- Returns {title, authors, journal, year, doi, pmid, pmcid, is_open_access,
  is_preprint, abstract, url}. Follow up with `medical_article_fetch` for full text.
- `is_preprint: true` means medRxiv/bioRxiv-style preprint — NOT peer-reviewed.
  Keep it in the picture, but caveat it as preliminary; never present a
  preprint's findings as established or settled.

**medical_article_fetch** — Get one article by DOI / PMID / PMCID: abstract
always, open-access full text when it exists, and a legally-posted free
PDF/link (via Unpaywall) when the publisher copy is paywalled.

When to use:
- You have an identifier from a search hit and want the full text or a
  readable free copy to quote/summarize.

How to use:
- `identifier`: a DOI (`10.xxxx/...`), PMID (digits), or PMCID (`PMC123...`).
- `include_full_text`: default true; set false for just metadata + access links.
- If no open/free version exists, you get the abstract + publisher link and a
  note. Never bypass a paywall — report what's legally available.
"""


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Biomedical literature query. Field syntax supported (e.g. AUTH:, TITLE:).",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": HARD_MAX_RESULTS,
            "description": f"Max results. Default {DEFAULT_MAX_RESULTS}, max {HARD_MAX_RESULTS}.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "identifier": {
            "type": "string",
            "description": "DOI (10.xxxx/...), PMID (digits), or PMCID (PMC...).",
        },
        "include_full_text": {
            "type": "boolean",
            "description": "Include open-access full text when available (default true).",
        },
    },
    "required": ["identifier"],
    "additionalProperties": False,
}


# --- handlers (MCP shape; shared by SDK + provider paths) ---


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


async def _handle_search(args: dict[str, Any], *, config) -> dict[str, Any]:
    query = args.get("query")
    max_results = args.get("max_results") or DEFAULT_MAX_RESULTS
    if not isinstance(query, str) or not query.strip():
        return _error("query must be a non-empty string")
    try:
        backend = get_backend_name()
        results = await search_literature(
            query, max_results=max_results, backend=backend,
            email=_contact_email(config),
        )
    except ResearchToolError as e:
        log.info("medical_literature_search failed: %s", e)
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected medical_literature_search failure")
        return _error(f"Unexpected error: {e}")
    return _ok({
        "query": query,
        "backend": backend,
        "count": len(results),
        "results": [r.to_payload() for r in results],
    })


async def _handle_fetch(args: dict[str, Any], *, config) -> dict[str, Any]:
    identifier = args.get("identifier")
    include_full_text = args.get("include_full_text")
    if include_full_text is None:
        include_full_text = True
    if not isinstance(identifier, str) or not identifier.strip():
        return _error("identifier must be a non-empty string")
    try:
        payload = await fetch_article(
            identifier, include_full_text=bool(include_full_text),
            email=_contact_email(config),
        )
    except ResearchToolError as e:
        log.info("medical_article_fetch failed: %s", e)
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected medical_article_fetch failure")
        return _error(f"Unexpected error: {e}")
    return _ok(payload)


# --- SDK-shape registration ---


def medical_tools_disabled() -> bool:
    return os.environ.get("MEDICAL_TOOLS_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def build_research_sdk_tools(config) -> list:
    """SdkMcpTool wrappers. Returns [] if the SDK isn't installed."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    @tool(
        name="medical_literature_search",
        description=(
            "Search biomedical literature (Europe PMC: PubMed + PMC + preprints). "
            "Returns structured hits with abstracts; follow up with "
            "medical_article_fetch for full text."
        ),
        input_schema=SEARCH_SCHEMA,
    )
    async def search_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_search(args, config=config)

    @tool(
        name="medical_article_fetch",
        description=(
            "Fetch one article by DOI/PMID/PMCID: abstract, open-access full "
            "text, and a legal free PDF/link via Unpaywall when paywalled."
        ),
        input_schema=FETCH_SCHEMA,
    )
    async def fetch_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_fetch(args, config=config)

    return [search_tool, fetch_tool]
