"""Tests for the medical literature tools (src/tools/research.py).

All HTTP is mocked via httpx.MockTransport so these run offline and
deterministically. We exercise parsing for both backends, the
open-access full-text path, graceful degradation when nothing is openly
available, and identifier normalization.
"""

import json

import httpx
import pytest

from src.tools import research as R


# --- canned API payloads ---

_EPMC_SEARCH = {
    "resultList": {
        "result": [
            {
                "title": "Effect of metformin on cardiovascular outcomes.",
                "authorString": "Smith J, Doe A, Roe B",
                "journalTitle": "Diabetes Care",
                "pubYear": 2021,
                "doi": "10.1234/dc.2021.001",
                "pmid": "33333333",
                "pmcid": "PMC8000001",
                "isOpenAccess": "Y",
                "inEPMC": "Y",
                "abstractText": "Metformin reduced MACE in this cohort.",
            }
        ]
    }
}

# NCBI BioC full-text shape: collection → documents → passages → text.
_BIOC_FULLTEXT = [
    {
        "documents": [
            {
                "passages": [
                    {"text": "Effect of metformin on cardiovascular outcomes"},
                    {"text": "Results"},
                    {"text": "Metformin lowered cardiovascular events by 20%."},
                ]
            }
        ]
    }
]

_UNPAYWALL_OA = {
    "is_oa": True,
    "oa_status": "gold",
    "best_oa_location": {
        "url": "https://example.org/article",
        "url_for_pdf": "https://example.org/article.pdf",
        "version": "publishedVersion",
        "host_type": "publisher",
    },
}

_UNPAYWALL_CLOSED = {"is_oa": False, "oa_status": "closed", "best_oa_location": None}

_ESEARCH = {"esearchresult": {"idlist": ["33333333", "44444444"]}}
_ESUMMARY = {
    "result": {
        "33333333": {
            "title": "Metformin and the heart.",
            "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
            "fulljournalname": "Diabetes Care",
            "pubdate": "2021 Jan",
            "articleids": [{"idtype": "doi", "value": "10.1234/dc.2021.001"}],
        },
        "44444444": {
            "title": "Vitamin D review.",
            "authors": [{"name": "Roe B"}],
            "fulljournalname": "BMJ",
            "pubdate": "2020 Mar",
            "articleids": [],
        },
    }
}


def _make_client(router) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(router))


def _json_response(payload, status=200) -> httpx.Response:
    return httpx.Response(status, json=payload)


class TestEuropePmcSearch:
    @pytest.mark.asyncio
    async def test_parses_results(self):
        def router(req: httpx.Request) -> httpx.Response:
            assert "europepmc" in str(req.url)
            assert req.url.params.get("resultType") == "core"
            return _json_response(_EPMC_SEARCH)

        client = _make_client(router)
        results = await R.search_literature(
            "metformin cardiovascular", backend="europepmc", client=client,
        )
        await client.aclose()

        assert len(results) == 1
        a = results[0]
        assert a.title == "Effect of metformin on cardiovascular outcomes"  # trailing dot stripped
        assert a.authors == ["Smith J", "Doe A", "Roe B"]
        assert a.doi == "10.1234/dc.2021.001"
        assert a.pmcid == "PMC8000001"
        assert a.is_open_access is True
        assert a.url == "https://doi.org/10.1234/dc.2021.001"
        assert a.source == "europepmc"


class TestPubmedSearch:
    @pytest.mark.asyncio
    async def test_esearch_then_esummary(self):
        def router(req: httpx.Request) -> httpx.Response:
            if req.url.path.endswith("esearch.fcgi"):
                assert req.url.params.get("email")  # polite identifier passed
                return _json_response(_ESEARCH)
            if req.url.path.endswith("esummary.fcgi"):
                return _json_response(_ESUMMARY)
            raise AssertionError(f"unexpected URL {req.url}")

        client = _make_client(router)
        results = await R.search_literature(
            "vitamin d", backend="pubmed", client=client, email="me@example.com",
        )
        await client.aclose()

        assert [r.pmid for r in results] == ["33333333", "44444444"]
        assert results[0].doi == "10.1234/dc.2021.001"
        assert results[0].year == "2021"
        assert results[1].journal == "BMJ"
        assert all(r.source == "pubmed" for r in results)


class TestFetchArticle:
    @pytest.mark.asyncio
    async def test_open_access_full_text(self):
        def router(req: httpx.Request) -> httpx.Response:
            host, path = req.url.host, req.url.path
            if "europepmc" in str(req.url) and path.endswith("/search"):
                return _json_response(_EPMC_SEARCH)
            if "bionlp" in path or "BioC" in path:
                return _json_response(_BIOC_FULLTEXT)
            if "unpaywall" in host:
                return _json_response(_UNPAYWALL_OA)
            raise AssertionError(f"unexpected URL {req.url}")

        client = _make_client(router)
        payload = await R.fetch_article(
            "10.1234/dc.2021.001", email="me@example.com", client=client,
        )
        await client.aclose()

        assert payload["found"] is True
        assert "Metformin lowered cardiovascular events" in payload["full_text"]
        assert payload["access"]["open_access_full_text"] is True
        assert payload["access"]["free_version_available"] is True
        assert payload["access"]["free_pdf"] == "https://example.org/article.pdf"

    @pytest.mark.asyncio
    async def test_paywalled_degrades_gracefully(self):
        # Record exists, but no PMCID (no OA full text) and Unpaywall says closed.
        epmc_no_pmc = {
            "resultList": {"result": [{
                "title": "Paywalled study.",
                "authorString": "Author X",
                "journalTitle": "Nature",
                "pubYear": 2019,
                "doi": "10.9999/nat.2019",
                "pmid": "55555555",
                "isOpenAccess": "N",
                "abstractText": "Abstract only.",
            }]}
        }

        def router(req: httpx.Request) -> httpx.Response:
            if "europepmc" in str(req.url) and req.url.path.endswith("/search"):
                return _json_response(epmc_no_pmc)
            if "unpaywall" in req.url.host:
                return _json_response(_UNPAYWALL_CLOSED)
            raise AssertionError(f"unexpected URL {req.url}")

        client = _make_client(router)
        payload = await R.fetch_article(
            "10.9999/nat.2019", email="me@example.com", client=client,
        )
        await client.aclose()

        assert payload["found"] is True
        assert "full_text" not in payload
        assert payload["access"]["free_version_available"] is False
        assert "do not attempt to bypass" in payload["access"]["note"].lower()
        assert payload["abstract"] == "Abstract only."

    @pytest.mark.asyncio
    async def test_not_found(self):
        def router(req: httpx.Request) -> httpx.Response:
            if "europepmc" in str(req.url):
                return _json_response({"resultList": {"result": []}})
            if "unpaywall" in req.url.host:
                return httpx.Response(404, json={"error": True})
            raise AssertionError(f"unexpected URL {req.url}")

        client = _make_client(router)
        payload = await R.fetch_article(
            "10.0000/missing", email="me@example.com", client=client,
        )
        await client.aclose()
        assert payload["found"] is False


class TestPreprintFlagging:
    """Preprints (medRxiv/bioRxiv) stay in results but are flagged is_preprint
    so the model can caveat them as not-peer-reviewed."""

    def test_epmc_preprint_by_source_ppr(self):
        a = R._epmc_article({
            "title": "A long covid preprint", "source": "PPR",
            "doi": "10.1101/2026.01.01", "pubYear": 2026,
            "journalTitle": "", "authorString": "Smith J",
        })
        assert a.is_preprint is True
        assert a.to_payload()["is_preprint"] is True

    def test_epmc_preprint_by_pubtype(self):
        a = R._epmc_article({
            "title": "X", "source": "MED", "pubType": "Preprint",
            "doi": "10.1/x",
        })
        assert a.is_preprint is True

    def test_epmc_peer_reviewed_not_flagged(self):
        a = R._epmc_article({
            "title": "Peer reviewed study", "source": "MED",
            "journalTitle": "Nature", "doi": "10.1/x",
        })
        assert a.is_preprint is False
        assert a.to_payload()["is_preprint"] is False

    @pytest.mark.asyncio
    async def test_search_payload_carries_flag(self):
        preprint_hit = {
            "resultList": {"result": [{
                "title": "medRxiv finding", "source": "PPR",
                "journalTitle": "medRxiv", "pubYear": 2026,
                "doi": "10.1101/x", "authorString": "A B",
            }]}
        }

        def router(req: httpx.Request) -> httpx.Response:
            return _json_response(preprint_hit)

        client = _make_client(router)
        results = await R.search_literature("anything", client=client)
        await client.aclose()
        assert results[0].is_preprint is True
        assert results[0].journal == "medRxiv"


class TestNormalizeId:
    def test_doi(self):
        assert R._normalize_id("10.1234/abc") == ("doi", "10.1234/abc")
        assert R._normalize_id("doi:10.1/x") == ("doi", "10.1/x")

    def test_pmcid(self):
        assert R._normalize_id("PMC123456") == ("pmcid", "PMC123456")
        assert R._normalize_id("pmc99") == ("pmcid", "PMC99")

    def test_pmid(self):
        assert R._normalize_id("33333333") == ("pmid", "33333333")

    def test_empty_raises(self):
        with pytest.raises(R.ResearchToolError):
            R._normalize_id("  ")


class TestHandlers:
    @pytest.mark.asyncio
    async def test_search_handler_rejects_empty_query(self):
        class _Cfg:
            research_contact_email = "me@example.com"

        result = await R._handle_search({"query": "  "}, config=_Cfg())
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        with pytest.raises(R.ResearchToolError):
            R.get_backend_name("scopus")
