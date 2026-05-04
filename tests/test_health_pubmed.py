"""PubMed source tests — projection logic + the HTTP fan-out, with
aiohttp mocked at the session layer.

We don't test against live NCBI; that'd be flaky and rate-limited.
The unit tests cover the projection (esummary record → Evidence) and
the request shape (params, base URL); a sanity smoke against real
NCBI is left as a manual validation step.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from src.health.retrieval.pubmed import (
    PubMedSource,
    _format_authors,
    _journal_quality_score,
    _parse_pub_date,
    _summary_to_evidence,
)


class TestParsePubDate:
    def test_year_only(self):
        assert _parse_pub_date("2024") == date(2024, 1, 1)

    def test_year_month(self):
        assert _parse_pub_date("2024 Mar") == date(2024, 3, 1)

    def test_year_month_day(self):
        assert _parse_pub_date("2024 Mar 15") == date(2024, 3, 15)

    def test_seasonal_falls_back_to_year(self):
        # PubMed sometimes emits "2024 Spring" — we don't try to map
        # seasons to months; year-only is the safe fallback.
        assert _parse_pub_date("2024 Spring") == date(2024, 1, 1)

    def test_garbage_returns_none(self):
        assert _parse_pub_date("not a date") is None
        assert _parse_pub_date("") is None
        assert _parse_pub_date(None) is None  # type: ignore[arg-type]


class TestFormatAuthors:
    def test_empty(self):
        assert _format_authors([]) == ""

    def test_one_author(self):
        assert _format_authors(["Smith J"]) == "Smith J"

    def test_multiple_authors_collapsed(self):
        # First-author + "et al." matches PubMed citation conventions
        assert _format_authors(["Smith J", "Doe K", "Lee P"]) == "Smith J et al."


class TestJournalQualityScore:
    def test_unknown_journal_baseline(self):
        assert _journal_quality_score("Some Random Journal") == 0.6

    def test_no_journal(self):
        # No journal shouldn't punish too hard — the authority_weight
        # of the source already says "this is PubMed."
        assert _journal_quality_score("") == 0.5

    def test_prestigious_journal_bumped(self):
        for j in ["JAMA", "Lancet", "N Engl J Med", "BMJ"]:
            assert _journal_quality_score(j) == 0.9, f"{j} should be 0.9"

    def test_journal_substring_match(self):
        # "JAMA Pediatrics", "JAMA Internal Medicine" all count
        assert _journal_quality_score("JAMA Internal Medicine") == 0.9


class TestSummaryToEvidence:
    def test_minimal_record(self):
        summary = {
            "title": "Effect of metformin on cardiovascular outcomes",
            "fulljournalname": "Lancet",
            "pubdate": "2024 Mar 15",
            "authors": [{"name": "Smith J"}, {"name": "Doe K"}],
        }
        ev = _summary_to_evidence("12345", summary)
        assert ev is not None
        assert ev.source == "pubmed"
        assert ev.title.startswith("Effect of metformin")
        assert "Smith J et al." in ev.snippet
        assert "Lancet" in ev.snippet
        assert ev.published == date(2024, 3, 15)
        assert ev.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
        assert ev.quality_score == 0.9  # Lancet is prestigious

    def test_no_title_returns_none(self):
        # Skip malformed records rather than surface blanks
        assert _summary_to_evidence("1", {"title": ""}) is None

    def test_authors_optional(self):
        ev = _summary_to_evidence("1", {
            "title": "x", "fulljournalname": "j", "pubdate": "2024",
        })
        assert ev is not None  # works without authors

    def test_no_metadata_snippet_fallback(self):
        ev = _summary_to_evidence("1", {"title": "x"})
        assert ev is not None
        assert "(no source metadata)" in ev.snippet

    def test_url_uses_pmid(self):
        ev = _summary_to_evidence("99999", {"title": "t"})
        assert ev is not None
        assert "99999" in ev.url


# --- HTTP-level tests with mocked aiohttp session ---


class _MockResponse:
    """Minimal stand-in for aiohttp's response context manager."""

    def __init__(self, status: int, json_body: dict | None = None, text_body: str = ""):
        self.status = status
        self._json = json_body or {}
        self._text = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _MockSession:
    """Records URLs/params; returns scripted responses in order."""

    def __init__(self, responses: list[_MockResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, params=None, **kw):
        self.calls.append((url, params or {}))
        if not self._responses:
            raise AssertionError("MockSession out of responses")
        return self._responses.pop(0)


def _mock_make_session(responses):
    """Patch make_session() to yield a scripted MockSession."""
    return patch(
        "src.health.retrieval.pubmed.make_session",
        return_value=_MockSession(responses),
    )


class TestPubMedSearch:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        # Don't burn an API call on no input
        src = PubMedSource()
        with patch("src.health.retrieval.pubmed.make_session") as mk:
            results = await src.search("")
        assert results == []
        mk.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pmids_short_circuits(self):
        """When esearch returns no IDs we don't bother calling esummary."""
        responses = [
            _MockResponse(200, {"esearchresult": {"idlist": []}}),
        ]
        with _mock_make_session(responses):
            src = PubMedSource()
            results = await src.search("nonsense query xyz")
        assert results == []

    @pytest.mark.asyncio
    async def test_full_flow(self):
        esearch_resp = {
            "esearchresult": {"idlist": ["111", "222"]}
        }
        esummary_resp = {
            "result": {
                "uids": ["111", "222"],
                "111": {
                    "title": "Metformin and lactic acidosis",
                    "fulljournalname": "JAMA",
                    "pubdate": "2024 Jan 10",
                    "authors": [{"name": "Smith J"}],
                },
                "222": {
                    "title": "B12 deficiency in metformin users",
                    "fulljournalname": "Diabetes Care",
                    "pubdate": "2023 Aug",
                    "authors": [{"name": "Doe K"}, {"name": "Lee P"}],
                },
            }
        }
        responses = [
            _MockResponse(200, esearch_resp),
            _MockResponse(200, esummary_resp),
        ]
        with _mock_make_session(responses) as mk:
            src = PubMedSource()
            results = await src.search("metformin", topic="diabetes")

        assert len(results) == 2
        assert all(r.source == "pubmed" for r in results)
        # First call is esearch, second esummary — verify the URL shape
        # and that the api_key wasn't accidentally added when not set.
        calls = mk.return_value.calls
        assert "esearch.fcgi" in calls[0][0]
        assert "esummary.fcgi" in calls[1][0]
        assert "api_key" not in calls[0][1]

    @pytest.mark.asyncio
    async def test_api_key_passed_when_set(self):
        responses = [
            _MockResponse(200, {"esearchresult": {"idlist": []}}),
        ]
        with _mock_make_session(responses) as mk:
            src = PubMedSource(api_key="ncbi-key-abc")
            await src.search("x")
        params = mk.return_value.calls[0][1]
        assert params.get("api_key") == "ncbi-key-abc"

    @pytest.mark.asyncio
    async def test_esearch_error_raises(self):
        """5xx from NCBI bubbles up as an exception so the orchestrator
        can catch and continue with other sources."""
        responses = [_MockResponse(503, text_body="upstream busy")]
        with _mock_make_session(responses):
            src = PubMedSource()
            with pytest.raises(RuntimeError, match="503"):
                await src.search("x")
