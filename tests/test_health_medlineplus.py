"""MedlinePlus source tests — XML parsing + the HTTP path with mocked
aiohttp."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.health.retrieval.medlineplus import (
    MedlinePlusSource,
    _parse_results,
    _strip_html,
)
import xml.etree.ElementTree as ET


_SAMPLE_XML = """<?xml version="1.0"?>
<nlmSearchResult>
  <list>
    <document rank="0" url="https://medlineplus.gov/metformin.html">
      <content name="title">Metformin</content>
      <content name="FullSummary">&lt;p&gt;&lt;span class="qt0"&gt;Metformin&lt;/span&gt; is used to treat type 2 diabetes. It works by lowering glucose production in the liver and improving insulin sensitivity.&lt;/p&gt;</content>
    </document>
    <document rank="1" url="https://medlineplus.gov/diabetes.html">
      <content name="title">Diabetes &lt;span class="qt0"&gt;Type 2&lt;/span&gt;</content>
      <content name="FullSummary">A condition affecting how your body uses glucose.</content>
    </document>
  </list>
</nlmSearchResult>"""


_EMPTY_XML = """<?xml version="1.0"?>
<nlmSearchResult><list></list></nlmSearchResult>"""


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>hello <b>world</b></p>") == "hello world"

    def test_removes_query_term_highlights(self):
        # MedlinePlus wraps query-term hits in qt0 spans
        text = '<span class="qt0">metformin</span> is a drug'
        assert _strip_html(text) == "metformin is a drug"

    def test_collapses_whitespace(self):
        assert _strip_html("foo\n\n  bar  baz") == "foo bar baz"

    def test_handles_empty(self):
        assert _strip_html("") == ""
        assert _strip_html(None) == ""  # type: ignore[arg-type]


class TestParseResults:
    def test_basic_parse(self):
        root = ET.fromstring(_SAMPLE_XML)
        results = _parse_results(root)
        assert len(results) == 2
        assert results[0].source == "medlineplus"
        assert results[0].title == "Metformin"
        assert "type 2 diabetes" in results[0].snippet
        # HTML stripped from snippet
        assert "<p>" not in results[0].snippet
        assert "<span" not in results[0].snippet
        assert results[0].url == "https://medlineplus.gov/metformin.html"

    def test_html_stripped_from_title(self):
        root = ET.fromstring(_SAMPLE_XML)
        results = _parse_results(root)
        # The second doc has a span inside the title
        assert "<span" not in results[1].title
        assert "Type 2" in results[1].title

    def test_empty_response(self):
        root = ET.fromstring(_EMPTY_XML)
        assert _parse_results(root) == []

    def test_skips_doc_without_title(self):
        xml = """<?xml version="1.0"?>
<nlmSearchResult><list>
  <document rank="0" url="https://x.test/">
    <content name="FullSummary">no title here</content>
  </document>
</list></nlmSearchResult>"""
        root = ET.fromstring(xml)
        assert _parse_results(root) == []

    def test_skips_doc_without_url(self):
        xml = """<?xml version="1.0"?>
<nlmSearchResult><list>
  <document rank="0">
    <content name="title">No URL</content>
  </document>
</list></nlmSearchResult>"""
        root = ET.fromstring(xml)
        assert _parse_results(root) == []

    def test_long_summary_truncated(self):
        long_summary = "x" * 1000
        xml = f"""<?xml version="1.0"?>
<nlmSearchResult><list>
  <document rank="0" url="https://x.test/">
    <content name="title">T</content>
    <content name="FullSummary">{long_summary}</content>
  </document>
</list></nlmSearchResult>"""
        root = ET.fromstring(xml)
        results = _parse_results(root)
        assert len(results[0].snippet) <= 400

    def test_published_always_none(self):
        """MedlinePlus pages don't have reliable per-page dates;
        published=None is correct, the ranker handles it."""
        root = ET.fromstring(_SAMPLE_XML)
        for r in _parse_results(root):
            assert r.published is None


# --- HTTP-level tests ---


class _MockResponse:
    def __init__(self, status: int, text_body: str = ""):
        self.status = status
        self._text = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text


class _MockSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, params=None, **kw):
        self.calls.append((url, params or {}))
        if not self._responses:
            raise AssertionError("MockSession out of responses")
        return self._responses.pop(0)


def _mock(responses):
    return patch(
        "src.health.retrieval.medlineplus.make_session",
        return_value=_MockSession(responses),
    )


class TestMedlinePlusSearch:
    @pytest.mark.asyncio
    async def test_empty_query(self):
        src = MedlinePlusSource()
        with patch("src.health.retrieval.medlineplus.make_session") as mk:
            results = await src.search("")
        assert results == []
        mk.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_flow(self):
        with _mock([_MockResponse(200, _SAMPLE_XML)]) as mk:
            src = MedlinePlusSource()
            results = await src.search("metformin")
        assert len(results) == 2
        params = mk.return_value.calls[0][1]
        assert params["term"] == "metformin"
        assert params["db"] == "healthTopics"

    @pytest.mark.asyncio
    async def test_malformed_xml_returns_empty_no_crash(self):
        """Defensive: bad XML from upstream shouldn't break the turn."""
        with _mock([_MockResponse(200, "<broken><xml")]):
            src = MedlinePlusSource()
            results = await src.search("x")
        assert results == []

    @pytest.mark.asyncio
    async def test_5xx_raises_for_orchestrator_to_catch(self):
        with _mock([_MockResponse(503, "upstream busy")]):
            src = MedlinePlusSource()
            with pytest.raises(RuntimeError, match="503"):
                await src.search("x")
