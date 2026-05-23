"""Tests for the native websearch tool (Phase C.5 part 2).

We never hit DuckDuckGo for real — every test injects a fake backend
or mocks the duckduckgo-search package at the import site. The DDG
HTML format can change anytime, so contract-level testing against the
backend abstraction is the durable surface.

Coverage:
- get_backend(): env-driven selection (ddg/brave/serper/tavily),
  default to ddg, unknown name raises SearchToolError
- search(): query validation, max_results clamping, backend
  delegation
- DDGBackend: returns parsed SearchResult list when DDGS yields rows;
  empty list when nothing returned; SearchToolError when the package
  raises or isn't installed
- Stub backends (Brave/Serper/Tavily): search() raises SearchToolError
  with a clear "not implemented" message
- Handler MCP shape: success + error paths
- Provider bridge unwrap
- Env flag (WEBSEARCH_DISABLED) parsing
"""

from __future__ import annotations

import json
import sys

import pytest

from src.tools.search import (
    DEFAULT_MAX_RESULTS,
    HARD_MAX_RESULTS,
    BraveBackend,
    DDGBackend,
    SearchBackend,
    SearchResult,
    SearchToolError,
    SerperBackend,
    TavilyBackend,
    _handle_websearch,
    get_backend,
    search,
    websearch_disabled,
)


# --- Fake backend for testing ---


class _FakeBackend(SearchBackend):
    name = "fake"

    def __init__(self, results: list[SearchResult] | Exception):
        self.results = results
        self.last_query: str | None = None
        self.last_max: int | None = None

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.last_query = query
        self.last_max = max_results
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results)


# --- get_backend ---


class TestGetBackend:
    def test_default_is_ddg(self, monkeypatch):
        monkeypatch.delenv("WEBSEARCH_BACKEND", raising=False)
        be = get_backend()
        assert isinstance(be, DDGBackend)

    def test_env_selects_brave_stub(self, monkeypatch):
        monkeypatch.setenv("WEBSEARCH_BACKEND", "brave")
        be = get_backend()
        assert isinstance(be, BraveBackend)

    def test_explicit_name_overrides_env(self, monkeypatch):
        monkeypatch.setenv("WEBSEARCH_BACKEND", "brave")
        be = get_backend("tavily")
        assert isinstance(be, TavilyBackend)

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("WEBSEARCH_BACKEND", "bogus")
        with pytest.raises(SearchToolError, match="Unknown WEBSEARCH_BACKEND"):
            get_backend()

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("WEBSEARCH_BACKEND", "DDG")
        assert isinstance(get_backend(), DDGBackend)


# --- search() ---


@pytest.mark.asyncio
class TestSearch:
    async def test_delegates_to_backend(self):
        backend = _FakeBackend([
            SearchResult(title="A", url="https://a", snippet="..."),
            SearchResult(title="B", url="https://b", snippet="..."),
        ])
        out = await search("hello", max_results=5, backend=backend)
        assert len(out) == 2
        assert backend.last_query == "hello"
        assert backend.last_max == 5

    async def test_empty_query_rejected(self):
        with pytest.raises(SearchToolError, match="non-empty"):
            await search("")

    async def test_whitespace_only_query_rejected(self):
        with pytest.raises(SearchToolError, match="non-empty"):
            await search("   \n  ")

    async def test_non_string_rejected(self):
        with pytest.raises(SearchToolError, match="non-empty"):
            await search(None)  # type: ignore[arg-type]

    async def test_max_results_clamped_high(self):
        backend = _FakeBackend([])
        await search("q", max_results=1000, backend=backend)
        assert backend.last_max == HARD_MAX_RESULTS

    async def test_max_results_clamped_low(self):
        backend = _FakeBackend([])
        await search("q", max_results=0, backend=backend)
        assert backend.last_max == 1


# --- DDGBackend ---


@pytest.mark.asyncio
class TestDDGBackend:
    async def test_happy_path(self, monkeypatch):
        # Build a stub duckduckgo_search module with a DDGS that
        # supports the context-manager + .text() API.
        fake_results = [
            {"title": "T1", "href": "https://x.test/a", "body": "snippet 1"},
            {"title": "T2", "href": "https://x.test/b", "body": "snippet 2"},
        ]

        class _StubDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def text(self, query, max_results):
                return iter(fake_results)

        module = type(sys)("duckduckgo_search")
        module.DDGS = _StubDDGS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "duckduckgo_search", module)

        backend = DDGBackend()
        results = await backend.search("anything", max_results=5)
        assert len(results) == 2
        assert results[0].title == "T1"
        assert results[0].url == "https://x.test/a"
        assert results[0].snippet == "snippet 1"

    async def test_missing_package_raises_clean_error(self, monkeypatch):
        # Force ImportError on the lazy import
        monkeypatch.setitem(sys.modules, "duckduckgo_search", None)
        backend = DDGBackend()
        with pytest.raises(SearchToolError, match="duckduckgo-search"):
            await backend.search("q", max_results=5)

    async def test_package_exception_wrapped(self, monkeypatch):
        # The package can raise from rate-limit / parse / network. Confirm
        # we surface a SearchToolError rather than the raw exception.
        class _StubDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def text(self, query, max_results):
                raise RuntimeError("rate limit hit")

        module = type(sys)("duckduckgo_search")
        module.DDGS = _StubDDGS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "duckduckgo_search", module)

        backend = DDGBackend()
        with pytest.raises(SearchToolError, match="DDG search failed"):
            await backend.search("q", max_results=5)

    async def test_empty_results_returns_empty_list(self, monkeypatch):
        class _StubDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def text(self, query, max_results):
                return iter([])

        module = type(sys)("duckduckgo_search")
        module.DDGS = _StubDDGS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "duckduckgo_search", module)

        backend = DDGBackend()
        assert await backend.search("nothing", max_results=5) == []

    async def test_malformed_rows_skipped(self, monkeypatch):
        # Robustness: if DDGS returns a row missing title AND url, skip
        # it rather than emit a useless empty SearchResult.
        class _StubDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def text(self, query, max_results):
                return iter([
                    {"title": "ok", "href": "https://ok", "body": "."},
                    {"title": "", "href": "", "body": "nothing useful"},
                    "this is not even a dict",
                ])

        module = type(sys)("duckduckgo_search")
        module.DDGS = _StubDDGS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "duckduckgo_search", module)

        backend = DDGBackend()
        results = await backend.search("q", max_results=5)
        assert len(results) == 1
        assert results[0].title == "ok"


# --- Stub backends ---


@pytest.mark.asyncio
class TestStubBackends:
    @pytest.mark.parametrize("cls", [BraveBackend, SerperBackend, TavilyBackend])
    async def test_stub_raises_with_signup_url(self, cls):
        backend = cls()
        with pytest.raises(SearchToolError, match="not yet implemented"):
            await backend.search("q", max_results=5)


# --- Handler MCP shape ---


@pytest.mark.asyncio
class TestHandlerMcpShape:
    async def test_success_payload(self, monkeypatch):
        # Patch search() to return a known result
        async def fake_search(query, **kw):
            return [SearchResult(title="X", url="https://x", snippet="...")]

        monkeypatch.setattr("src.tools.search.search", fake_search)

        result = await _handle_websearch({"query": "hello"})
        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["query"] == "hello"
        assert payload["count"] == 1
        assert payload["results"][0]["url"] == "https://x"
        assert "backend" in payload

    async def test_empty_query_returns_is_error(self):
        result = await _handle_websearch({})
        assert result.get("is_error") is True

    async def test_search_error_returns_is_error(self, monkeypatch):
        async def boom(query, **kw):
            raise SearchToolError("DDG search failed: rate limit")

        monkeypatch.setattr("src.tools.search.search", boom)

        result = await _handle_websearch({"query": "anything"})
        assert result.get("is_error") is True
        assert "rate limit" in result["content"][0]["text"]


# --- Provider bridge unwrap ---


@pytest.mark.asyncio
class TestProviderBridge:
    async def test_success_via_unwrap(self, monkeypatch):
        from src.tools import _unwrap_mcp_result

        async def fake_search(query, **kw):
            return [SearchResult(title="A", url="https://a", snippet="s")]

        monkeypatch.setattr("src.tools.search.search", fake_search)

        mcp = await _handle_websearch({"query": "test"})
        text = _unwrap_mcp_result(mcp, tool_name="websearch")
        payload = json.loads(text)
        assert payload["count"] == 1

    async def test_error_via_unwrap(self, monkeypatch):
        from src.tools import _unwrap_mcp_result

        async def boom(query, **kw):
            raise SearchToolError("simulated")

        monkeypatch.setattr("src.tools.search.search", boom)

        mcp = await _handle_websearch({"query": "q"})
        with pytest.raises(RuntimeError, match="simulated"):
            _unwrap_mcp_result(mcp, tool_name="websearch")


# --- Env flag ---


class TestEnvFlag:
    def test_disabled_when_set(self, monkeypatch):
        monkeypatch.setenv("WEBSEARCH_DISABLED", "true")
        assert websearch_disabled() is True

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("WEBSEARCH_DISABLED", raising=False)
        assert websearch_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("WEBSEARCH_DISABLED", val)
        assert websearch_disabled() is True
