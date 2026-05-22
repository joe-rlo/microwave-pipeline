"""Tests for the native webfetch tool.

Coverage:
- SSRF guard: blocks localhost, private IPs, .internal, schemes other than http(s)
- HTML → text conversion: tags stripped, scripts/styles dropped, entities decoded
- Content-type whitelist: text/html allowed, application/octet-stream rejected
- Size caps: max_chars truncation, max_bytes pre-flight rejection by Content-Length
- Error paths: 4xx, network error, malformed input
- Handler shape: success returns MCP-shape payload, errors return is_error=True
- Provider bridge: handler unwraps cleanly via existing _unwrap_mcp_result

We never make a real network call. httpx.MockTransport drives every
test, identical pattern to test_near_provider.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.tools.web import (
    WebFetchError,
    _handle_webfetch,
    _html_to_text,
    _is_text_content_type,
    _normalize_whitespace,
    _validate_url,
    fetch_url_as_text,
)


# --- _validate_url ---


class TestValidateUrl:
    def test_https_ok(self):
        _validate_url("https://example.com/path")

    def test_http_ok(self):
        _validate_url("http://example.com")

    def test_empty_rejected(self):
        with pytest.raises(WebFetchError, match="non-empty"):
            _validate_url("")

    def test_non_string_rejected(self):
        with pytest.raises(WebFetchError, match="non-empty"):
            _validate_url(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("scheme", ["ftp", "file", "javascript", "data"])
    def test_bad_schemes_rejected(self, scheme):
        with pytest.raises(WebFetchError, match="scheme"):
            _validate_url(f"{scheme}://example.com")

    @pytest.mark.parametrize("host", [
        "localhost",
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "::1",
        "169.254.169.254",  # AWS / GCP metadata
        "0.0.0.0",
    ])
    def test_private_and_loopback_blocked(self, host):
        with pytest.raises(WebFetchError):
            _validate_url(f"http://{host}/")

    def test_dot_internal_blocked(self):
        with pytest.raises(WebFetchError, match="internal-only"):
            _validate_url("https://admin.internal/")

    def test_dot_local_blocked(self):
        with pytest.raises(WebFetchError, match="internal-only"):
            _validate_url("https://printer.local/")

    def test_no_host_rejected(self):
        with pytest.raises(WebFetchError, match="hostname"):
            _validate_url("http:///path")


# --- HTML → text ---


class TestHtmlToText:
    def test_strips_tags(self):
        text = _html_to_text("<p>hello <b>world</b></p>")
        assert "hello" in text
        assert "world" in text
        assert "<p>" not in text
        assert "<b>" not in text

    def test_drops_scripts_and_styles(self):
        html = "<html><script>alert('x')</script><style>p{color:red}</style><p>visible</p></html>"
        text = _html_to_text(html)
        assert "visible" in text
        assert "alert" not in text
        assert "color:red" not in text

    def test_unescapes_entities(self):
        text = _html_to_text("<p>5 &lt; 10 &amp; 20 &gt; 15</p>")
        assert "5 < 10" in text
        assert "& 20" in text
        assert "20 > 15" in text


class TestWhitespaceNormalization:
    def test_collapses_spaces(self):
        assert _normalize_whitespace("a    b    c") == "a b c"

    def test_collapses_blank_lines(self):
        assert _normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_per_line(self):
        text = _normalize_whitespace("  line one  \n  line two  ")
        assert text == "line one\nline two"


class TestContentTypeWhitelist:
    @pytest.mark.parametrize("ct", [
        "text/html",
        "text/plain",
        "text/markdown",
        "application/json",
        "application/xml",
        "application/json; charset=utf-8",
        "TEXT/HTML",  # case insensitive
    ])
    def test_allowed(self, ct):
        assert _is_text_content_type(ct) is True

    @pytest.mark.parametrize("ct", [
        "image/png",
        "application/pdf",
        "application/octet-stream",
        "video/mp4",
        "audio/mpeg",
    ])
    def test_rejected(self, ct):
        assert _is_text_content_type(ct) is False

    def test_empty_treated_as_allowed(self):
        # Many origins omit Content-Type; lean permissive.
        assert _is_text_content_type("") is True


# --- fetch_url_as_text (HTTP layer) ---


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )


@pytest.mark.asyncio
class TestFetchUrlAsText:
    async def test_basic_html_extraction(self):
        async def h(req):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body><h1>Hi</h1><p>Body text</p></body></html>",
            )

        client = _mock_client(h)
        text = await fetch_url_as_text("https://example.com/", client=client)
        await client.aclose()
        assert "Hi" in text
        assert "Body text" in text
        assert "<h1>" not in text

    async def test_plain_text_passes_through(self):
        async def h(req):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="just  some   text\n\n\n\nlines",
            )

        client = _mock_client(h)
        text = await fetch_url_as_text("https://example.com/file.txt", client=client)
        await client.aclose()
        # Whitespace normalized but content preserved
        assert "just some text" in text
        assert "lines" in text

    async def test_4xx_raises(self):
        async def h(req):
            return httpx.Response(404, text="not found")

        client = _mock_client(h)
        with pytest.raises(WebFetchError) as exc_info:
            await fetch_url_as_text("https://example.com/missing", client=client)
        await client.aclose()
        assert exc_info.value.status == 404

    async def test_non_text_content_type_rejected(self):
        async def h(req):
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=b"\x00\x01\x02\x03",
            )

        client = _mock_client(h)
        with pytest.raises(WebFetchError, match="not text"):
            await fetch_url_as_text("https://example.com/blob", client=client)
        await client.aclose()

    async def test_max_chars_truncates(self):
        big_html = "<p>" + ("x" * 100_000) + "</p>"

        async def h(req):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=big_html,
            )

        client = _mock_client(h)
        text = await fetch_url_as_text(
            "https://example.com/big", client=client, max_chars=1000
        )
        await client.aclose()
        assert text.endswith("…[truncated]")
        # Truncated to roughly max_chars (+ marker), not the full 100K
        assert len(text) < 2000

    async def test_content_length_oversize_rejected(self):
        async def h(req):
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/html",
                    "content-length": "10000000",
                },
                text="<p>tiny actual</p>",
            )

        client = _mock_client(h)
        with pytest.raises(WebFetchError, match="too large"):
            await fetch_url_as_text(
                "https://example.com/huge", client=client, max_bytes=1_000_000
            )
        await client.aclose()

    async def test_ssrf_guard_runs_before_network(self):
        # Should raise without ever hitting the mock transport.
        async def h(req):  # pragma: no cover — must not be called
            raise AssertionError("Network reached for localhost")

        client = _mock_client(h)
        with pytest.raises(WebFetchError, match="localhost"):
            await fetch_url_as_text("http://localhost:8080/", client=client)
        await client.aclose()

    async def test_network_error_surfaced(self):
        async def h(req):
            raise httpx.ConnectError("connection refused")

        client = _mock_client(h)
        with pytest.raises(WebFetchError, match="Network error"):
            await fetch_url_as_text("https://example.com/", client=client)
        await client.aclose()


# --- Handler MCP shape ---


@pytest.mark.asyncio
class TestHandlerMcpShape:
    async def test_success_returns_payload(self, monkeypatch):
        async def fake_fetch(url, **kw):
            return "<h1>ok</h1>"  # not run through html-to-text in this test

        monkeypatch.setattr("src.tools.web.fetch_url_as_text", fake_fetch)
        result = await _handle_webfetch({"url": "https://example.com"})

        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["url"] == "https://example.com"
        assert payload["char_count"] == len("<h1>ok</h1>")
        assert payload["text"] == "<h1>ok</h1>"

    async def test_url_missing_returns_is_error(self):
        result = await _handle_webfetch({})
        assert result.get("is_error") is True

    async def test_url_non_string_returns_is_error(self):
        result = await _handle_webfetch({"url": 42})
        assert result.get("is_error") is True

    async def test_webfetch_error_returns_is_error(self, monkeypatch):
        from src.tools.web import WebFetchError as _WFE

        async def fake_fetch(url, **kw):
            raise _WFE("simulated failure")

        monkeypatch.setattr("src.tools.web.fetch_url_as_text", fake_fetch)
        result = await _handle_webfetch({"url": "https://example.com"})
        assert result.get("is_error") is True
        assert "simulated failure" in result["content"][0]["text"]


# --- Bridge: provider-shape handler unwraps cleanly ---


@pytest.mark.asyncio
class TestProviderBridge:
    async def test_handler_returns_text_via_unwrap(self, monkeypatch):
        from src.tools import _unwrap_mcp_result

        async def fake_fetch(url, **kw):
            return "hello"

        monkeypatch.setattr("src.tools.web.fetch_url_as_text", fake_fetch)
        mcp = await _handle_webfetch({"url": "https://example.com"})
        text = _unwrap_mcp_result(mcp, tool_name="webfetch")
        payload = json.loads(text)
        assert payload["text"] == "hello"

    async def test_handler_error_raises_via_unwrap(self, monkeypatch):
        from src.tools import _unwrap_mcp_result
        from src.tools.web import WebFetchError as _WFE

        async def fake_fetch(url, **kw):
            raise _WFE("boom")

        monkeypatch.setattr("src.tools.web.fetch_url_as_text", fake_fetch)
        mcp = await _handle_webfetch({"url": "https://example.com"})
        with pytest.raises(RuntimeError, match="boom"):
            _unwrap_mcp_result(mcp, tool_name="webfetch")
