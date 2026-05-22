"""Native web tools — replaces Agent SDK built-ins for the new provider path.

Phase C.4 introduces the first one: `webfetch`. It mirrors what
Claude Code's `WebFetch` does (fetch URL, return content) but without
the Anthropic-internal sub-LLM step — the calling model just gets the
extracted text and decides what to do with it.

What `webfetch` does:
- HTTP GET via httpx with reasonable timeouts and a polite User-Agent
- Caps response size so a runaway page can't blow the context budget
- Strips HTML tags to plain text (no fancy main-content extraction —
  the LLM is smart enough to read around boilerplate, and a
  library-grade extractor like trafilatura is too heavy a dep for what
  we'd gain)
- Returns text up to a cap; bigger pages get truncated with a clear marker

What it does NOT do:
- No JavaScript rendering. Static HTML only. Single-page apps with
  client-rendered content will return mostly empty. That's the same
  limitation Claude Code's built-in WebFetch has.
- No authentication. Cookies, headers, OAuth — out of scope.
  Use a dedicated integration tool for authenticated APIs.
- No POST or other methods. Read-only by design.
- No internal/private URLs. Refuses localhost, 127.0.0.1, link-local,
  RFC1918 ranges (10/8, 172.16/12, 192.168/16) so a prompt injection
  can't pivot through the bot to local services.

Why it's not gated by an env key:
- No external API required (httpx GET to public web)
- No per-user secrets
- The SSRF guard is the only real safety surface; it always runs

This tool is registered automatically when present. To disable, users
can set `WEB_TOOLS_DISABLED=true` in env — escape hatch for paranoid
deployments or testing isolation.
"""

from __future__ import annotations

import html as html_module
import ipaddress
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)


# Practical limit on returned text. ~50K chars = ~12K tokens, enough for
# a long blog post or doc page, fits comfortably in any model's context.
DEFAULT_MAX_CHARS = 50_000

# Per-request timeout. Page loads should be quick; if a host is slow,
# we'd rather fail fast and let the model react than hang the whole turn.
DEFAULT_TIMEOUT_SECONDS = 15.0

# Hard cap on bytes pulled from the network before we stop reading.
# A page bigger than this is almost certainly not what we want (and
# might be a streaming endpoint that never ends).
DEFAULT_MAX_BYTES = 2_000_000

USER_AGENT = "MicrowaveOS-WebFetch/0.1 (+https://0x4a6f65.com)"

# Tools the model sees in its system context. Wired into the catalog
# by the registry.
WEB_TOOL_DOCS = """\
**webfetch** — Retrieve the contents of a public web page as plain text.

When to use:
- The user shares a URL and asks about its content ("what does this say", "summarize this").
- You need information from a specific public page (docs, blog, news, GitHub README).

When NOT to use:
- For search ("find pages about X") — webfetch needs a specific URL.
- For authenticated content (logged-in pages, private APIs) — won't work.
- For interactive / JS-rendered apps — returns empty or boilerplate.

How to use:
- `url`: the full URL including scheme. Required.
- `max_chars`: optional cap on returned text (default ~50K).
- Returns the page text, HTML stripped to a basic plain-text form.
- A page bigger than max_chars is truncated with "…[truncated]" at the end.

If the page is dynamic / requires JS, the response will be mostly empty.
Tell the user that page can't be fetched as-is.
"""

WEBFETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Full URL to fetch (http or https). Required.",
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1000,
            "maximum": 200_000,
            "description": (
                f"Maximum text length to return. Defaults to {DEFAULT_MAX_CHARS}. "
                "Beyond max_chars the result is truncated with an explicit marker."
            ),
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}


# --- Handler (shared by SDK + provider paths) ------------------------------


async def fetch_url_as_text(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Fetch a URL and return its text content.

    Pure function. Tools wrap this to produce MCP / provider-shape
    responses; tests call it directly. Raises `WebFetchError` on any
    failure with a useful message.

    `client` is injected for tests; production opens one per call.
    """
    _validate_url(url)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=timeout / 3, read=timeout, write=5.0, pool=5.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            max_redirects=5,
        )

    try:
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            raise WebFetchError(f"Network error fetching {url}: {e}") from e

        if resp.status_code >= 400:
            raise WebFetchError(
                f"HTTP {resp.status_code} fetching {url}",
                status=resp.status_code,
            )

        content_type = resp.headers.get("content-type", "").lower()
        # Allow text/* and a handful of structured-text types. Binary
        # content (images, PDFs, octet-stream) gets rejected — the LLM
        # can't usefully consume raw bytes here.
        if not _is_text_content_type(content_type):
            raise WebFetchError(
                f"Cannot fetch {url}: content-type {content_type!r} is not text"
            )

        # httpx eagerly reads the full body by default. Bound it by checking
        # the content-length header first; if missing, accept up to max_bytes
        # of body and let httpx complain if the wire delivers more.
        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise WebFetchError(
                f"Page too large ({content_length} bytes > {max_bytes} cap); "
                "consider asking the user to paste the relevant section."
            )

        body = resp.text
        # httpx already decoded; check size after decoding to be safe.
        if len(body.encode("utf-8", errors="ignore")) > max_bytes:
            body = body[: max_bytes]

        text = _html_to_text(body) if "html" in content_type else body
        text = _normalize_whitespace(text)

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n…[truncated]"

        return text
    finally:
        if own_client:
            await client.aclose()


class WebFetchError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


# --- URL validation (SSRF guard) -------------------------------------------


def _validate_url(url: str) -> None:
    """Raise WebFetchError if `url` shouldn't be fetched.

    The bot can be prompted to fetch URLs; without this check, a malicious
    message could pivot through it to local services (localhost:6379,
    internal admin panels, etc.). Block all private and link-local
    ranges. Hostnames are resolved at fetch time by the OS, but we can
    short-circuit literal IPs and known-suspicious hostnames cheaply.
    """
    if not isinstance(url, str) or not url:
        raise WebFetchError("url must be a non-empty string")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebFetchError(
            f"url scheme {parsed.scheme!r} not allowed (only http/https)"
        )
    if not parsed.hostname:
        raise WebFetchError(f"url has no hostname: {url!r}")

    host = parsed.hostname.lower()
    # Hostname literals
    if host in ("localhost", "localhost.localdomain"):
        raise WebFetchError("Refusing to fetch localhost")
    if host.endswith(".internal") or host.endswith(".local"):
        raise WebFetchError(f"Refusing to fetch internal-only hostname {host!r}")
    # IP literals
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified):
        raise WebFetchError(f"Refusing to fetch private/loopback address {host!r}")


# --- HTML → text -----------------------------------------------------------


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text.

    Deliberately simple: strip <script>/<style> blocks, then strip all
    remaining tags, then unescape entities. We're not trying to produce
    pretty markdown — the LLM is the parser; we just need to remove
    structural noise.
    """
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html_module.unescape(cleaned)
    return cleaned


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and excessive blank lines for token efficiency."""
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _is_text_content_type(content_type: str) -> bool:
    """Whitelist text-like content types we'll attempt to read.

    HTML, plain text, markdown, XML, JSON, YAML all decode cleanly to
    text we can hand to the model. Binary types (images, PDFs, etc.) are
    rejected — they'd surface as garbled bytes.
    """
    if not content_type:
        # Many servers omit Content-Type entirely for HTML; lean permissive.
        return True
    primary = content_type.split(";")[0].strip().lower()
    if primary.startswith("text/"):
        return True
    if primary in (
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/x-yaml",
        "application/yaml",
        "application/atom+xml",
        "application/rss+xml",
        "application/ld+json",
    ):
        return True
    return False


# --- SDK-shape registration ------------------------------------------------


def build_webfetch_sdk_tools() -> list:
    """SdkMcpTool wrapper. Returns [] if the SDK isn't installed."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    @tool(
        name="webfetch",
        description=(
            "Fetch a public web page and return its content as plain text. "
            "Use when the user shares a URL or you need information from a "
            "specific public page. Won't work for authenticated, "
            "JS-rendered, or non-text content."
        ),
        input_schema=WEBFETCH_SCHEMA,
    )
    async def webfetch(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_webfetch(args)

    return [webfetch]


async def _handle_webfetch(args: dict[str, Any]) -> dict[str, Any]:
    """Shared handler — returns MCP shape, used by both SDK and provider paths.

    The provider bridge in src/tools/__init__.py unwraps `{"is_error": True}`
    into a raised exception so the LLMSession tool loop reports it as
    is_error=True to the model. So both paths get the same semantics.
    """
    url = args.get("url")
    max_chars = args.get("max_chars") or DEFAULT_MAX_CHARS

    if not isinstance(url, str) or not url:
        return _error("url must be a non-empty string")

    try:
        text = await fetch_url_as_text(url, max_chars=max_chars)
    except WebFetchError as e:
        log.info("webfetch failed for %s: %s", url, e)
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected webfetch failure for %s", url)
        return _error(f"Unexpected error: {e}")

    payload = {"url": url, "char_count": len(text), "text": text}
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }


def _error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }
