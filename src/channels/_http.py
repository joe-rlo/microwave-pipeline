"""Shared aiohttp session factory with certifi-backed SSL trust.

macOS python.org installs ship without a working system trust store —
aiohttp's default `ssl.create_default_context()` ends up with an empty
verify locations bundle, and HTTPS requests to anything (api.openai.com
for Whisper / TTS, signal-cli-rest-api over TLS, instacart's API, …)
fail with `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer
certificate`.

The canonical user-side fix on macOS is running
`/Applications/Python 3.X/Install Certificates.command`, but the
permanent code-side fix is just to point aiohttp at certifi's bundle
explicitly. That works on macOS, Linux, and Docker without any setup
because certifi ships its own copy of Mozilla's trust roots and is
already a transitive dependency of `openai` / `aiohttp`.
"""

from __future__ import annotations

import ssl

import aiohttp
import certifi


def make_session(**kwargs) -> aiohttp.ClientSession:
    """Construct an aiohttp.ClientSession with certifi-backed SSL trust.

    Pass any kwargs you'd normally hand to `aiohttp.ClientSession`; if
    you didn't already supply a `connector`, we'll add a TCPConnector
    with the right SSL context. Callers that need bespoke connector
    settings (custom DNS, proxies) can still supply their own — they're
    on the hook for SSL trust in that case.
    """
    if "connector" not in kwargs:
        kwargs["connector"] = aiohttp.TCPConnector(ssl=ssl_context())
    return aiohttp.ClientSession(**kwargs)


def ssl_context() -> ssl.SSLContext:
    """Return a fresh SSL context with certifi's CA bundle loaded.

    Each call returns a new context — the SDK and aiohttp may mutate
    the context (e.g., set ALPN), so sharing one across multiple
    sessions is risky in practice.
    """
    return ssl.create_default_context(cafile=certifi.where())
