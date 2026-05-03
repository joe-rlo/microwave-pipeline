"""TTS reply tests — voice override detection, markdown stripping,
and the synthesize-then-send path.

Real OpenAI TTS calls are mocked at the aiohttp boundary; we're
verifying the channel wiring (override detection, markdown cleanup
for spoken output, error fallback to text), not the model audio.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.channels.signal import (
    _extract_reply_override,
    _strip_markdown_emphasis,
)
from src.channels.tts import TTSError, synthesize


class TestExtractReplyOverride:
    def test_no_marker_returns_none(self):
        cleaned, override = _extract_reply_override(["hello", "how are you"])
        assert override is None
        assert cleaned == ["hello", "how are you"]

    def test_reply_by_voice_marker(self):
        cleaned, override = _extract_reply_override(
            ["what's the weather, reply by voice please"]
        )
        assert override == "voice"
        # Marker stripped, surrounding text preserved
        assert "what's the weather" in cleaned[0]
        assert "reply by voice" not in cleaned[0].lower()

    def test_reply_by_text_marker(self):
        cleaned, override = _extract_reply_override(
            ["reply by text — what's the weather"]
        )
        assert override == "text"
        assert "what's the weather" in cleaned[0]
        assert "reply by text" not in cleaned[0].lower()

    def test_case_insensitive(self):
        _, override_upper = _extract_reply_override(["REPLY BY VOICE"])
        _, override_mixed = _extract_reply_override(["Reply By Voice"])
        assert override_upper == "voice"
        assert override_mixed == "voice"

    def test_voice_wins_over_text_when_both_present(self):
        """If a user includes both markers (rare), voice wins — asking
        for voice is the more deliberate signal."""
        _, override = _extract_reply_override(
            ["reply by text, no actually reply by voice"]
        )
        assert override == "voice"

    def test_marker_across_multiple_messages(self):
        """Override applies if the marker appears in any of the buffered
        messages (this is how chunked input would deliver it)."""
        cleaned, override = _extract_reply_override(
            ["what's the weather", "reply by voice"]
        )
        assert override == "voice"
        assert cleaned[0] == "what's the weather"
        assert cleaned[1].strip() == ""

    def test_word_boundary_protected(self):
        """We use \\b boundaries so 'replicate' or 'reply by texture'
        don't accidentally trigger."""
        _, override = _extract_reply_override(["please replicate that"])
        assert override is None


class TestStripMarkdownEmphasis:
    def test_bold_double_asterisk(self):
        assert _strip_markdown_emphasis("**hello**") == "hello"

    def test_italic_single_asterisk(self):
        assert _strip_markdown_emphasis("*hello*") == "hello"

    def test_italic_underscore(self):
        assert _strip_markdown_emphasis("_hello_") == "hello"

    def test_inline_code(self):
        assert _strip_markdown_emphasis("`hello`") == "hello"

    def test_combined(self):
        cleaned = _strip_markdown_emphasis(
            "Check **the docs** for `_async_` usage"
        )
        # All decorators removed; the surrounding prose stays
        assert "the docs" in cleaned
        assert "async" in cleaned
        assert "**" not in cleaned and "`" not in cleaned

    def test_preserves_plain_text(self):
        plain = "no formatting here, just words and 5 + 3 = 8."
        assert _strip_markdown_emphasis(plain) == plain


class _FakeResp:
    def __init__(self, status: int, body: bytes | str = b""):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self) -> bytes:
        if isinstance(self._body, bytes):
            return self._body
        return self._body.encode()

    async def text(self) -> str:
        if isinstance(self._body, str):
            return self._body
        return self._body.decode()


class _FakeSession:
    def __init__(self, resp: _FakeResp):
        self.resp = resp
        self.captured_url: str | None = None
        self.captured_payload: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, timeout=None):
        self.captured_url = url
        self.captured_payload = json
        return self.resp


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        with pytest.raises(TTSError, match="OPENAI_API_KEY"):
            await synthesize("hello", api_key="")

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        with pytest.raises(TTSError, match="empty"):
            await synthesize("   ", api_key="sk-test")

    @pytest.mark.asyncio
    async def test_success_returns_bytes(self):
        fake = _FakeSession(_FakeResp(200, body=b"AAC_AUDIO_BYTES"))
        with patch("aiohttp.ClientSession", return_value=fake):
            result = await synthesize("hello world", api_key="sk-test")
        assert result == b"AAC_AUDIO_BYTES"
        assert fake.captured_payload["input"] == "hello world"
        assert fake.captured_payload["voice"]  # uses default voice

    @pytest.mark.asyncio
    async def test_long_text_is_truncated(self):
        """OpenAI rejects very long inputs; trim defensively rather
        than fail the API call mid-reply."""
        fake = _FakeSession(_FakeResp(200, body=b"x"))
        long_text = "a" * 10_000
        with patch("aiohttp.ClientSession", return_value=fake):
            await synthesize(long_text, api_key="sk-test")
        # Truncated to MAX_INPUT_CHARS = 4000
        assert len(fake.captured_payload["input"]) == 4000

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        fake = _FakeSession(_FakeResp(429, body="rate limited"))
        with patch("aiohttp.ClientSession", return_value=fake):
            with pytest.raises(TTSError, match="429"):
                await synthesize("hello", api_key="sk-test")
