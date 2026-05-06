"""Thinking-nudge tests — phrase rotation, send timestamp capture,
and remote-delete-after-reply.

These exercise the public surface of `_thinking_nudge` and
`_remote_delete` without spinning up the full SignalChannel websocket
loop. The point is to lock the contract: a phrase fires, its
timestamp is captured, and the delete request hits the right URL.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.signal import (
    _THINKING_PHRASES,
    SignalChannel,
)


def _make_channel() -> SignalChannel:
    """Build a SignalChannel skipping the real start() (which opens a
    websocket). All HTTP-touching state mocked."""
    ch = SignalChannel.__new__(SignalChannel)
    ch.rest_url = "http://test"
    ch.phone_number = "+15551234567"
    ch.allowed_senders = set()
    ch.openai_api_key = ""
    ch._session = MagicMock()
    ch._stop_event = asyncio.Event()
    ch._pending = {}
    ch._processing = {}
    ch._queued = {}
    ch._pipeline_started_at = {}
    ch._state_lock = asyncio.Lock()
    return ch


class TestThinkingPhrases:
    def test_phrases_are_microwave_themed(self):
        """Sanity check that the phrase list still has the microwave
        flavor — guards against accidental regression to generic
        'thinking…' if someone trims the list."""
        all_phrases = " ".join(_THINKING_PHRASES).lower()
        # At least a few microwave-specific tokens should appear
        microwave_signals = [
            "microwave", "heat", "thaw", "defrost", "spin", "rotat",
            "magnetron", "popcorn", "carousel", "ding",
        ]
        hits = sum(1 for s in microwave_signals if s in all_phrases)
        assert hits >= 5, (
            f"Expected several microwave-themed phrases; only {hits} signals "
            f"matched in {_THINKING_PHRASES!r}"
        )

    def test_phrases_end_with_ellipsis(self):
        """Format consistency: every phrase ends with `…` so the
        italicized rendering reads as in-progress action."""
        for p in _THINKING_PHRASES:
            assert p.endswith("…"), f"phrase {p!r} should end with `…`"

    def test_no_leading_whitespace_or_underscores(self):
        """The italics wrapper is added at send time. Phrases must
        themselves be clean — no stray markdown artifacts."""
        for p in _THINKING_PHRASES:
            assert not p.startswith("_"), f"phrase {p!r} has leading underscore"
            assert not p.startswith(" "), f"phrase {p!r} has leading whitespace"


class TestThinkingNudgeSend:
    @pytest.mark.asyncio
    async def test_nudge_captures_timestamp(self, monkeypatch):
        """When the nudge fires and _send_text returns a timestamp, the
        holder list gets populated so the caller can delete later."""
        ch = _make_channel()
        # Patch sleep to fire immediately
        monkeypatch.setattr("src.channels.signal.THINKING_NUDGE_SECONDS", 0)
        ch._send_text = AsyncMock(return_value=1234567890123)

        ts_holder: list[int] = []
        await ch._thinking_nudge("+15559999999", ts_holder)

        assert ts_holder == [1234567890123]
        # Sent message uses italic wrapper around a microwave phrase
        ch._send_text.assert_awaited_once()
        sent_text = ch._send_text.await_args.args[1]
        assert sent_text.startswith("_") and sent_text.endswith("_")
        # Body (without the italic markers) is one of our phrases
        assert sent_text[1:-1] in _THINKING_PHRASES

    @pytest.mark.asyncio
    async def test_nudge_cancelled_before_send_no_timestamp(self):
        """Fast turn cancels the nudge during sleep — no message sent,
        no timestamp captured."""
        ch = _make_channel()
        ch._send_text = AsyncMock(return_value=999)
        ts_holder: list[int] = []
        task = asyncio.create_task(ch._thinking_nudge("+1", ts_holder))
        # Cancel before the (real) sleep elapses
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert ts_holder == []
        ch._send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_failure_leaves_holder_empty(self, monkeypatch):
        """_send_text returns None on failure — the holder must stay
        empty so the caller doesn't try to delete a phantom timestamp."""
        ch = _make_channel()
        monkeypatch.setattr("src.channels.signal.THINKING_NUDGE_SECONDS", 0)
        ch._send_text = AsyncMock(return_value=None)
        ts_holder: list[int] = []
        await ch._thinking_nudge("+1", ts_holder)
        assert ts_holder == []


class TestRemoteDelete:
    @pytest.mark.asyncio
    async def test_delete_calls_correct_endpoint(self):
        """signal-cli-rest-api: DELETE /v1/remote-delete/{number}
        with {recipient, timestamp} body."""
        ch = _make_channel()

        captured = {}

        class _MockResp:
            status = 201

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def text(self):
                return ""

        def delete(url, json=None, **kw):
            captured["url"] = url
            captured["json"] = json
            return _MockResp()

        ch._session.delete = delete

        ok = await ch._remote_delete("+15559999999", 1234567890123)
        assert ok is True
        assert "/v1/remote-delete/" in captured["url"]
        # Bot's number is URL-encoded into the path
        assert "%2B15551234567" in captured["url"]
        assert captured["json"] == {
            "recipient": "+15559999999",
            "timestamp": 1234567890123,
        }

    @pytest.mark.asyncio
    async def test_delete_4xx_returns_false_no_raise(self):
        """A failed delete shouldn't bubble up — leaving the placeholder
        visible is no worse than today's behavior, definitely better
        than crashing the reply path."""
        ch = _make_channel()

        class _MockResp:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def text(self):
                return "not found"

        ch._session.delete = lambda *a, **kw: _MockResp()
        ok = await ch._remote_delete("+1", 1)
        assert ok is False

    @pytest.mark.asyncio
    async def test_delete_transport_error_returns_false(self):
        ch = _make_channel()

        def boom(*a, **kw):
            raise RuntimeError("network down")

        ch._session.delete = boom
        ok = await ch._remote_delete("+1", 1)
        assert ok is False
