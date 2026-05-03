"""Tests for the debounce buffer, typing-indicator awareness,
streaming-interrupt, and adaptive-response-length features."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.channels import signal as signal_mod
from src.channels.signal import SignalChannel, _format_quote_context
from src.pipeline.assembly import assemble
from src.session.models import SearchResult


# --- Adaptive response length (assembly layer) ---


class _FakeMemoryStore:
    def assemble_stable_context(self, channel=None, bible_path=None):
        return ""


class _FakeMemoryIndex:
    def get_promotion_candidates(self, min_retrievals=3):
        return []


def _ctx(complexity: str) -> str:
    return assemble(
        SearchResult(fragments=[]),
        _FakeMemoryStore(),
        _FakeMemoryIndex(),
        channel=None,
        complexity=complexity,
    ).memory_context


class TestAdaptiveLength:
    def test_simple_adds_brief_hint(self):
        ctx = _ctx("simple")
        assert "[Response length]" in ctx
        # Match either "brief" or "brevity" — the hint exists in some form.
        assert "brief" in ctx.lower() or "brevity" in ctx.lower()

    def test_complex_adds_depth_hint(self):
        ctx = _ctx("complex")
        assert "[Response length]" in ctx
        assert "depth" in ctx.lower() or "develop" in ctx.lower()

    def test_moderate_adds_no_hint(self):
        ctx = _ctx("moderate")
        assert "[Response length]" not in ctx

    def test_unknown_complexity_treated_as_moderate(self):
        ctx = _ctx("garbage-value")
        assert "[Response length]" not in ctx

    def test_length_hint_is_last(self):
        ctx = _ctx("simple")
        # Must come after channel file-output instructions so it has
        # higher recency in the prompt.
        assert ctx.rfind("[Response length]") > ctx.rfind("[File output")


# --- Signal channel buffering / interrupt ---


@pytest.fixture
def fast_channel(monkeypatch):
    """A SignalChannel with tightened timers and stubbed I/O.

    Returns a channel whose `invocations` list captures every call that
    would have hit the pipeline. Use this to test the buffer layer in
    isolation.
    """
    # Tighten timing so tests stay fast. Real values are 2.5s / 60s.
    monkeypatch.setattr(signal_mod, "DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(signal_mod, "MAX_HOLD_SECONDS", 0.30)

    ch = SignalChannel(
        orchestrator=None,
        rest_url="http://test",
        phone_number="+1",
        allowed_senders=["+2"],
    )

    ch.invocations: list[tuple[str, str]] = []
    ch.process_durations: list[float] = []
    ch.cancelled_count = 0

    async def _fake_process(text: str, source: str, images=None) -> None:
        ch.invocations.append((source, text))
        # Simulate a slow pipeline so cancellation tests have something
        # to cancel. Tests that need fast finishing override this.
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            ch.cancelled_count += 1
            raise

    async def _noop(*args, **kw):
        pass

    ch._process_and_respond = _fake_process
    ch._send_text = _noop
    ch._send_read_receipt = _noop
    ch._send_typing = _noop
    ch._download_attachment = _noop
    ch._transcribe = _noop

    return ch


def _msg_event(text: str, source: str = "+2", ts: int = 1) -> dict:
    return {
        "envelope": {
            "source": source,
            "sourceNumber": source,
            "timestamp": ts,
            "dataMessage": {"message": text, "timestamp": ts},
        }
    }


def _typing_event(action: str, source: str = "+2") -> dict:
    return {
        "envelope": {
            "source": source,
            "sourceNumber": source,
            "typingMessage": {"action": action, "timestamp": int(time.time() * 1000)},
        }
    }


class TestDebounceBuffer:
    @pytest.mark.asyncio
    async def test_single_message_fires_once(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("hello"))
        await asyncio.sleep(0.15)  # > DEBOUNCE_SECONDS

        assert len(fast_channel.invocations) == 1
        source, text = fast_channel.invocations[0]
        assert source == "+2"
        assert "hello" in text

    @pytest.mark.asyncio
    async def test_back_to_back_combines(self, fast_channel):
        # Two messages within the debounce window must coalesce into one
        # pipeline call. This is the bug-fix that motivated the whole feature.
        await fast_channel._handle_incoming(_msg_event("first"))
        await asyncio.sleep(0.02)
        await fast_channel._handle_incoming(_msg_event("second", ts=2))
        await asyncio.sleep(0.02)
        await fast_channel._handle_incoming(_msg_event("third", ts=3))
        await asyncio.sleep(0.20)

        assert len(fast_channel.invocations) == 1
        _, combined = fast_channel.invocations[0]
        assert "first" in combined
        assert "second" in combined
        assert "third" in combined

    @pytest.mark.asyncio
    async def test_messages_outside_window_fire_separately(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("alpha"))
        await asyncio.sleep(0.20)  # full debounce + slack
        await fast_channel._handle_incoming(_msg_event("beta", ts=2))
        await asyncio.sleep(0.20)

        assert len(fast_channel.invocations) == 2
        assert "alpha" in fast_channel.invocations[0][1]
        assert "beta" in fast_channel.invocations[1][1]

    @pytest.mark.asyncio
    async def test_disallowed_sender_does_not_buffer(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("hi", source="+999"))
        await asyncio.sleep(0.20)
        assert fast_channel.invocations == []
        assert fast_channel._pending == {}


class TestTypingIndicator:
    @pytest.mark.asyncio
    async def test_typing_started_pauses_debounce(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("composing"))
        # User starts typing again before debounce fires
        await asyncio.sleep(0.02)
        await fast_channel._handle_incoming(_typing_event("STARTED"))
        # Wait past where debounce WOULD have fired
        await asyncio.sleep(0.15)
        assert fast_channel.invocations == []  # still pending — typing pauses it

        # Now stop typing — fresh debounce window starts
        await fast_channel._handle_incoming(_typing_event("STOPPED"))
        await asyncio.sleep(0.15)
        assert len(fast_channel.invocations) == 1

    @pytest.mark.asyncio
    async def test_typing_with_no_buffer_is_noop(self, fast_channel):
        # Typing without any pending message must not crash or pre-arm a buffer
        await fast_channel._handle_incoming(_typing_event("STARTED"))
        await asyncio.sleep(0.10)
        assert fast_channel._pending == {}
        assert fast_channel.invocations == []

    @pytest.mark.asyncio
    async def test_max_hold_fires_when_typing_never_stops(self, fast_channel):
        # User starts typing forever. Max-hold should still drain the buffer.
        await fast_channel._handle_incoming(_msg_event("trapped"))
        await fast_channel._handle_incoming(_typing_event("STARTED"))
        # MAX_HOLD_SECONDS is 0.30 in this fixture
        await asyncio.sleep(0.40)

        assert len(fast_channel.invocations) == 1
        assert "trapped" in fast_channel.invocations[0][1]


class TestStreamingInterrupt:
    @pytest.mark.asyncio
    async def test_new_message_cancels_in_flight(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("first run"))
        await asyncio.sleep(0.10)  # debounce fires; processing starts
        # Processing is now sleeping 0.5s in the fake. Send a new message.
        assert "+2" in fast_channel._processing  # sanity

        await fast_channel._handle_incoming(_msg_event("interrupt", ts=2))
        await asyncio.sleep(0.05)
        # First task should now be cancelled
        assert fast_channel.cancelled_count >= 1

        await asyncio.sleep(0.20)  # let the second debounce fire + process
        # The interrupting message is buffered and runs as its own turn.
        # We should see at least one completed run with "interrupt" in it.
        completed = [t for _, t in fast_channel.invocations]
        # Find any invocation containing "interrupt" — the cancelled first run
        # may or may not have been recorded depending on timing, but the
        # new combined input MUST land.
        assert any("interrupt" in t for t in completed)

    @pytest.mark.asyncio
    async def test_processing_slot_is_cleared_after_completion(self, fast_channel):
        await fast_channel._handle_incoming(_msg_event("solo"))
        await asyncio.sleep(0.10)  # debounce fires
        # Processing in flight
        assert "+2" in fast_channel._processing

        await asyncio.sleep(0.60)  # let the fake 0.5s processing finish
        # _safe_process's finally block should have removed the slot.
        assert "+2" not in fast_channel._processing


# --- Quote / swipe-to-reply context ---


class TestFormatQuoteContext:
    def test_empty_text_returns_empty(self):
        assert _format_quote_context({"text": ""}, bot_number="+1") == ""
        assert _format_quote_context({}, bot_number="+1") == ""

    def test_basic_quote_prefix(self):
        out = _format_quote_context(
            {"text": "Note 1 — hot take\nNote 2 — essay", "author": "+999"},
            bot_number="+1",
        )
        assert "[Replying to an earlier message]" in out
        assert "> Note 1 — hot take" in out
        assert "> Note 2 — essay" in out  # multiline quotes get prefixed line by line

    def test_self_authored_quote_labeled_distinctly(self):
        # When the bot quoted its own earlier reply, we want the LLM to
        # know it was its own output — different attribution than a
        # quoted human message.
        out = _format_quote_context(
            {"text": "Here are 3 notes...", "author": "+1"},
            bot_number="+1",
        )
        assert "your earlier reply" in out

    def test_long_quote_truncated(self):
        long = "x" * 1000
        out = _format_quote_context({"text": long, "author": "+999"}, bot_number="+1")
        # Quoted body should be capped, not pasted whole
        assert len(out) < 700
        assert out.endswith("…")

    def test_blank_author_is_treated_as_other(self):
        out = _format_quote_context(
            {"text": "hi", "author": ""}, bot_number="+1"
        )
        assert "an earlier message" in out


class TestQuoteRouting:
    """End-to-end: a Signal event with a quote produces an enriched
    pipeline input. We use the same fast_channel fixture as the
    debounce tests."""

    @pytest.mark.asyncio
    async def test_quote_prepended_to_user_text(self, fast_channel):
        event = {
            "envelope": {
                "source": "+2",
                "sourceNumber": "+2",
                "timestamp": 1,
                "dataMessage": {
                    "message": "scrap the second one",
                    "timestamp": 1,
                    "quote": {
                        "text": "Note 1 — hot take\nNote 2 — essay\nNote 3 — field note",
                        "author": "+1",  # the bot itself in this fixture
                    },
                },
            }
        }
        await fast_channel._handle_incoming(event)
        await asyncio.sleep(0.20)  # > debounce

        assert len(fast_channel.invocations) == 1
        _, combined = fast_channel.invocations[0]
        # Quote prefix arrived
        assert "Replying to your earlier reply" in combined
        assert "Note 2 — essay" in combined
        # User's actual text still present
        assert "scrap the second one" in combined
        # Order matters: quote context goes first so the user message
        # reads as the followup, not the lead
        assert combined.index("Replying to") < combined.index("scrap the second one")

    @pytest.mark.asyncio
    async def test_quote_only_no_user_text(self, fast_channel):
        # Edge case: user swipes to reply but sends an empty message
        # (Signal lets you do this with just an emoji reaction sometimes).
        # The quote context alone is enough to keep the pipeline alive.
        event = {
            "envelope": {
                "source": "+2",
                "sourceNumber": "+2",
                "timestamp": 1,
                "dataMessage": {
                    "message": "👍",
                    "timestamp": 1,
                    "quote": {"text": "your earlier suggestion", "author": "+1"},
                },
            }
        }
        await fast_channel._handle_incoming(event)
        await asyncio.sleep(0.20)

        assert len(fast_channel.invocations) == 1
        _, combined = fast_channel.invocations[0]
        assert "your earlier suggestion" in combined
        assert "👍" in combined

    @pytest.mark.asyncio
    async def test_no_quote_means_no_prefix(self, fast_channel):
        # Sanity check that the non-reply path stays untouched.
        event = {
            "envelope": {
                "source": "+2",
                "sourceNumber": "+2",
                "timestamp": 1,
                "dataMessage": {"message": "plain hello", "timestamp": 1},
            }
        }
        await fast_channel._handle_incoming(event)
        await asyncio.sleep(0.20)

        _, combined = fast_channel.invocations[0]
        assert "Replying to" not in combined
        assert combined.strip() == "plain hello"
