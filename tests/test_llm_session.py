"""Tests for the multi-turn LLMSession.

We inject a fake LLMProvider rather than using NEARProvider against
httpx.MockTransport — NEARProvider's stream parsing already has its
own coverage (test_near_provider.py). Here we focus on the session's
state machine: connect/reconnect/disconnect, escalation, history
accumulation, memory_context prepending, error handling, and the
exact chunk shape the orchestrator consumes.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from src.llm.provider import (
    Done,
    Error,
    ProviderMessage,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ToolUseStart,
    Usage,
)
from src.llm.session import LLMSession


class _ScriptedProvider:
    """A provider that returns a pre-baked stream and captures the
    most recent request. Used in place of NEARProvider for tests."""

    name = "scripted"
    supports_tools = True
    supports_thinking = True

    def __init__(self, events: list[StreamEvent]):
        self._events = events
        self.last_request: ProviderRequest | None = None
        # Each send() advances through a queue when multiple scripts are loaded.
        self._queue: list[list[StreamEvent]] = []

    def queue(self, events: list[StreamEvent]) -> None:
        """Stage a sequence of events for the NEXT send() call."""
        self._queue.append(events)

    def send(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        self.last_request = req
        events = self._queue.pop(0) if self._queue else self._events

        async def gen() -> AsyncIterator[StreamEvent]:
            for evt in events:
                yield evt

        return gen()

    async def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


# --- Lifecycle ---


@pytest.mark.asyncio
class TestLifecycle:
    async def test_send_before_connect_raises(self):
        provider = _ScriptedProvider([])
        session = LLMSession(model="m", provider=provider)
        with pytest.raises(RuntimeError, match="before connect"):
            async for _ in session.send("hi"):
                pass

    async def test_connect_sets_system_prompt(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("you are helpful")

        async for _ in session.send("hi"):
            pass

        assert provider.last_request.system == "you are helpful"

    async def test_reconnect_resets_history(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("v1")

        # Run a turn so history has a user + assistant message
        async for _ in session.send("first"):
            pass
        assert len(session.history) == 2  # user + assistant

        # Reconnect — history cleared, prompt swapped
        await session.reconnect("v2")
        assert session.history == []

        # Next send sees the new system prompt
        async for _ in session.send("again"):
            pass
        assert provider.last_request.system == "v2"

    async def test_disconnect_clears_state(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")
        async for _ in session.send("hi"):
            pass
        await session.disconnect()

        with pytest.raises(RuntimeError, match="before connect"):
            async for _ in session.send("hello again"):
                pass


# --- Chunk shape ---


@pytest.mark.asyncio
class TestChunkShape:
    async def test_text_deltas_become_delta_chunks(self):
        provider = _ScriptedProvider([
            TextDelta(text="hel"),
            TextDelta(text="lo"),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        chunks = [c async for c in session.send("hi")]
        deltas = [c for c in chunks if c["type"] == "delta"]
        assert [c["text"] for c in deltas] == ["hel", "lo"]

    async def test_terminal_result_chunk_emitted(self):
        provider = _ScriptedProvider([
            TextDelta(text="full text"),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        chunks = [c async for c in session.send("hi")]
        # Last chunk is always a result with assembled text
        assert chunks[-1] == {"type": "result", "text": "full text"}

    async def test_tool_use_visibility_chunk(self):
        provider = _ScriptedProvider([
            TextDelta(text="calling tool"),
            ToolUseStart(id="t1", name="search"),
            Done(stop_reason="tool_use"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        chunks = [c async for c in session.send("hi")]
        tool_chunks = [c for c in chunks if c["type"] == "tool_use"]
        # Phase C.1: surface tool_use for typing-indicator visibility,
        # but no round-trip yet (that's C.2's tool_loop).
        assert tool_chunks == [{"type": "tool_use", "name": "search"}]

    async def test_usage_events_dropped(self):
        # Usage isn't part of the LLMClient contract; consumers shouldn't see it.
        provider = _ScriptedProvider([
            TextDelta(text="hi"),
            Usage(input_tokens=10, output_tokens=2, is_final=True),
            Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        chunks = [c async for c in session.send("x")]
        assert all(c["type"] in ("delta", "result") for c in chunks)


# --- History accumulation ---


@pytest.mark.asyncio
class TestHistory:
    async def test_history_grows_across_turns(self):
        provider = _ScriptedProvider([
            TextDelta(text="r1"), Done(stop_reason="end_turn"),
        ])
        provider.queue([TextDelta(text="r2"), Done(stop_reason="end_turn")])
        # _ScriptedProvider's `events` is used when queue is empty; we
        # consume the queued response first, then fall back to the original.
        # Actually with the current shape, queue is checked first.
        # Reload queue so two turns each have their own script:
        provider._queue = [
            [TextDelta(text="r1"), Done(stop_reason="end_turn")],
            [TextDelta(text="r2"), Done(stop_reason="end_turn")],
        ]
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        async for _ in session.send("q1"):
            pass
        async for _ in session.send("q2"):
            pass

        # After two turns: 2 user + 2 assistant = 4 messages
        assert len(session.history) == 4
        assert session.history[0].role == "user"
        # First user message includes "q1"
        assert "q1" in session.history[0].content
        assert session.history[1].role == "assistant"
        assert session.history[1].content == "r1"
        assert "q2" in session.history[2].content
        assert session.history[3].content == "r2"

    async def test_memory_context_prepended(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"), Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        async for _ in session.send("question", memory_context="fact A"):
            pass

        user_msg = session.history[0].content
        assert "[Relevant memory context]" in user_msg
        assert "fact A" in user_msg
        assert "question" in user_msg

    async def test_empty_response_doesnt_pollute_history(self):
        # If the model returns nothing (just Done with no deltas), the
        # assistant turn must not be appended — otherwise the next turn's
        # context contains an empty assistant message that confuses the
        # model.
        provider = _ScriptedProvider([Done(stop_reason="end_turn")])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        async for _ in session.send("hi"):
            pass

        # Only the user message landed; no empty assistant turn
        assert len(session.history) == 1
        assert session.history[0].role == "user"


# --- Escalation ---


@pytest.mark.asyncio
class TestEscalation:
    async def test_escalate_swaps_model_and_sets_budget(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"), Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="sonnet", provider=provider)
        await session.connect("p")

        await session.escalate("opus", "high")
        async for _ in session.send("complex query"):
            pass

        assert provider.last_request.model == "opus"
        assert provider.last_request.thinking_budget == 32_000

    async def test_de_escalate_restores_base_model(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"), Done(stop_reason="end_turn"),
        ])
        # Two scripted responses for two turns
        provider._queue = [
            [TextDelta(text="r1"), Done(stop_reason="end_turn")],
            [TextDelta(text="r2"), Done(stop_reason="end_turn")],
        ]
        session = LLMSession(model="sonnet", provider=provider)
        await session.connect("p")

        await session.escalate("opus", "max")
        async for _ in session.send("hard"):
            pass
        assert provider.last_request.model == "opus"
        assert provider.last_request.thinking_budget == 64_000

        await session.de_escalate()
        async for _ in session.send("easy"):
            pass
        assert provider.last_request.model == "sonnet"
        assert provider.last_request.thinking_budget is None

    async def test_unknown_effort_yields_no_budget(self):
        provider = _ScriptedProvider([
            TextDelta(text="ok"), Done(stop_reason="end_turn"),
        ])
        session = LLMSession(model="s", provider=provider)
        await session.connect("p")

        await session.escalate("opus", "nonsense_value")
        async for _ in session.send("q"):
            pass

        assert provider.last_request.model == "opus"
        # Unknown effort → None budget (safer than guessing a default)
        assert provider.last_request.thinking_budget is None


# --- Error handling ---


@pytest.mark.asyncio
class TestErrors:
    async def test_error_with_no_text_raises(self):
        provider = _ScriptedProvider([
            Error(message="rate limited", status=429, retryable=True),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        with pytest.raises(RuntimeError, match="rate limited"):
            async for _ in session.send("hi"):
                pass

    async def test_error_after_partial_text_does_not_raise(self):
        # Sometimes the stream errors mid-flight. We've already streamed
        # text to the user, so raising would double-emit. Behavior: emit
        # the partial response and complete cleanly. The orchestrator can
        # decide whether to retry based on quality.
        provider = _ScriptedProvider([
            TextDelta(text="part"),
            Error(message="dropped", status=500, retryable=True),
        ])
        session = LLMSession(model="m", provider=provider)
        await session.connect("p")

        chunks = [c async for c in session.send("hi")]
        # No exception — the user already saw "part"
        deltas = [c for c in chunks if c["type"] == "delta"]
        result = [c for c in chunks if c["type"] == "result"][0]
        assert deltas[0]["text"] == "part"
        assert result["text"] == "part"


# --- Provider factory ---


class TestProviderFactory:
    def test_missing_near_key_raises(self, monkeypatch):
        monkeypatch.delenv("NEAR_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NEAR_API_KEY"):
            LLMSession(model="m")  # default factory runs

    def test_near_key_present_constructs(self, monkeypatch):
        monkeypatch.setenv("NEAR_API_KEY", "k")
        # Construction shouldn't fail; we don't actually call send().
        session = LLMSession(model="m")
        assert session.model == "m"
