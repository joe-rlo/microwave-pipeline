"""Tests for the LLM provider contract.

This is shape-only — no I/O, no adapters. The protocol is the contract,
and these tests pin down what the contract guarantees:

- ContentBlock variants enforce single-typed-field invariant.
- ProviderMessage enforces tool-role linkage requirement.
- StreamEvent union members discriminate cleanly on `.type`.
- ProviderRequest defaults are sensible (stream=True, max_tokens set).

If any of these tests fail, the protocol has drifted in a way that
would break adapters silently. That's exactly what type tests are for.
"""

from __future__ import annotations

from typing import AsyncIterator, get_args

import pytest

from src.llm.provider import (
    ContentBlock,
    Done,
    Error,
    ImageBlock,
    LLMProvider,
    ProviderMessage,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolDefinition,
    ToolUse,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    ToolResult,
    Usage,
)


# --- ContentBlock invariants ---


class TestContentBlock:
    def test_text_block_constructs(self):
        b = ContentBlock.of_text("hello")
        assert b.type == "text"
        assert b.text == "hello"
        assert b.tool_use is None

    def test_tool_use_block_constructs(self):
        tu = ToolUse(id="toolu_1", name="x", arguments={"k": 1})
        b = ContentBlock.of_tool_use(tu)
        assert b.type == "tool_use"
        assert b.tool_use is tu

    def test_tool_result_block_constructs(self):
        tr = ToolResult(tool_use_id="toolu_1", content="ok")
        b = ContentBlock.of_tool_result(tr)
        assert b.type == "tool_result"
        assert b.tool_result is tr
        assert b.tool_result.is_error is False  # default

    def test_image_block_constructs(self):
        img = ImageBlock(media_type="image/png", data_base64="aGVsbG8=")
        b = ContentBlock.of_image(img)
        assert b.type == "image"
        assert b.image is img

    def test_rejects_zero_fields_set(self):
        # All typed fields None — must raise.
        with pytest.raises(ValueError, match="exactly one"):
            ContentBlock(type="text")

    def test_rejects_multiple_fields_set(self):
        tu = ToolUse(id="x", name="y", arguments={})
        with pytest.raises(ValueError, match="exactly one"):
            ContentBlock(type="text", text="hi", tool_use=tu)

    def test_rejects_mismatched_type_and_field(self):
        # type says "text" but tool_use is the set field
        tu = ToolUse(id="x", name="y", arguments={})
        with pytest.raises(ValueError, match="text.*tool_use"):
            ContentBlock(type="text", tool_use=tu)

    def test_blocks_are_hashable(self):
        # frozen=True dataclasses with hashable members should hash.
        # This matters because callers may dedupe lists of blocks.
        a = ContentBlock.of_text("a")
        b = ContentBlock.of_text("a")
        assert hash(a) == hash(b)
        assert a == b


# --- ProviderMessage invariants ---


class TestProviderMessage:
    def test_simple_user_message(self):
        m = ProviderMessage(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert m.tool_call_id is None

    def test_assistant_with_blocks(self):
        blocks = [
            ContentBlock.of_text("calling tool"),
            ContentBlock.of_tool_use(
                ToolUse(id="toolu_1", name="t", arguments={"q": 1})
            ),
        ]
        m = ProviderMessage(role="assistant", content=blocks)
        assert m.role == "assistant"
        assert isinstance(m.content, list)
        assert len(m.content) == 2

    def test_tool_role_with_tool_call_id(self):
        # OpenAI-shaped: top-level tool_call_id
        m = ProviderMessage(
            role="tool", content="result text", tool_call_id="call_1"
        )
        assert m.role == "tool"
        assert m.tool_call_id == "call_1"

    def test_tool_role_with_embedded_result_block(self):
        # Anthropic-shaped: tool_result block carries the id
        m = ProviderMessage(
            role="tool",
            content=[
                ContentBlock.of_tool_result(
                    ToolResult(tool_use_id="toolu_1", content="result")
                )
            ],
        )
        assert m.role == "tool"
        # No top-level id needed when the block has it
        assert m.tool_call_id is None

    def test_tool_role_without_linkage_raises(self):
        with pytest.raises(ValueError, match="tool_call_id or an embedded"):
            ProviderMessage(role="tool", content="orphan result")


# --- ProviderRequest defaults ---


class TestProviderRequest:
    def test_minimal_request(self):
        req = ProviderRequest(
            model="claude-sonnet-4-6",
            messages=[ProviderMessage(role="user", content="hi")],
        )
        assert req.stream is True
        assert req.max_tokens == 4096
        assert req.tools == []
        assert req.metadata == {}
        assert req.thinking_budget is None

    def test_tools_default_is_fresh_list(self):
        # Guard against the classic mutable-default-arg bug. Each
        # request should get its own list, not share one across all
        # instances.
        a = ProviderRequest(model="x", messages=[])
        b = ProviderRequest(model="x", messages=[])
        # Both should be empty lists; mutating one must NOT mutate the
        # other (this would fail catastrophically if the dataclass used
        # `tools: list = []` instead of field(default_factory=list)).
        # We can't actually mutate because the dataclass is frozen,
        # but the identity check confirms a fresh list per instance.
        assert a.tools is not b.tools

    def test_thinking_budget_propagates(self):
        req = ProviderRequest(
            model="claude-opus-4-7", messages=[], thinking_budget=32_000
        )
        assert req.thinking_budget == 32_000


# --- StreamEvent discrimination ---


class TestStreamEvents:
    def test_text_delta_type(self):
        e = TextDelta(text="hi")
        assert e.type == "text_delta"

    def test_tool_use_start_type(self):
        e = ToolUseStart(id="toolu_1", name="x")
        assert e.type == "tool_use_start"

    def test_tool_use_delta_type(self):
        e = ToolUseDelta(id="toolu_1", arguments_delta='{"k":')
        assert e.type == "tool_use_delta"

    def test_tool_use_end_type(self):
        e = ToolUseEnd(id="toolu_1", name="x", arguments={"k": 1})
        assert e.type == "tool_use_end"

    def test_thinking_delta_type(self):
        e = ThinkingDelta(text="…")
        assert e.type == "thinking_delta"

    def test_usage_type_and_defaults(self):
        e = Usage(input_tokens=100, output_tokens=50)
        assert e.type == "usage"
        assert e.cache_creation_tokens == 0
        assert e.cache_read_tokens == 0
        assert e.is_final is False

    def test_done_type(self):
        e = Done(stop_reason="end_turn")
        assert e.type == "done"

    def test_error_type_and_defaults(self):
        e = Error(message="boom")
        assert e.type == "error"
        assert e.status is None
        assert e.retryable is False

    def test_dispatch_by_type(self):
        # The orchestrator pattern-matches on event.type. Confirm the
        # discriminator works for every event in the union.
        events: list[StreamEvent] = [
            TextDelta(text="a"),
            ToolUseStart(id="t1", name="n"),
            ToolUseDelta(id="t1", arguments_delta="{}"),
            ToolUseEnd(id="t1", name="n", arguments={}),
            ThinkingDelta(text="…"),
            Usage(input_tokens=1, output_tokens=1, is_final=True),
            Done(stop_reason="end_turn"),
            Error(message="x"),
        ]
        seen: set[str] = set()
        for e in events:
            seen.add(e.type)
        assert seen == {
            "text_delta", "tool_use_start", "tool_use_delta",
            "tool_use_end", "thinking_delta", "usage", "done", "error",
        }

    def test_stream_event_union_covers_all_members(self):
        # Pin down which concrete types are in the union — if someone
        # adds a new event class without updating StreamEvent, this fails.
        members = set(get_args(StreamEvent))
        assert members == {
            TextDelta, ToolUseStart, ToolUseDelta, ToolUseEnd,
            ThinkingDelta, Usage, Done, Error,
        }


# --- Protocol conformance ---


class _FakeProvider:
    """A minimal adapter just to confirm the Protocol shape compiles
    against real code. Not used elsewhere — this is the contract test."""

    name = "fake"
    supports_tools = True
    supports_thinking = False

    async def send(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="hello")
        yield Done(stop_reason="end_turn")

    async def estimate_tokens(self, text: str) -> int:
        # ~4 chars per token approximation
        return max(1, len(text) // 4)


class TestProtocolConformance:
    def test_fake_provider_satisfies_protocol(self):
        # Protocol checks are structural — assigning to an LLMProvider-typed
        # variable will fail at type-check time if the shape is wrong.
        # At runtime, this just confirms construction works.
        p: LLMProvider = _FakeProvider()
        assert p.name == "fake"
        assert p.supports_tools is True

    @pytest.mark.asyncio
    async def test_fake_provider_streams(self):
        p = _FakeProvider()
        req = ProviderRequest(model="m", messages=[])
        events = []
        async for evt in p.send(req):
            events.append(evt)
        assert len(events) == 2
        assert events[0].type == "text_delta"
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        p = _FakeProvider()
        n = await p.estimate_tokens("a" * 40)
        # 40 chars / 4 ≈ 10 tokens
        assert n == 10


# --- ToolDefinition shape ---


class TestToolDefinition:
    def test_basic_shape(self):
        td = ToolDefinition(
            name="example",
            description="does something",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        assert td.name == "example"
        # Schema is stored verbatim — adapters reshape, not us.
        assert td.input_schema["required"] == ["x"]
