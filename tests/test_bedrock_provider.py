"""Tests for the AWS Bedrock provider — the BAA path adapter.

Three layers of coverage:

1. **Pure translation** — `_message_to_anthropic`, `_tool_to_anthropic`,
   `_map_anthropic_stop_reason`, `_extract_chunk_bytes`,
   `_event_to_error`. Functions / module-level — no async, no boto3.

2. **Anthropic event-stream parser** — `_BedrockAnthropicParser`
   exercised directly with synthetic event JSONs. Covers text deltas,
   tool_use round-trip, thinking deltas, usage aggregation, stop_reason
   mapping, defensive flush of truncated streams.

3. **send() integration** — full async path with a fake boto3 client
   that returns a list of event-stream frames. Confirms the threaded
   queue plumbing, request body shape (anthropic_version pin,
   system/tools/thinking/temperature), and error-path translation.

We never import boto3 — that's a Phase D deployment dependency. The
adapter's `client` injection point lets us drop in a dict-shaped fake.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pytest

from src.llm.provider import (
    ContentBlock,
    Done,
    Error,
    ImageBlock,
    ProviderMessage,
    ProviderRequest,
    TextDelta,
    ThinkingDelta,
    ToolDefinition,
    ToolResult,
    ToolUse,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)
from src.llm.providers.bedrock import (
    ANTHROPIC_BEDROCK_VERSION,
    BedrockProvider,
    _BedrockAnthropicParser,
    _event_to_error,
    _exception_to_error,
    _extract_chunk_bytes,
    _map_anthropic_stop_reason,
    _message_to_anthropic,
    _tool_to_anthropic,
)


# --- Helpers ---------------------------------------------------------------


def _frame(payload: dict) -> dict:
    """Build a fake Bedrock event-stream frame."""
    return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}


class _FakeBoto3:
    """Mimics boto3.client('bedrock-runtime') for tests.

    `invoke_model_with_response_stream` returns `{"body": <iterable>}`,
    matching the real response shape (body is iterable of frames).
    """

    def __init__(self, frames: Iterable[Any]):
        self._frames = list(frames)
        self.last_call: dict[str, Any] | None = None

    def invoke_model_with_response_stream(self, **kwargs):
        self.last_call = kwargs
        return {"body": iter(self._frames)}


# --- Constructor ---


class TestConstructor:
    def test_requires_region(self):
        with pytest.raises(ValueError, match="region"):
            BedrockProvider(region="", client=_FakeBoto3([]))

    def test_injected_client_used(self):
        fake = _FakeBoto3([])
        p = BedrockProvider(region="us-east-1", client=fake)
        assert p._client is fake


# --- _tool_to_anthropic ---


class TestToolTranslation:
    def test_basic(self):
        td = ToolDefinition(
            name="t",
            description="desc",
            input_schema={"type": "object", "properties": {}},
        )
        out = _tool_to_anthropic(td)
        # Anthropic shape: top-level name, description, input_schema
        assert out == {
            "name": "t",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {}},
        }


# --- _message_to_anthropic ---


class TestMessageTranslation:
    def test_plain_user(self):
        m = ProviderMessage(role="user", content="hi")
        assert _message_to_anthropic(m) == {"role": "user", "content": "hi"}

    def test_assistant_text_and_tool_use(self):
        tu = ToolUse(id="toolu_1", name="x", arguments={"k": 1})
        m = ProviderMessage(
            role="assistant",
            content=[
                ContentBlock.of_text("planning"),
                ContentBlock.of_tool_use(tu),
            ],
        )
        out = _message_to_anthropic(m)
        assert out["role"] == "assistant"
        assert out["content"] == [
            {"type": "text", "text": "planning"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "x",
                "input": {"k": 1},
            },
        ]

    def test_tool_role_with_string_content(self):
        # OpenAI-shaped tool message → must become a user message with a
        # tool_result block on the Anthropic side.
        m = ProviderMessage(role="tool", content="result", tool_call_id="toolu_1")
        out = _message_to_anthropic(m)
        assert out["role"] == "user"
        assert out["content"] == [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "result",
        }]

    def test_tool_role_with_block_content(self):
        m = ProviderMessage(
            role="tool",
            content=[
                ContentBlock.of_tool_result(
                    ToolResult(tool_use_id="toolu_1", content="ok")
                ),
                ContentBlock.of_tool_result(
                    ToolResult(tool_use_id="toolu_2", content="bad", is_error=True)
                ),
            ],
        )
        out = _message_to_anthropic(m)
        assert out["role"] == "user"
        assert out["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "bad",
                "is_error": True,
            },
        ]

    def test_image_block(self):
        img = ImageBlock(media_type="image/png", data_base64="aGVsbG8=")
        m = ProviderMessage(
            role="user",
            content=[ContentBlock.of_image(img)],
        )
        out = _message_to_anthropic(m)
        assert out["content"][0] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aGVsbG8=",
            },
        }


# --- _map_anthropic_stop_reason ---


class TestStopReasonMapping:
    @pytest.mark.parametrize("reason,expected", [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("tool_use", "tool_use"),
        (None, "other"),
        ("unknown_value", "other"),
    ])
    def test_mapping(self, reason, expected):
        assert _map_anthropic_stop_reason(reason) == expected


# --- _extract_chunk_bytes / _event_to_error ---


class TestEventFraming:
    def test_extract_chunk_bytes_happy(self):
        frame = {"chunk": {"bytes": b'{"type":"x"}'}}
        assert _extract_chunk_bytes(frame) == b'{"type":"x"}'

    def test_extract_chunk_bytes_non_dict_returns_none(self):
        assert _extract_chunk_bytes("string") is None
        assert _extract_chunk_bytes(None) is None
        assert _extract_chunk_bytes({"chunk": "wrong"}) is None
        assert _extract_chunk_bytes({"not-chunk": {}}) is None

    def test_event_to_error_throttle(self):
        evt = {"throttlingException": {"message": "slow down"}}
        err = _event_to_error(evt)
        assert err is not None
        assert err.retryable is True
        assert "throttlingException" in err.message

    def test_event_to_error_validation_not_retryable(self):
        evt = {"validationException": {"message": "bad arg"}}
        err = _event_to_error(evt)
        assert err is not None
        assert err.retryable is False

    def test_event_to_error_non_error_frame_returns_none(self):
        assert _event_to_error({"chunk": {"bytes": b"x"}}) is None
        assert _event_to_error(None) is None

    def test_exception_to_error_retryable_class(self):
        class ThrottlingException(Exception):
            pass
        err = _exception_to_error(ThrottlingException("slow"))
        assert err.retryable is True

    def test_exception_to_error_other_class(self):
        err = _exception_to_error(ValueError("bad"))
        assert err.retryable is False


# --- _BedrockAnthropicParser ---


class TestAnthropicParser:
    def test_text_streaming(self):
        parser = _BedrockAnthropicParser()
        out: list = []
        out += parser.feed(json.dumps({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 100}},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello "},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_stop", "index": 0,
        }).encode())
        out += parser.feed(json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        }).encode())
        out += parser.feed(json.dumps({"type": "message_stop"}).encode())
        out += parser.flush()

        text = "".join(e.text for e in out if isinstance(e, TextDelta))
        assert text == "hello world"
        usages = [e for e in out if isinstance(e, Usage)]
        assert usages and usages[-1].input_tokens == 100
        assert usages[-1].output_tokens == 50
        assert usages[-1].is_final is True
        dones = [e for e in out if isinstance(e, Done)]
        assert dones[-1].stop_reason == "end_turn"

    def test_tool_use_round_trip(self):
        parser = _BedrockAnthropicParser()
        out: list = []
        out += parser.feed(json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_X", "name": "search"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"x"}'},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_stop", "index": 0,
        }).encode())
        out += parser.feed(json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
        }).encode())
        out += parser.feed(json.dumps({"type": "message_stop"}).encode())
        out += parser.flush()

        starts = [e for e in out if isinstance(e, ToolUseStart)]
        deltas = [e for e in out if isinstance(e, ToolUseDelta)]
        ends = [e for e in out if isinstance(e, ToolUseEnd)]
        assert len(starts) == 1
        assert starts[0].id == "toolu_X"
        assert starts[0].name == "search"
        assert len(deltas) == 2
        assert "".join(d.arguments_delta for d in deltas) == '{"q":"x"}'
        assert len(ends) == 1
        assert ends[0].arguments == {"q": "x"}

    def test_thinking_delta(self):
        parser = _BedrockAnthropicParser()
        out: list = []
        out += parser.feed(json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "let me think..."},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_stop", "index": 0,
        }).encode())

        thinks = [e for e in out if isinstance(e, ThinkingDelta)]
        assert len(thinks) == 1
        assert thinks[0].text == "let me think..."

    def test_malformed_json_silently_skipped(self):
        parser = _BedrockAnthropicParser()
        # Returns [] rather than raising — keeps the stream alive.
        assert parser.feed(b"not-json") == []

    def test_truncated_stream_emits_done(self):
        # No message_stop or message_delta — just open and abandon.
        parser = _BedrockAnthropicParser()
        out = parser.flush()
        # flush() must emit a terminal Done so consumers know the
        # stream ended (defensive against truncated network frames).
        dones = [e for e in out if isinstance(e, Done)]
        assert len(dones) == 1
        assert dones[0].stop_reason == "other"

    def test_malformed_tool_args_yields_empty_args(self):
        parser = _BedrockAnthropicParser()
        out: list = []
        out += parser.feed(json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_Y", "name": "t"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{broken"},
        }).encode())
        out += parser.feed(json.dumps({
            "type": "content_block_stop", "index": 0,
        }).encode())
        ends = [e for e in out if isinstance(e, ToolUseEnd)]
        assert len(ends) == 1
        assert ends[0].arguments == {}


# --- send() integration ---


@pytest.mark.asyncio
class TestSendIntegration:
    async def test_basic_text_stream(self):
        frames = [
            _frame({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _frame({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            _frame({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"}}),
            _frame({"type": "content_block_stop", "index": 0}),
            _frame({"type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5}}),
            _frame({"type": "message_stop"}),
        ]
        fake = _FakeBoto3(frames)
        provider = BedrockProvider(region="us-east-1", client=fake)

        req = ProviderRequest(
            model="anthropic.claude-sonnet-4-x",
            messages=[ProviderMessage(role="user", content="hello")],
            system="be helpful",
        )

        events = [e async for e in provider.send(req)]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hi"
        assert isinstance(events[-1], Done)
        assert events[-1].stop_reason == "end_turn"

    async def test_request_body_shape(self):
        frames = [
            _frame({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            _frame({"type": "message_stop"}),
        ]
        fake = _FakeBoto3(frames)
        provider = BedrockProvider(region="us-east-1", client=fake)

        req = ProviderRequest(
            model="anthropic.claude-opus-4-x",
            messages=[ProviderMessage(role="user", content="hi")],
            system="sys",
            tools=[ToolDefinition(name="t", description="d", input_schema={"type": "object"})],
            max_tokens=2048,
            temperature=0.3,
            thinking_budget=8000,
        )

        async for _ in provider.send(req):
            pass

        call = fake.last_call
        assert call["modelId"] == "anthropic.claude-opus-4-x"
        body = json.loads(call["body"])
        assert body["anthropic_version"] == ANTHROPIC_BEDROCK_VERSION
        assert body["max_tokens"] == 2048
        assert body["temperature"] == 0.3
        assert body["system"] == "sys"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["tools"] == [{
            "name": "t", "description": "d", "input_schema": {"type": "object"},
        }]
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 8000}
        # CRITICAL: no prompt-cache-control field present.
        # The existing health spec disables caching pending BAA verification.
        body_str = call["body"]
        assert "cache_control" not in body_str

    async def test_throttling_frame_yields_error(self):
        # Bedrock can interleave error frames into the stream.
        frames = [
            _frame({"type": "message_start", "message": {"usage": {}}}),
            {"throttlingException": {"message": "rate exceeded"}},
        ]
        fake = _FakeBoto3(frames)
        provider = BedrockProvider(region="us-east-1", client=fake)

        req = ProviderRequest(model="m", messages=[])
        events = [e async for e in provider.send(req)]
        errors = [e for e in events if isinstance(e, Error)]
        assert errors
        assert errors[0].retryable is True

    async def test_invoke_exception_yields_error(self):
        class _Failing:
            def invoke_model_with_response_stream(self, **kw):
                # boto3's ValidationException etc. surface as exceptions
                raise RuntimeError("ValidationException: bad model id")

        provider = BedrockProvider(region="us-east-1", client=_Failing())
        req = ProviderRequest(model="bad", messages=[])
        events = [e async for e in provider.send(req)]
        assert any(isinstance(e, Error) for e in events)
