"""Tests for the NEAR provider adapter.

Two kinds of coverage:

1. **Pure translation** — `_message_to_openai`, `_tool_to_openai`,
   `_map_finish_reason`, and the SSE `_OpenAIStreamParser`. No network,
   no async — these are functions and a class with state.

2. **send() integration** — the full async stream. Uses an injected
   httpx.AsyncClient backed by httpx.MockTransport, so we exercise the
   real SSE-line parsing pipeline without hitting a real server.

The mock-transport approach matches what httpx itself recommends and
avoids monkeypatching internals.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.llm.provider import (
    ContentBlock,
    Done,
    Error,
    ProviderMessage,
    ProviderRequest,
    TextDelta,
    ToolDefinition,
    ToolResult,
    ToolUse,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)
from src.llm.providers.near import (
    NEARProvider,
    _OpenAIStreamParser,
    _map_finish_reason,
    _message_to_openai,
    _tool_to_openai,
)


# --- _tool_to_openai ---


class TestToolTranslation:
    def test_basic_shape(self):
        td = ToolDefinition(
            name="x", description="d", input_schema={"type": "object"}
        )
        out = _tool_to_openai(td)
        assert out["type"] == "function"
        assert out["function"]["name"] == "x"
        assert out["function"]["description"] == "d"
        assert out["function"]["parameters"] == {"type": "object"}


# --- _message_to_openai ---


class TestMessageTranslation:
    def test_plain_user(self):
        m = ProviderMessage(role="user", content="hi")
        out = _message_to_openai(m)
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_text_blocks(self):
        m = ProviderMessage(
            role="assistant",
            content=[ContentBlock.of_text("a"), ContentBlock.of_text("b")],
        )
        out = _message_to_openai(m)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "ab"

    def test_assistant_with_tool_use(self):
        tu = ToolUse(id="call_1", name="t", arguments={"q": 1})
        m = ProviderMessage(
            role="assistant",
            content=[ContentBlock.of_text("calling"), ContentBlock.of_tool_use(tu)],
        )
        out = _message_to_openai(m)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "calling"
        assert len(out[0]["tool_calls"]) == 1
        tc = out[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "t"
        # arguments must be a JSON string per OpenAI spec, not a dict
        assert json.loads(tc["function"]["arguments"]) == {"q": 1}

    def test_tool_role_string_content(self):
        m = ProviderMessage(
            role="tool", content="result text", tool_call_id="call_1"
        )
        out = _message_to_openai(m)
        assert out == [
            {"role": "tool", "content": "result text", "tool_call_id": "call_1"}
        ]

    def test_tool_role_block_content(self):
        # Anthropic-shaped input: one tool message with two tool_result blocks
        # → emits two OpenAI tool messages, one per result.
        m = ProviderMessage(
            role="tool",
            content=[
                ContentBlock.of_tool_result(
                    ToolResult(tool_use_id="call_1", content="r1")
                ),
                ContentBlock.of_tool_result(
                    ToolResult(tool_use_id="call_2", content="r2")
                ),
            ],
        )
        out = _message_to_openai(m)
        assert len(out) == 2
        assert out[0]["tool_call_id"] == "call_1"
        assert out[0]["content"] == "r1"
        assert out[1]["tool_call_id"] == "call_2"
        assert out[1]["content"] == "r2"


# --- _map_finish_reason ---


class TestFinishReasonMapping:
    @pytest.mark.parametrize("openai_reason,expected", [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "other"),
        (None, "other"),
        ("unknown_thing", "other"),
    ])
    def test_mapping(self, openai_reason, expected):
        assert _map_finish_reason(openai_reason) == expected


# --- _OpenAIStreamParser ---


def _sse(payload: dict | str) -> str:
    """Build an SSE data line. dict → JSON; str → passed through (for [DONE])."""
    if isinstance(payload, str):
        return f"data: {payload}"
    return f"data: {json.dumps(payload)}"


class TestStreamParser:
    def test_text_deltas_flow(self):
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"content": "hel"}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"content": "lo"}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"finish_reason": "stop", "delta": {}}]
        }))
        events += parser.flush()

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hello"
        assert any(isinstance(e, Done) and e.stop_reason == "end_turn" for e in events)

    def test_ignores_non_data_lines(self):
        parser = _OpenAIStreamParser()
        # Empty line, keep-alive comment — both must be silently dropped.
        assert parser.feed_line("") == []
        assert parser.feed_line(": keep-alive") == []
        # [DONE] terminator must not throw.
        assert parser.feed_line(_sse("[DONE]")) == []

    def test_malformed_json_skipped(self):
        parser = _OpenAIStreamParser()
        # Some servers occasionally send malformed comment-style frames.
        # Parser must not raise; just drops them.
        assert parser.feed_line("data: not-json") == []

    def test_tool_call_streamed_across_deltas(self):
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"tool_calls": [{
                "id": "call_a",
                "index": 0,
                "function": {"name": "get_weather", "arguments": '{"loc":'},
            }]}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": '"sf"}'},
            }]}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"finish_reason": "tool_calls", "delta": {}}]
        }))
        events += parser.flush()

        starts = [e for e in events if isinstance(e, ToolUseStart)]
        deltas = [e for e in events if isinstance(e, ToolUseDelta)]
        ends = [e for e in events if isinstance(e, ToolUseEnd)]
        assert len(starts) == 1
        assert starts[0].id == "call_a"
        assert starts[0].name == "get_weather"
        # Two argument deltas → assembled at end
        assert len(deltas) == 2
        assert deltas[0].arguments_delta == '{"loc":'
        assert deltas[1].arguments_delta == '"sf"}'
        assert len(ends) == 1
        assert ends[0].arguments == {"loc": "sf"}

    def test_tool_call_id_carried_by_index(self):
        # Some OpenAI-compatible servers omit id after the first chunk.
        # Parser must keep streaming deltas under the original id.
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"tool_calls": [{
                "id": "call_b",
                "index": 0,
                "function": {"name": "t", "arguments": "{"},
            }]}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"tool_calls": [{
                "index": 0,  # NO id this time
                "function": {"arguments": '"k":1}'},
            }]}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"finish_reason": "tool_calls", "delta": {}}]
        }))
        events += parser.flush()
        ends = [e for e in events if isinstance(e, ToolUseEnd)]
        assert len(ends) == 1
        assert ends[0].id == "call_b"
        assert ends[0].arguments == {"k": 1}

    def test_malformed_tool_args_yields_empty_args(self):
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"tool_calls": [{
                "id": "call_c",
                "index": 0,
                "function": {"name": "t", "arguments": "{not-json"},
            }]}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"finish_reason": "tool_calls", "delta": {}}]
        }))
        events += parser.flush()
        ends = [e for e in events if isinstance(e, ToolUseEnd)]
        assert ends[0].arguments == {}

    def test_usage_chunk_captured_at_end(self):
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"content": "hi"}}]
        }))
        events += parser.feed_line(_sse({
            "choices": [{"finish_reason": "stop", "delta": {}}]
        }))
        # OpenAI sends the final usage chunk AFTER the finish_reason chunk
        # when stream_options.include_usage=True.
        events += parser.feed_line(_sse({
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }))
        events += parser.flush()

        usages = [e for e in events if isinstance(e, Usage)]
        assert len(usages) == 1
        assert usages[0].input_tokens == 100
        assert usages[0].output_tokens == 50
        assert usages[0].is_final is True

    def test_synthetic_done_when_finish_reason_missing(self):
        # Stream cuts off cleanly but the server forgot to send finish_reason.
        # Parser must still emit a terminal Done so downstream knows.
        parser = _OpenAIStreamParser()
        events = []
        events += parser.feed_line(_sse({
            "choices": [{"delta": {"content": "incomplete"}}]
        }))
        events += parser.feed_line(_sse("[DONE]"))
        events += parser.flush()

        dones = [e for e in events if isinstance(e, Done)]
        assert len(dones) == 1
        assert dones[0].stop_reason == "other"


# --- send() integration with httpx.MockTransport ---


def _mock_client_streaming(sse_body: str, status: int = 200) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient whose POST returns the given SSE body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.invalid")


def _mock_client_json(body: dict, status: int = 200) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            headers={"content-type": "application/json"},
            json=body,
        )

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.invalid")


@pytest.mark.asyncio
class TestSendStreaming:
    async def test_simple_text_response(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\n'
            'data: [DONE]\n\n'
        )
        client = _mock_client_streaming(sse)
        provider = NEARProvider(api_key="k", client=client)
        req = ProviderRequest(
            model="claude-haiku-4-5",
            messages=[ProviderMessage(role="user", content="hi")],
            stream=True,
        )

        events = []
        async for evt in provider.send(req):
            events.append(evt)
        await client.aclose()

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hello"
        assert any(isinstance(e, Done) for e in events)
        # Done event should be the terminal event.
        assert isinstance(events[-1], Done)

    async def test_4xx_yields_error_event(self):
        client = _mock_client_streaming("unauthorized", status=401)
        provider = NEARProvider(api_key="bad", client=client)
        req = ProviderRequest(model="m", messages=[], stream=True)

        events = []
        async for evt in provider.send(req):
            events.append(evt)
        await client.aclose()

        assert len(events) == 1
        assert isinstance(events[0], Error)
        assert events[0].status == 401
        assert events[0].retryable is False  # 401 not in retry set

    async def test_429_marked_retryable(self):
        client = _mock_client_streaming("rate limited", status=429)
        provider = NEARProvider(api_key="k", client=client)
        req = ProviderRequest(model="m", messages=[], stream=True)

        events = [e async for e in provider.send(req)]
        await client.aclose()
        assert isinstance(events[0], Error)
        assert events[0].retryable is True

    async def test_request_body_shape(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\ndata: [DONE]\n\n',
            )

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, base_url="https://example.invalid")
        provider = NEARProvider(api_key="k", client=client)
        req = ProviderRequest(
            model="claude-sonnet-4-6",
            system="be helpful",
            messages=[ProviderMessage(role="user", content="hi")],
            tools=[ToolDefinition(
                name="get_weather", description="d",
                input_schema={"type": "object"},
            )],
            max_tokens=1024,
            temperature=0.5,
            stream=True,
        )

        async for _ in provider.send(req):
            pass
        await client.aclose()

        body = captured["body"]
        assert body["model"] == "claude-sonnet-4-6"
        # System prompt becomes the first message
        assert body["messages"][0] == {"role": "system", "content": "be helpful"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.5
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert captured["auth"] == "Bearer k"


@pytest.mark.asyncio
class TestSendOneshot:
    async def test_oneshot_text(self):
        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "hi back"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        client = _mock_client_json(body)
        provider = NEARProvider(api_key="k", client=client)
        req = ProviderRequest(model="m", messages=[], stream=False)

        events = [e async for e in provider.send(req)]
        await client.aclose()

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hi back"
        usages = [e for e in events if isinstance(e, Usage)]
        assert usages and usages[0].input_tokens == 5
        assert isinstance(events[-1], Done)
        assert events[-1].stop_reason == "end_turn"

    async def test_oneshot_with_tool_calls(self):
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "t", "arguments": '{"k":1}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        client = _mock_client_json(body)
        provider = NEARProvider(api_key="k", client=client)
        req = ProviderRequest(model="m", messages=[], stream=False)

        events = [e async for e in provider.send(req)]
        await client.aclose()

        starts = [e for e in events if isinstance(e, ToolUseStart)]
        ends = [e for e in events if isinstance(e, ToolUseEnd)]
        assert len(starts) == 1 and starts[0].name == "t"
        assert len(ends) == 1 and ends[0].arguments == {"k": 1}
        assert isinstance(events[-1], Done)
        assert events[-1].stop_reason == "tool_use"


# --- Constructor ---


class TestConstructor:
    def test_empty_key_rejected(self):
        with pytest.raises(ValueError):
            NEARProvider(api_key="")

    def test_base_url_trimmed(self):
        p = NEARProvider(api_key="k", base_url="https://x.invalid/v1/")
        assert p.base_url == "https://x.invalid/v1"
