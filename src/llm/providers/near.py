"""NEAR AI Cloud provider — OpenAI-compatible adapter.

Talks to `https://cloud-api.near.ai/v1/chat/completions` over httpx.
NEAR's surface is OpenAI-compatible (Chat Completions, Models, Files),
so the request shape is `model + messages + tools` in OpenAI form and
the response stream is OpenAI SSE.

This file translates in both directions:

REQUEST  ProviderRequest → OpenAI body
  - system → message with role="system"
  - ContentBlock list → OpenAI message content (string for plain text,
    or assistant tool_calls field for tool_use blocks, or tool-role
    message for tool_result blocks)
  - ToolDefinition → {"type": "function", "function": {...}}
  - thinking_budget → not supported in OpenAI shape; ignored with a warning
    (NEAR exposes reasoning models that use this differently; see Reasoning
    Models docs for the slugs that accept it)

RESPONSE OpenAI SSE → StreamEvent
  - delta.content → TextDelta
  - delta.tool_calls[0].function.arguments → ToolUseDelta (accumulated per id)
  - finish_reason="tool_calls" → emit ToolUseEnd for each open call, then Done
  - finish_reason="stop" → Done(end_turn)
  - finish_reason="length" → Done(max_tokens)
  - usage → Usage event (final tally; OpenAI sends one at stream end
    when stream_options.include_usage=True, which we set)
  - [DONE] → stream terminator (no event)

What we deliberately don't do:
- No httpx retry layer. NEAR returns crisp 4xx/5xx; surface them as
  Error events and let the orchestrator decide. Mass retries on a
  rate-limit hammer the API.
- No tokenizer-on-server estimate_tokens. We use the same 4-char
  heuristic as the rest of the codebase. Phase B doesn't need
  precision; Phase C can revisit if compaction triggers misfire.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from src.llm.provider import (
    ContentBlock,
    Done,
    Error,
    LLMProvider,
    ProviderMessage,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ToolDefinition,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://cloud-api.near.ai/v1"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class NEARProvider:
    """OpenAI-compatible adapter for NEAR AI Cloud.

    `name`, `supports_tools`, `supports_thinking` are class-level so the
    selector can inspect without constructing.
    """

    name = "near"
    supports_tools = True
    # Thinking is model-specific on NEAR; the adapter passes the param
    # through when the request asks for it, but doesn't claim universal
    # support. Selector should only ask for thinking on known-thinking
    # models (o3 Mini, GPT-5.x, Claude Opus 4.x — verified at Phase C).
    supports_thinking = True

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise ValueError("NEAR API key required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Optional injected client for tests; production opens per call.
        self._client = client
        self._timeout = timeout

    async def send(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        body = self._build_body(req)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if req.stream else "application/json",
        }

        own_client = self._client is None
        client = self._client if not own_client else httpx.AsyncClient(timeout=self._timeout)

        try:
            if req.stream:
                async for evt in self._stream(client, url, headers, body):
                    yield evt
            else:
                async for evt in self._oneshot(client, url, headers, body):
                    yield evt
        finally:
            if own_client:
                await client.aclose()

    async def estimate_tokens(self, text: str) -> int:
        # ~4 chars per token; the existing pipeline uses the same shape
        # in src/pipeline/orchestrator.py compaction-budget math.
        return max(1, len(text) // 4)

    # --- Request shaping ---

    def _build_body(self, req: ProviderRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.extend(_message_to_openai(m))

        body: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = [_tool_to_openai(t) for t in req.tools]
        if req.stream:
            # Ask NEAR to include the final usage tally on the stream.
            # OpenAI-compatible flag; cheap when the server honors it
            # and harmless when it doesn't.
            body["stream_options"] = {"include_usage": True}
        if req.thinking_budget is not None:
            # Some OpenAI-compatible servers accept `reasoning_effort` /
            # `max_completion_tokens` for thinking-style models. NEAR's
            # docs reference Reasoning Models with their own knobs.
            # Pass through under a documented field; non-thinking models
            # ignore it. (Phase C will pin the exact field name once a
            # reasoning-model integration is needed.)
            body["reasoning_budget_tokens"] = req.thinking_budget
        return body

    # --- Streaming path ---

    async def _stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        try:
            async with client.stream(
                "POST", url, headers=headers, json=body
            ) as resp:
                if resp.status_code >= 400:
                    # Read the body so error messages aren't useless.
                    err_text = (await resp.aread()).decode("utf-8", errors="replace")
                    yield Error(
                        message=f"NEAR {resp.status_code}: {err_text[:500]}",
                        status=resp.status_code,
                        retryable=resp.status_code in (429, 500, 502, 503, 504),
                    )
                    return

                parser = _OpenAIStreamParser()
                async for line in resp.aiter_lines():
                    for evt in parser.feed_line(line):
                        yield evt
                # Flush any tool-call ends still buffered.
                for evt in parser.flush():
                    yield evt
        except httpx.HTTPError as e:
            yield Error(message=f"NEAR network error: {e}", retryable=True)

    # --- Non-streaming path ---

    async def _oneshot(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            yield Error(message=f"NEAR network error: {e}", retryable=True)
            return

        if resp.status_code >= 400:
            yield Error(
                message=f"NEAR {resp.status_code}: {resp.text[:500]}",
                status=resp.status_code,
                retryable=resp.status_code in (429, 500, 502, 503, 504),
            )
            return

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            yield Error(message=f"NEAR malformed JSON: {e}")
            return

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if text:
            yield TextDelta(text=text)
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (KeyError, json.JSONDecodeError):
                args = {}
            yield ToolUseStart(id=tc["id"], name=tc["function"]["name"])
            yield ToolUseEnd(
                id=tc["id"], name=tc["function"]["name"], arguments=args
            )
        usage = data.get("usage") or {}
        if usage:
            yield Usage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                cache_creation_tokens=int(usage.get("cache_creation_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
                is_final=True,
            )
        yield Done(stop_reason=_map_finish_reason(choice.get("finish_reason")))


# --- Helpers (module-level for unit testing) --------------------------------


def _tool_to_openai(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        },
    }


def _message_to_openai(m: ProviderMessage) -> list[dict[str, Any]]:
    """Translate one ProviderMessage to one OpenAI message (or several
    when the assistant turn includes tool_use blocks).

    OpenAI's model is "role + content + optional tool_calls" — multiple
    of our content blocks may collapse into a single OpenAI message.
    Returns a list so this function can emit >1 OpenAI message when the
    input was a tool-role message containing multiple tool_result blocks
    (each result is a separate OpenAI tool message).
    """
    role = m.role

    if role == "tool":
        return list(_tool_role_to_openai(m))

    # Plain text content
    if isinstance(m.content, str):
        return [{"role": role, "content": m.content}]

    # Block list — collect text into one string and tool_use into the
    # tool_calls field (assistant role only). For user role with images,
    # OpenAI accepts an array content; we just pass blocks through.
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    array_content: list[dict[str, Any]] = []
    has_array = False

    for block in m.content:
        if block.type == "text":
            assert block.text is not None
            text_chunks.append(block.text)
            array_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            assert block.tool_use is not None
            tool_calls.append({
                "id": block.tool_use.id,
                "type": "function",
                "function": {
                    "name": block.tool_use.name,
                    "arguments": json.dumps(block.tool_use.arguments),
                },
            })
        elif block.type == "image":
            assert block.image is not None
            has_array = True
            array_content.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        block.image.source_url
                        or f"data:{block.image.media_type};base64,{block.image.data_base64}"
                    ),
                },
            })
        # tool_result blocks inside a non-tool message shouldn't happen;
        # the constructor allows it but the spec says tool_result lives
        # on role=tool. Skip silently.

    msg: dict[str, Any] = {"role": role}
    if has_array:
        msg["content"] = array_content
    else:
        msg["content"] = "".join(text_chunks) if text_chunks else None
    if tool_calls and role == "assistant":
        msg["tool_calls"] = tool_calls
    return [msg]


def _tool_role_to_openai(m: ProviderMessage) -> AsyncIterator[dict[str, Any]] | list[dict[str, Any]]:
    """tool-role messages may carry one or more tool_result blocks (the
    Anthropic shape) or a plain string (the OpenAI shape).

    OpenAI wants ONE message per tool result, with `tool_call_id` at the
    top level. Anthropic wants one role=user message with multiple
    tool_result blocks. We always emit OpenAI shape on the way out.
    """
    out: list[dict[str, Any]] = []

    if isinstance(m.content, str):
        # OpenAI shape: plain string content + top-level tool_call_id
        if not m.tool_call_id:
            raise ValueError("tool message with string content needs tool_call_id")
        out.append({
            "role": "tool",
            "content": m.content,
            "tool_call_id": m.tool_call_id,
        })
        return out

    # Anthropic shape: list of tool_result blocks
    for block in m.content:
        if block.type != "tool_result":
            continue
        assert block.tool_result is not None
        out.append({
            "role": "tool",
            "content": block.tool_result.content,
            "tool_call_id": block.tool_result.tool_use_id,
        })
    return out


def _map_finish_reason(reason: str | None) -> str:
    """OpenAI finish_reason → our Done stop_reason taxonomy."""
    if reason == "stop":
        return "end_turn"
    if reason == "length":
        return "max_tokens"
    if reason == "tool_calls":
        return "tool_use"
    if reason == "content_filter":
        return "other"
    if reason is None:
        return "other"
    return "other"


# --- SSE parser -------------------------------------------------------------


class _OpenAIStreamParser:
    """Stateful parser for OpenAI-shaped SSE.

    OpenAI sends `data: {json}` lines, with `data: [DONE]` as the
    terminator. Each delta carries either text content or partial
    tool_calls. Tool calls stream in fragments per id — we track open
    calls and emit ToolUseStart on first sight, ToolUseDelta for each
    argument fragment, and ToolUseEnd once the call's `function` block
    stops appearing (typically at finish_reason).

    We don't emit ToolUseEnd until we see finish_reason or the stream
    ends — OpenAI doesn't have a per-call "end" marker.
    """

    def __init__(self) -> None:
        # tool_call_id → {"name": str, "args_buf": str, "started": bool}
        self._tool_calls: dict[str, dict[str, Any]] = {}
        # We track which calls have been "started" so we don't emit
        # ToolUseStart twice when the model interleaves multiple calls.
        self._final_usage: Usage | None = None
        self._final_done: Done | None = None

    def feed_line(self, line: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if not line.startswith("data: "):
            return events
        payload = line[len("data: "):]
        if payload == "[DONE]":
            # Stream terminator; final events will be flushed by flush().
            return events

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # Some servers emit keep-alive comments; skip.
            return events

        # Usage chunk (final tally with stream_options.include_usage=True).
        # OpenAI sends this as a separate chunk where `choices` is empty.
        usage = data.get("usage")
        if usage:
            self._final_usage = Usage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                cache_creation_tokens=int(usage.get("cache_creation_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
                is_final=True,
            )

        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                events.append(TextDelta(text=content))

            for tc in delta.get("tool_calls") or []:
                tc_id = tc.get("id")
                idx = tc.get("index", 0)
                # Some OpenAI-compatible servers (incl. some NEAR upstreams)
                # only send `id` on the first chunk and then identify by
                # `index`; carry the id forward by index.
                if not tc_id:
                    tc_id = self._id_by_index(idx)
                if not tc_id:
                    # First chunk for this index has no id; synthesize.
                    tc_id = f"call_idx_{idx}"

                fn = tc.get("function") or {}
                name = fn.get("name")
                args_chunk = fn.get("arguments")

                state = self._tool_calls.setdefault(
                    tc_id, {"name": name or "", "args_buf": "", "started": False, "index": idx}
                )
                if name and not state["name"]:
                    state["name"] = name
                if args_chunk:
                    state["args_buf"] += args_chunk

                if not state["started"] and state["name"]:
                    state["started"] = True
                    events.append(
                        ToolUseStart(id=tc_id, name=state["name"])
                    )
                if args_chunk:
                    events.append(
                        ToolUseDelta(id=tc_id, arguments_delta=args_chunk)
                    )

            finish = choice.get("finish_reason")
            if finish:
                # Stream is wrapping up for this choice — assemble any
                # open tool calls into ToolUseEnd events first, then
                # emit the Done. Final usage chunk may arrive AFTER the
                # finish_reason chunk; flush() handles ordering.
                events.extend(self._end_tool_calls())
                self._final_done = Done(stop_reason=_map_finish_reason(finish))

        return events

    def _id_by_index(self, idx: int) -> str | None:
        for tc_id, state in self._tool_calls.items():
            if state.get("index") == idx:
                return tc_id
        return None

    def _end_tool_calls(self) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        for tc_id, state in self._tool_calls.items():
            if not state["started"]:
                continue
            try:
                args = json.loads(state["args_buf"]) if state["args_buf"] else {}
            except json.JSONDecodeError:
                # Defensive: emit empty args rather than crashing the stream.
                # The orchestrator will surface the tool failure to the LLM.
                log.warning("Malformed tool arguments for %s: %s", tc_id, state["args_buf"][:200])
                args = {}
            out.append(
                ToolUseEnd(id=tc_id, name=state["name"], arguments=args)
            )
        self._tool_calls.clear()
        return out

    def flush(self) -> list[StreamEvent]:
        """Emit any buffered terminal events. Called once after the stream closes."""
        out: list[StreamEvent] = []
        # If we still have open tool calls (no finish_reason seen), end them.
        out.extend(self._end_tool_calls())
        if self._final_usage:
            out.append(self._final_usage)
        if self._final_done:
            out.append(self._final_done)
        else:
            # Stream closed without an explicit finish_reason. Emit a
            # synthetic Done so downstream consumers know the stream is
            # over. "other" is the catch-all.
            out.append(Done(stop_reason="other"))
        self._final_usage = None
        self._final_done = None
        return out


