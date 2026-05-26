"""AWS Bedrock provider — the BAA path for health PHI turns.

Wraps `boto3 bedrock-runtime` and translates Anthropic-on-Bedrock's
event-stream framing into our StreamEvent union. Used only when the
health router resolves to the `phi` path (see spec §7).

Why Bedrock and not direct Anthropic API for the BAA path:
- AWS signs BAAs through AWS Artifact. Anthropic's direct API also
  offers BAAs but the existing health spec (`microwave-health-spec.md`)
  picked Bedrock as the primary because the AWS BAA process is
  established for indie / solo deployments.
- The body shape is identical to Anthropic native — the only diff is
  the version header (`anthropic_version: bedrock-2023-05-31`) and the
  invocation surface.

What this adapter deliberately does NOT do:
- **No prompt caching.** Bedrock supports it on some models, but BAA
  coverage of cached prompts is a separate verification — the existing
  health spec keeps caching OFF until that's confirmed.
- **No request-body logging.** Only metadata (model id, latency, token
  counts) flows to the audit table. Prompt and response content never
  hit a log.
- **No region fallback.** Region is pinned at construction; the
  provider refuses to start without an explicit region. Multi-region
  BAA configs are out of scope for Phase D.1.
- **No retry-with-jitter.** Bedrock returns crisp `ThrottlingException`
  / `ValidationException` — surface them as Error events and let the
  orchestrator decide.

Auth — verified 2026-05-22:
- The simplest path is a long-term Bedrock API key set as
  `AWS_BEARER_TOKEN_BEDROCK` in env. boto3 reads it automatically; you
  don't pass it as a constructor kwarg (boto3 currently only accepts
  Bedrock bearer tokens via env, see boto3/boto3#4723). Traditional
  IAM access keys also work via the standard
  AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY env vars.
- Long-term keys are convenient for personal/exploration use; rotate
  every ~90 days. Short-term keys are preferred for production.

Model IDs — verified 2026-05-22:
- Newer Anthropic models require CROSS-REGION INFERENCE PROFILE IDs
  for on-demand invocation, not the raw model id. Passing
  `anthropic.claude-haiku-4-5-...` directly fails with "on-demand
  throughput isn't supported." Use the profile ID instead:
    us.anthropic.claude-haiku-4-5-20251001-v1:0    (US)
    eu.anthropic.claude-...                        (EU)
    apac.anthropic.claude-...                      (Asia Pacific)
    global.anthropic.claude-...                    (global cross-region)
- List what your account has access to via
  `boto3.client("bedrock").list_inference_profiles()`. The adapter
  passes whatever you put in `req.model` straight to Bedrock — no
  translation here.

boto3 is synchronous. We wrap it via `asyncio.to_thread` to play nice
with the rest of the async pipeline; the stream-event iteration also
hops to a thread because boto3's EventStream is a sync iterator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterable

from src.llm.provider import (
    Done,
    Error,
    LLMProvider,
    ProviderMessage,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolDefinition,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)

log = logging.getLogger(__name__)


# Anthropic-on-Bedrock version pin. Bedrock requires this in the body.
ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"


class BedrockProvider:
    """Bedrock adapter using boto3 under the hood.

    Surface is the same `LLMProvider` protocol every other adapter
    implements. Construction is cheap; the boto3 client is held on the
    instance and reused across `send()` calls.
    """

    name = "bedrock"
    supports_tools = True
    supports_thinking = True  # extended thinking lands via the Anthropic body

    def __init__(
        self,
        *,
        region: str,
        client: Any = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ):
        """Build a BedrockProvider.

        `client` is an injected boto3 `bedrock-runtime` client (used
        by tests). When None, the constructor creates one via boto3's
        env-var / IAM-role discovery. Explicit keys here override env.

        We deliberately don't import boto3 at module-load time —
        Bedrock is opt-in (only configured users need it). Importing
        it lazily means the rest of the codebase can run without boto3
        installed.
        """
        if not region:
            raise ValueError(
                "BedrockProvider requires an AWS region (e.g. 'us-east-1')"
            )
        self.region = region

        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as e:
                raise RuntimeError(
                    "BedrockProvider requires boto3. Install with "
                    "`pip install boto3` or `pip install -e \".[dev]\"`."
                ) from e
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                aws_session_token=session_token,
            )

    async def send(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        body = self._build_anthropic_body(req)

        # boto3's invoke_model_with_response_stream is synchronous and
        # returns an iterable EventStream. Run it in a worker thread,
        # then drain the stream in another thread, queuing parsed
        # events back to the async generator.
        try:
            resp = await asyncio.to_thread(
                self._client.invoke_model_with_response_stream,
                modelId=req.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as e:
            # boto3 surfaces ClientError, ValidationException, etc. We
            # don't want to import every exception class; pattern-match
            # on the response shape if available.
            yield _exception_to_error(e)
            return

        event_stream = resp.get("body") if isinstance(resp, dict) else getattr(resp, "body", None)
        if event_stream is None:
            yield Error(message="Bedrock response has no body stream")
            return

        # Async fan-out: a worker pulls events from the sync iterator and
        # pushes them on a queue; we await the queue here. The sentinel
        # value signals "stream drained" — None terminates the loop.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        parser = _BedrockAnthropicParser()

        def _drain():
            try:
                for raw_event in event_stream:
                    chunk = _extract_chunk_bytes(raw_event)
                    if chunk is None:
                        # Could be a ServiceUnavailableException-style
                        # framing event — translate.
                        err = _event_to_error(raw_event)
                        if err is not None:
                            loop.call_soon_threadsafe(queue.put_nowait, err)
                        continue
                    for evt in parser.feed(chunk):
                        loop.call_soon_threadsafe(queue.put_nowait, evt)
                for evt in parser.flush():
                    loop.call_soon_threadsafe(queue.put_nowait, evt)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, _exception_to_error(e)
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(_drain))
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield evt
        finally:
            await worker

    async def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # --- Body shaping ---

    def _build_anthropic_body(self, req: ProviderRequest) -> dict[str, Any]:
        """Translate a ProviderRequest into the Anthropic-on-Bedrock body.

        Extended thinking shape:
          - Newer models on Bedrock (Sonnet 4.6+, Opus 4.7+) require
            the ADAPTIVE shape: `thinking.type=adaptive` + a top-level
            `output_config.effort` field. The legacy
            `thinking.type=enabled` + `budget_tokens` shape returns
            ValidationException on these models.
          - We always emit the adaptive shape when thinking is
            requested. If the caller only supplied `thinking_budget`
            (legacy field) and no `thinking_effort`, we derive the
            effort string from the budget so existing callers that
            haven't been updated still work.
        """
        body: dict[str, Any] = {
            "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": req.max_tokens,
            "messages": [_message_to_anthropic(m) for m in req.messages],
        }
        if req.system:
            body["system"] = req.system
        if req.temperature is not None:
            body["temperature"] = req.temperature

        effort = _resolve_thinking_effort(req)
        if effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}

        if req.tools:
            body["tools"] = [_tool_to_anthropic(t) for t in req.tools]
        # NOTE: prompt caching is intentionally NOT enabled here. See
        # module docstring and microwave-health-spec.md §7.3.
        return body


# --- Helpers ---


def _tool_to_anthropic(t: ToolDefinition) -> dict[str, Any]:
    """Anthropic tool shape: { name, description, input_schema }."""
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }


# Valid effort strings the Bedrock adaptive thinking shape accepts.
# Mirrors LLMSession._EFFORT_BUDGETS keys so the round-trip is clean.
_VALID_EFFORTS = ("low", "medium", "high", "max")


def _resolve_thinking_effort(req) -> str | None:
    """Decide which effort string to send for Bedrock's adaptive shape.

    Preference order:
      1. `thinking_effort` if explicitly set (and valid)
      2. derived from `thinking_budget` via reverse mapping
      3. None (no thinking section emitted)

    Reverse mapping mirrors LLMSession._EFFORT_BUDGETS:
      <= 2K  → low
      <= 8K  → medium
      <= 32K → high
      else   → max
    """
    if req.thinking_effort:
        effort = req.thinking_effort.strip().lower()
        if effort in _VALID_EFFORTS:
            return effort
        # Unknown effort string — fall through to budget-based derivation
        # if available, else None.
    if req.thinking_budget is None or req.thinking_budget <= 0:
        return None
    b = req.thinking_budget
    if b <= 2_000:
        return "low"
    if b <= 8_000:
        return "medium"
    if b <= 32_000:
        return "high"
    return "max"


def _message_to_anthropic(m: ProviderMessage) -> dict[str, Any]:
    """Translate ProviderMessage → Anthropic message.

    Anthropic's role taxonomy is just user / assistant. Tool results
    come through as a user message containing tool_result blocks.
    """
    role = "user" if m.role in ("user", "tool") else m.role

    if isinstance(m.content, str):
        if m.role == "tool":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }],
            }
        return {"role": role, "content": m.content}

    # Block list — translate each block to Anthropic shape
    blocks: list[dict[str, Any]] = []
    for cb in m.content:
        if cb.type == "text":
            blocks.append({"type": "text", "text": cb.text})
        elif cb.type == "tool_use":
            tu = cb.tool_use
            assert tu is not None
            blocks.append({
                "type": "tool_use",
                "id": tu.id,
                "name": tu.name,
                "input": tu.arguments,
            })
        elif cb.type == "tool_result":
            tr = cb.tool_result
            assert tr is not None
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tr.tool_use_id,
                "content": tr.content,
            }
            if tr.is_error:
                block["is_error"] = True
            blocks.append(block)
        elif cb.type == "image":
            img = cb.image
            assert img is not None
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.data_base64,
                },
            })
    return {"role": role, "content": blocks}


def _map_anthropic_stop_reason(reason: str | None) -> str:
    """Anthropic stop_reason values map 1:1 to our Done taxonomy
    except for `tool_use` (no rename) and `stop_sequence` (kept)."""
    if reason in ("end_turn", "max_tokens", "stop_sequence", "tool_use"):
        return reason
    return "other"


def _extract_chunk_bytes(raw_event: Any) -> bytes | None:
    """Pull the bytes payload out of one Bedrock event-stream frame.

    Bedrock frames look like `{"chunk": {"bytes": b"<json>"}}`. Error
    frames look like `{"internalServerException": {...}}` etc. — we
    return None for those and let `_event_to_error` translate.
    """
    if not isinstance(raw_event, dict):
        return None
    chunk = raw_event.get("chunk")
    if not isinstance(chunk, dict):
        return None
    payload = chunk.get("bytes")
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    return None


def _event_to_error(raw_event: Any) -> Error | None:
    """Translate a non-chunk Bedrock framing event to an Error."""
    if not isinstance(raw_event, dict):
        return None
    for key in (
        "internalServerException",
        "modelStreamErrorException",
        "modelTimeoutException",
        "throttlingException",
        "validationException",
    ):
        info = raw_event.get(key)
        if info is not None:
            retryable = key in ("internalServerException", "throttlingException", "modelTimeoutException")
            message = (info.get("message") if isinstance(info, dict) else None) or key
            return Error(message=f"Bedrock {key}: {message}", retryable=retryable)
    return None


def _exception_to_error(e: Exception) -> Error:
    """Best-effort translation of boto3 exceptions → Error event.

    Avoids importing botocore.exceptions at module level so the
    bedrock module can be imported in environments without boto3.
    """
    name = type(e).__name__
    msg = str(e) or name
    retryable = name in (
        "ThrottlingException",
        "InternalServerException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
    )
    return Error(message=f"Bedrock {name}: {msg}", retryable=retryable)


# --- Anthropic event-stream parser -----------------------------------------


class _BedrockAnthropicParser:
    """Translate Anthropic streaming events (as decoded from Bedrock
    event-stream chunks) into our StreamEvent union.

    Anthropic event types we care about:
      message_start          — model id, initial usage
      content_block_start    — block starts (text | tool_use | thinking)
      content_block_delta    — text_delta | input_json_delta | thinking_delta
      content_block_stop     — block ends
      message_delta          — final stop_reason + output usage
      message_stop           — terminator
    """

    def __init__(self):
        # Track open content blocks by index. Bedrock streams blocks
        # in parallel-friendly form; each delta references its index.
        self._blocks: dict[int, dict[str, Any]] = {}
        self._final_usage: Usage | None = None
        self._final_stop_reason: str = "other"
        self._terminated = False

    def feed(self, raw_bytes: bytes) -> list[StreamEvent]:
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            return []

        evt_type = payload.get("type")
        out: list[StreamEvent] = []

        if evt_type == "message_start":
            msg = payload.get("message") or {}
            usage = msg.get("usage") or {}
            if usage:
                # message_start usage is input-only; output_tokens
                # arrives in message_delta. Track for final aggregation.
                self._final_usage = Usage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=0,
                    cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0)),
                    cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
                    is_final=False,
                )

        elif evt_type == "content_block_start":
            idx = int(payload.get("index", 0))
            block = payload.get("content_block") or {}
            block_type = block.get("type")
            if block_type == "tool_use":
                self._blocks[idx] = {
                    "type": "tool_use",
                    "id": block.get("id", f"toolu_idx_{idx}"),
                    "name": block.get("name", ""),
                    "args_buf": "",
                }
                out.append(
                    ToolUseStart(
                        id=self._blocks[idx]["id"],
                        name=self._blocks[idx]["name"],
                    )
                )
            else:
                self._blocks[idx] = {"type": block_type or "text"}

        elif evt_type == "content_block_delta":
            idx = int(payload.get("index", 0))
            delta = payload.get("delta") or {}
            delta_type = delta.get("type")
            state = self._blocks.get(idx)
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    out.append(TextDelta(text=text))
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                if state and state.get("type") == "tool_use":
                    state["args_buf"] += partial
                    out.append(
                        ToolUseDelta(id=state["id"], arguments_delta=partial)
                    )
            elif delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                if text:
                    out.append(ThinkingDelta(text=text))

        elif evt_type == "content_block_stop":
            idx = int(payload.get("index", 0))
            state = self._blocks.pop(idx, None)
            if state and state.get("type") == "tool_use":
                try:
                    args = json.loads(state["args_buf"]) if state["args_buf"] else {}
                except json.JSONDecodeError:
                    log.warning(
                        "Malformed tool arguments from Bedrock for %s: %s",
                        state["id"], state["args_buf"][:200],
                    )
                    args = {}
                out.append(
                    ToolUseEnd(
                        id=state["id"], name=state["name"], arguments=args
                    )
                )

        elif evt_type == "message_delta":
            delta = payload.get("delta") or {}
            stop = delta.get("stop_reason")
            if stop:
                self._final_stop_reason = _map_anthropic_stop_reason(stop)
            # Usage update with output tokens
            usage = payload.get("usage") or {}
            if usage and self._final_usage:
                self._final_usage = Usage(
                    input_tokens=self._final_usage.input_tokens,
                    output_tokens=int(usage.get("output_tokens", 0)),
                    cache_creation_tokens=self._final_usage.cache_creation_tokens,
                    cache_read_tokens=self._final_usage.cache_read_tokens,
                    is_final=False,
                )

        elif evt_type == "message_stop":
            self._terminated = True

        return out

    def flush(self) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        # Any open blocks at flush time get closed off (shouldn't happen
        # under normal flow but defensive against truncated streams).
        for idx, state in list(self._blocks.items()):
            if state.get("type") == "tool_use":
                try:
                    args = json.loads(state["args_buf"]) if state["args_buf"] else {}
                except json.JSONDecodeError:
                    args = {}
                out.append(
                    ToolUseEnd(
                        id=state["id"], name=state["name"], arguments=args
                    )
                )
        self._blocks.clear()
        if self._final_usage:
            out.append(
                Usage(
                    input_tokens=self._final_usage.input_tokens,
                    output_tokens=self._final_usage.output_tokens,
                    cache_creation_tokens=self._final_usage.cache_creation_tokens,
                    cache_read_tokens=self._final_usage.cache_read_tokens,
                    is_final=True,
                )
            )
        out.append(Done(stop_reason=self._final_stop_reason))
        return out
