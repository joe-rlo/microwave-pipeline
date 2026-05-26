"""Provider-agnostic LLM interface.

This module is the contract. Adapters in `src/llm/providers/` implement
`LLMProvider`; callers (pipeline, scheduler) talk to providers only
through this interface. Nothing here knows about Anthropic, OpenAI,
NEAR, or Bedrock — those details live in the adapters.

The shape borrows the union of what the providers we care about
expose, normalized to a single tagged-content model:

- Anthropic's native Messages API: typed content blocks (text,
  tool_use, tool_result, image), streaming via SSE with event types.
- OpenAI's chat completions: messages with `role` and string content
  (or content arrays), tool_calls in a separate field, streaming via
  SSE with `choices[].delta`.
- NEAR Cloud: OpenAI-compatible, same shape as OpenAI.
- AWS Bedrock: Anthropic-shaped body with `anthropic_version=bedrock-…`,
  streaming via Bedrock's event-stream framing.

Why a `Protocol` and not an ABC: callers depend on the shape, not on a
class hierarchy. Adapters can be plain modules + a small class. Tests
can hand in fakes without inheriting anything.

What this file is NOT:
- Not the selector. The selector (`src/llm/selector.py`, future) picks
  an adapter per call site; this file just defines what an adapter
  looks like.
- Not a base class. No shared implementation lives here — duplication
  inside each adapter beats premature abstraction at the boundary
  where the providers actually differ.
- Not a streaming client. It defines the events the adapter must emit;
  the consumer (orchestrator) decides what to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol


# --- Content blocks ----------------------------------------------------------

# Tagged-union shape for a single piece of content inside a message.
# Anthropic's native shape; OpenAI shape is mapped to this in the
# OpenAI-side adapter. Keeping one shape across providers means the
# orchestrator never branches on "which provider produced this".

ContentBlockType = Literal["text", "tool_use", "tool_result", "image"]


@dataclass(frozen=True)
class ToolUse:
    """The model is asking to call a tool."""

    id: str           # provider-assigned tool-call id (Anthropic: 'toolu_…', OpenAI: 'call_…')
    name: str         # tool name as registered with the provider
    arguments: dict[str, Any]  # parsed JSON arguments


@dataclass(frozen=True)
class ToolResult:
    """The caller is returning a tool's output to the model.

    `content` is the text the model sees; `is_error` lets the model
    know the tool failed without us inventing a sentinel string.
    Mirrors the MCP tool-result shape we use today in src/tools/.
    """

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ImageBlock:
    """Reserved for the v2 vision lane.

    Not consumed by any current provider in this codebase — the
    pipeline is text-only today (see README "No vision" limitation).
    Defined here so the contract doesn't shift later when vision lands.
    """

    media_type: str   # "image/png", "image/jpeg", etc.
    data_base64: str  # base64-encoded bytes
    source_url: str | None = None


@dataclass(frozen=True)
class ContentBlock:
    """One block in a message. Exactly one of the typed fields is set.

    Why not separate dataclasses per type with a Union: a single
    dataclass with a `type` discriminator is easier to serialize, easier
    to pattern-match in callers (`block.type == 'text'`), and survives
    JSON round-trips without `__class__` games. Validation is done at
    construction by `__post_init__`.
    """

    type: ContentBlockType
    text: str | None = None
    tool_use: ToolUse | None = None
    tool_result: ToolResult | None = None
    image: ImageBlock | None = None

    def __post_init__(self) -> None:
        # Enforce: exactly one typed field is set, and it matches `type`.
        # `object.__setattr__` would be needed if we mutated, but we just
        # validate and let frozen-ness do the rest.
        active = {
            "text": self.text is not None,
            "tool_use": self.tool_use is not None,
            "tool_result": self.tool_result is not None,
            "image": self.image is not None,
        }
        set_count = sum(active.values())
        if set_count != 1:
            raise ValueError(
                f"ContentBlock must have exactly one typed field set; got {set_count}"
            )
        active_field = next(k for k, v in active.items() if v)
        if active_field != self.type:
            raise ValueError(
                f"ContentBlock.type={self.type!r} but {active_field!r} field is set"
            )

    @classmethod
    def of_text(cls, text: str) -> "ContentBlock":
        return cls(type="text", text=text)

    @classmethod
    def of_tool_use(cls, tool_use: ToolUse) -> "ContentBlock":
        return cls(type="tool_use", tool_use=tool_use)

    @classmethod
    def of_tool_result(cls, tool_result: ToolResult) -> "ContentBlock":
        return cls(type="tool_result", tool_result=tool_result)

    @classmethod
    def of_image(cls, image: ImageBlock) -> "ContentBlock":
        return cls(type="image", image=image)


# --- Messages ---------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ProviderMessage:
    """One turn in a conversation as sent to a provider.

    `content` accepts either a raw string (convenience for plain text)
    or a list of `ContentBlock` (for assistant turns that include tool
    calls, or user turns that return tool results).

    `tool_call_id` is set on `role='tool'` messages — OpenAI requires
    it on tool responses; Anthropic encodes it inside the tool_result
    block instead. Adapters translate as needed.
    """

    role: Role
    content: str | list[ContentBlock]
    tool_call_id: str | None = None
    name: str | None = None  # OpenAI uses `name` on tool-role messages

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id and not self._has_tool_result():
            # The adapter needs *some* way to link this back to the tool call.
            # Either a top-level tool_call_id (OpenAI shape) or an embedded
            # tool_result block (Anthropic shape) — having neither is a bug.
            raise ValueError(
                "tool-role message needs tool_call_id or an embedded ToolResult"
            )

    def _has_tool_result(self) -> bool:
        if isinstance(self.content, str):
            return False
        return any(b.type == "tool_result" for b in self.content)


# --- Tool definitions -------------------------------------------------------

# The shape of a tool the model can call. Sent in the request body.
# We accept the JSON-Schema-flavored input_schema directly — adapters
# wrap it in whatever envelope the provider expects (Anthropic puts it
# at top-level; OpenAI nests under `function.parameters`).


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object


# --- Request ----------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRequest:
    """Everything an adapter needs to make one request.

    `metadata` is a free-form dict for telemetry/audit — request id,
    pipeline stage, health route, etc. Adapters should NOT send this
    to the provider; it stays on our side for logging.
    """

    model: str
    messages: list[ProviderMessage]
    system: str | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float | None = None
    # Two thinking knobs — providers pick whichever shape their target
    # model expects.
    #   thinking_budget: integer token budget. Used by Anthropic's
    #       legacy "enabled" mode and OpenAI's reasoning models. Some
    #       Bedrock-hosted Claude models still accept this.
    #   thinking_effort: string ("low" | "medium" | "high" | "max").
    #       Used by Bedrock's newer "adaptive" mode (Sonnet 4.6+, Opus
    #       4.7+) and by future provider-side adaptive shapes. When
    #       present, adapters that support adaptive prefer this.
    thinking_budget: int | None = None    # legacy enabled + budget_tokens
    thinking_effort: str | None = None    # adaptive: low|medium|high|max
    stream: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


# --- Stream events ----------------------------------------------------------

# Tagged-union of events an adapter emits over `send()`. The orchestrator
# pattern-matches on `type` and renders each event accordingly. All events
# share a `type` field so consumers can dispatch via a single switch.

StreamEventType = Literal[
    "text_delta",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_end",
    "usage",
    "thinking_delta",
    "done",
    "error",
]


@dataclass(frozen=True)
class TextDelta:
    type: Literal["text_delta"] = "text_delta"
    text: str = ""


@dataclass(frozen=True)
class ToolUseStart:
    """Model has begun a tool call. Arguments stream in via deltas."""

    id: str
    name: str
    type: Literal["tool_use_start"] = "tool_use_start"


@dataclass(frozen=True)
class ToolUseDelta:
    """Incremental JSON fragment for a tool call's arguments.

    Anthropic streams arguments as `input_json_delta` events with
    partial JSON strings; OpenAI streams them as `function.arguments`
    string deltas. Adapter responsibility to concatenate; consumers
    see a stream of fragments tagged with the call id.
    """

    id: str
    arguments_delta: str  # partial JSON
    type: Literal["tool_use_delta"] = "tool_use_delta"


@dataclass(frozen=True)
class ToolUseEnd:
    """A tool call has finished streaming. `arguments` is the assembled JSON."""

    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_use_end"] = "tool_use_end"


@dataclass(frozen=True)
class ThinkingDelta:
    """Extended-thinking text delta (Claude / o3 / GPT-5 thinking modes).

    Adapters that don't support thinking simply never emit this; consumers
    that don't care can ignore it. The orchestrator currently surfaces
    nothing — included in the event union so tracing tools can capture it.
    """

    text: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True)
class Usage:
    """Token accounting. Emitted at least once per response (typically
    near the end). Some providers emit deltas during streaming; some only
    a final tally. Consumers should accumulate or replace based on
    `is_final`.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    is_final: bool = False
    type: Literal["usage"] = "usage"


@dataclass(frozen=True)
class Done:
    """Stream completed normally. `stop_reason` mirrors Anthropic's
    field — adapters translate provider-specific reasons (OpenAI's
    `finish_reason`, Bedrock's `stop_reason`) into this set.
    """

    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "other"
    ]
    type: Literal["done"] = "done"


@dataclass(frozen=True)
class Error:
    """Stream terminated with an error. Consumers should treat this as
    a terminal event — no further events follow. Adapters should raise
    *only* for transport-level failures (network down, malformed
    response that can't be parsed at all); recoverable provider errors
    (rate limit, bad request) come through here so the orchestrator can
    inspect and decide whether to retry.
    """

    message: str
    status: int | None = None
    retryable: bool = False
    type: Literal["error"] = "error"


StreamEvent = (
    TextDelta | ToolUseStart | ToolUseDelta | ToolUseEnd
    | ThinkingDelta | Usage | Done | Error
)


# --- The provider protocol --------------------------------------------------


class LLMProvider(Protocol):
    """What every adapter implements.

    `name` identifies the provider for logging and selector matching
    (e.g., "anthropic_direct", "near", "bedrock"). `supports_tools` and
    `supports_thinking` let the selector skip providers that can't
    handle a given request (e.g., a Whisper provider gets skipped for
    chat work).
    """

    name: str
    supports_tools: bool
    supports_thinking: bool

    def send(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        """Stream a response.

        Always returns an async iterator, even when `req.stream=False` —
        the iterator just yields the final assembled events in one
        burst in that case. Keeps the caller code path uniform.

        Adapters MUST emit a terminal `Done` or `Error` event before
        the iterator stops. Consumers may rely on that to know the
        stream is truly finished and not just hung.
        """
        ...

    async def estimate_tokens(self, text: str) -> int:
        """Rough token count for budgeting.

        Used by the orchestrator's compaction trigger. Need not be
        exact — anthropic/openai/etc all have native tokenizers, but
        tiktoken on a Claude prompt is close enough for "are we
        approaching the limit" decisions. Adapters that can call a
        cheap server-side count endpoint may do so; the default
        implementation strategy (~4 chars/token) is fine.
        """
        ...
