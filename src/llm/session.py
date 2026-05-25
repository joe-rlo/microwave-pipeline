"""Multi-turn LLM session over the provider abstraction.

What this replaces: the persistent-connection behavior of the Agent
SDK's `ClaudeSDKClient` — system-prompt-once, then a sequence of
turns with growing history. NEAR / Anthropic / OpenAI all support
this natively via stateless HTTP + we-send-the-history-each-call.

What this DOES NOT do (yet — Phase C.3 / C.4):
- Anthropic direct (not via NEAR). For Phase C only the NEAR provider
  is wired; Anthropic Direct is a future provider option.
- Multimodal. The orchestrator passes images through `send(images=...)`;
  we accept and forward the kwarg but the underlying provider lacks
  vision handling at this phase. Defers to Phase C.4 alongside the
  Agent SDK removal.

The chunk shape on `send()` matches the existing `LLMClient.send()`
contract so the orchestrator can swap implementations without
changing its consumer logic:

  {"type": "delta",      "text": <streaming text chunk>}
  {"type": "tool_use",   "name": <tool name>}            # placeholder for C.2
  {"type": "result",     "text": <full assembled response>}

That last `result` event is the de-facto end-of-stream marker. The
existing orchestrator already keys on it for full-response assembly
(see src/pipeline/orchestrator.py line 522).
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable

from src.llm.provider import (
    ContentBlock,
    Done,
    Error,
    LLMProvider,
    ProviderMessage,
    ProviderRequest,
    TextDelta,
    ToolDefinition,
    ToolResult,
    ToolUse,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)
from src.llm.providers.near import NEARProvider

log = logging.getLogger(__name__)


# A tool handler is an async function: arguments dict → result string.
# Errors should be raised; the loop converts them into is_error=True
# ToolResult blocks so the model can recover gracefully.
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


# Cap on how many tool-use rounds the model can chain in a single send().
# 8 is a sensible default — Anthropic's own docs suggest most legitimate
# workflows complete in 2–4 rounds; 8 leaves headroom for adversarial
# patterns while preventing infinite loops on a broken tool.
DEFAULT_MAX_TOOL_ITERATIONS = 8


class LLMSession:
    """Stateful wrapper around a stateless provider.

    Mirrors the surface of `src.llm.client.LLMClient` so the
    orchestrator can substitute one for the other behind an env flag.
    Specifically:

    - `connect(stable_prompt)` — set system prompt, clear history
    - `reconnect(stable_prompt)` — same as connect; no persistent socket
      to recycle, so this just resets the system prompt + history
    - `disconnect()` — no-op (no socket); kept for shape parity
    - `escalate(model, effort)` — swap model + set thinking budget for
      the next send
    - `de_escalate()` — restore the base model
    - `send(user_message, memory_context, images)` — stream
    """

    # Effort → token budget. Mirrors the existing ESCALATION_EFFORT
    # mapping in the README so behavior is identical when the env flag
    # flips. See spec §5 for verification at Phase B/C bake-off.
    _EFFORT_BUDGETS = {
        "low": 2_000,
        "medium": 8_000,
        "high": 32_000,
        "max": 64_000,
    }

    def __init__(
        self,
        *,
        model: str,
        provider: LLMProvider | None = None,
        max_tokens: int = 8192,
        tools: list[ToolDefinition] | None = None,
        tool_handlers: dict[str, ToolHandler] | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        """Build a session.

        `provider` is injected for tests. Production uses the env-driven
        factory at module bottom: `build_default_session()`.

        `tools` + `tool_handlers` enable tool use. They MUST agree on
        names — every ToolDefinition.name needs a matching key in
        tool_handlers, or the loop will surface a clear error to the
        model when a call comes through.
        """
        self.model = model
        self._base_model = model
        self._provider = provider if provider is not None else _build_default_provider()
        self._max_tokens = max_tokens

        self._tools = list(tools or [])
        self._tool_handlers = dict(tool_handlers or {})
        self._max_tool_iters = max_tool_iterations
        # Cheap consistency check at construction so a typo doesn't
        # surface only when the model first calls the tool.
        for td in self._tools:
            if td.name not in self._tool_handlers:
                log.warning(
                    "Tool %r has a definition but no handler — calls will fail",
                    td.name,
                )

        self._system_prompt: str | None = None
        self._history: list[ProviderMessage] = []
        self._thinking_budget: int | None = None

    # --- Lifecycle ---

    async def connect(self, stable_prompt: str) -> None:
        self._system_prompt = stable_prompt
        self._history = []
        # Surface the provider so it's clear from logs whether this
        # session is talking to NEAR Cloud, AWS Bedrock, or a future
        # adapter — the model id alone isn't always disambiguating
        # (`anthropic/claude-sonnet-4-6` vs `us.anthropic.claude-sonnet-4-6`
        # differ by one prefix character).
        log.info(
            "[llm-session] connected provider=%s model=%s",
            getattr(self._provider, "name", "unknown"),
            self.model,
        )

    async def reconnect(self, stable_prompt: str) -> None:
        """Reset system prompt + history. Used by the orchestrator when
        the stable context changes (skill activation, channel switch).
        No socket to recycle; this is purely state replacement."""
        await self.connect(stable_prompt)

    async def disconnect(self) -> None:
        # Nothing persistent to close. Provider's httpx clients are
        # opened per send() and closed automatically.
        self._system_prompt = None
        self._history = []

    # --- Escalation ---

    async def escalate(self, model: str, effort: str = "high") -> None:
        self._base_model = self.model
        self.model = model
        self._thinking_budget = self._EFFORT_BUDGETS.get(effort.lower())
        log.info(
            "LLMSession escalated: model=%s effort=%s budget=%s",
            model, effort, self._thinking_budget,
        )

    async def de_escalate(self) -> None:
        self.model = self._base_model
        self._thinking_budget = None
        log.info("LLMSession de-escalated back to %s", self.model)

    # --- Send ---

    async def send(
        self,
        user_message: str,
        memory_context: str | None = None,
        images: list[tuple[bytes, str]] | None = None,
    ) -> AsyncIterator[dict]:
        """Stream a turn, running any tool-use rounds inline.

        Yields LLMClient-shape dict chunks:
          {"type": "delta",     "text": ...}        — streaming text
          {"type": "tool_use",  "name": ...}        — visibility (typing)
          {"type": "tool_result", "name": ...,
           "is_error": bool}                        — tool finished
          {"type": "result",    "text": <full>}    — final assembled

        Per-turn memory context is prepended to the user message (same
        as existing LLMClient behavior — see
        src/llm/client.py line 215). The tool loop runs internally:
        when the model emits tool_use blocks, we run handlers, append
        results to history, and continue streaming until the model
        finishes with stop_reason="end_turn" (or hits the iteration cap).
        """
        if self._system_prompt is None:
            raise RuntimeError("LLMSession.send() called before connect()")

        if images:
            log.warning(
                "LLMSession received %d image(s); multimodal handling lands "
                "in Phase C.4. Sending text-only for now.",
                len(images),
            )

        enriched = (
            f"[Relevant memory context]\n{memory_context}\n\n{user_message}"
            if memory_context else user_message
        )
        self._history.append(ProviderMessage(role="user", content=enriched))

        full_response_chunks: list[str] = []

        for iteration in range(self._max_tool_iters):
            # Per-iteration state: text and tool-use accumulators
            iter_text_chunks: list[str] = []
            tool_uses: list[ToolUse] = []
            error: Error | None = None
            stop_reason: str = "other"

            req = ProviderRequest(
                model=self.model,
                system=self._system_prompt,
                messages=list(self._history),
                tools=list(self._tools),
                max_tokens=self._max_tokens,
                thinking_budget=self._thinking_budget,
                stream=True,
                metadata={"stage": "main", "iter": str(iteration)},
            )

            async for evt in self._provider.send(req):
                if isinstance(evt, TextDelta):
                    iter_text_chunks.append(evt.text)
                    full_response_chunks.append(evt.text)
                    yield {"type": "delta", "text": evt.text}
                elif isinstance(evt, ToolUseStart):
                    yield {"type": "tool_use", "name": evt.name}
                elif isinstance(evt, ToolUseEnd):
                    tool_uses.append(
                        ToolUse(id=evt.id, name=evt.name, arguments=evt.arguments)
                    )
                elif isinstance(evt, Done):
                    stop_reason = evt.stop_reason
                elif isinstance(evt, Error):
                    error = evt
                # Usage / ThinkingDelta intentionally dropped — not in
                # the LLMClient contract.

            # If the stream errored before we got anything useful AND no
            # tool calls landed, surface to the caller.
            if error is not None and not iter_text_chunks and not tool_uses:
                raise RuntimeError(
                    f"LLMSession.send failed: {error.message} "
                    f"(status={error.status}, retryable={error.retryable})"
                )

            # Append the assistant turn (text + tool_uses) to history so
            # the next iteration sees it. Building the content list is
            # subtle: assistant turns with tool_use must include the
            # text + every tool_use as separate blocks, in order.
            assistant_blocks: list[ContentBlock] = []
            iter_text = "".join(iter_text_chunks)
            if iter_text:
                assistant_blocks.append(ContentBlock.of_text(iter_text))
            for tu in tool_uses:
                assistant_blocks.append(ContentBlock.of_tool_use(tu))

            if assistant_blocks:
                self._history.append(
                    ProviderMessage(role="assistant", content=assistant_blocks)
                )

            # Done with no tool calls → conversation is over for this send()
            if not tool_uses:
                break

            # Execute tools and stage the results for the next iteration.
            # We run sequentially because most use cases only have one or
            # two parallel calls per turn and concurrent execution would
            # complicate handler error semantics. If a real workload needs
            # parallelism, switch to asyncio.gather here.
            result_blocks: list[ContentBlock] = []
            for tu in tool_uses:
                handler = self._tool_handlers.get(tu.name)
                if handler is None:
                    text = f"No handler registered for tool {tu.name!r}"
                    is_error = True
                    log.warning("Unhandled tool call: %s", tu.name)
                else:
                    try:
                        text = await handler(tu.arguments)
                        is_error = False
                    except Exception as e:
                        text = f"Tool {tu.name!r} raised: {e}"
                        is_error = True
                        log.warning("Tool %s raised: %s", tu.name, e)

                result_blocks.append(
                    ContentBlock.of_tool_result(
                        ToolResult(
                            tool_use_id=tu.id, content=text, is_error=is_error
                        )
                    )
                )
                yield {
                    "type": "tool_result",
                    "name": tu.name,
                    "is_error": is_error,
                }

            # tool-role message carrying all results for this iteration
            self._history.append(
                ProviderMessage(role="tool", content=result_blocks)
            )

            # Continue to next iteration — model gets the tool results
            # and decides what to say next.
        else:
            # for-else: loop exhausted without break (i.e. without an
            # end_turn). The model kept calling tools past the cap.
            log.warning(
                "Tool loop hit max iterations (%d); ending conversation",
                self._max_tool_iters,
            )

        full_response = "".join(full_response_chunks)
        yield {"type": "result", "text": full_response}

    # --- Helpers for tests / introspection ---

    @property
    def history(self) -> list[ProviderMessage]:
        """Read-only view of accumulated turn history. Useful for tests
        and for the future compaction trigger to inspect token counts."""
        return list(self._history)


# --- Default provider factory ----------------------------------------------


def _build_default_provider() -> LLMProvider:
    """Build the default NEAR provider from env.

    Phase C.1 has only one provider option for sessions. The selector
    pattern (per-stage provider choice) will extend here in C.3 when
    main pipeline staging arrives.
    """
    api_key = os.environ.get("NEAR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LLMSession requires NEAR_API_KEY (Phase C.1 only supports NEAR)"
        )
    base_url = (
        os.environ.get("NEAR_BASE_URL", "").strip()
        or "https://cloud-api.near.ai/v1"
    )
    return NEARProvider(api_key=api_key, base_url=base_url)
