"""Multi-turn LLM session over the provider abstraction.

What this replaces: the persistent-connection behavior of the Agent
SDK's `ClaudeSDKClient` — system-prompt-once, then a sequence of
turns with growing history. NEAR / Anthropic / OpenAI all support
this natively via stateless HTTP + we-send-the-history-each-call.

What this DOES NOT do (yet — Phase C.2 / C.3):
- Tool use. Phase C.1 streams plain text only. `tool_loop.py` (next
  commit) will wrap this session to handle tool_use / tool_result
  round-trips before the user sees the final response.
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
from typing import AsyncIterator

from src.llm.provider import (
    Done,
    Error,
    LLMProvider,
    ProviderMessage,
    ProviderRequest,
    TextDelta,
    ToolUseStart,
    Usage,
)
from src.llm.providers.near import NEARProvider

log = logging.getLogger(__name__)


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
    ):
        """Build a session.

        `provider` is injected for tests. Production uses the env-driven
        factory at module bottom: `build_default_session()`.
        """
        self.model = model
        self._base_model = model
        self._provider = provider if provider is not None else _build_default_provider()
        self._max_tokens = max_tokens

        self._system_prompt: str | None = None
        self._history: list[ProviderMessage] = []
        self._thinking_budget: int | None = None

    # --- Lifecycle ---

    async def connect(self, stable_prompt: str) -> None:
        self._system_prompt = stable_prompt
        self._history = []
        log.info("LLMSession connected (model=%s)", self.model)

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
        """Stream a turn. Yields LLMClient-shape dict chunks.

        Per-turn memory context is prepended to the user message (same
        as existing LLMClient behavior — see
        src/llm/client.py line 215).
        """
        if self._system_prompt is None:
            raise RuntimeError("LLMSession.send() called before connect()")

        if images:
            # Phase C.1 limitation — flagged in module docstring. Log
            # loudly so we don't silently drop images.
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

        req = ProviderRequest(
            model=self.model,
            system=self._system_prompt,
            messages=list(self._history),
            max_tokens=self._max_tokens,
            thinking_budget=self._thinking_budget,
            stream=True,
            metadata={"stage": "main"},
        )

        full_response_chunks: list[str] = []
        error: Error | None = None
        async for evt in self._provider.send(req):
            if isinstance(evt, TextDelta):
                full_response_chunks.append(evt.text)
                yield {"type": "delta", "text": evt.text}
            elif isinstance(evt, ToolUseStart):
                # Visibility only at C.1 — tool calls don't actually
                # round-trip until C.2's tool_loop wraps this method.
                yield {"type": "tool_use", "name": evt.name}
            elif isinstance(evt, Usage):
                # Don't yield — usage isn't part of the LLMClient contract.
                # Phase C.3 may wire this to telemetry; for now we drop it.
                pass
            elif isinstance(evt, Done):
                # Provider's terminal event. We emit the orchestrator's
                # `result` chunk below regardless of stop_reason.
                pass
            elif isinstance(evt, Error):
                error = evt

        full_response = "".join(full_response_chunks)

        if error is not None and not full_response:
            # Stream produced nothing and reported an error — surface as
            # exception so the orchestrator's try/finally can clean up.
            raise RuntimeError(
                f"LLMSession.send failed: {error.message} "
                f"(status={error.status}, retryable={error.retryable})"
            )

        # Append assistant turn to history so the next send() sees it.
        if full_response:
            self._history.append(
                ProviderMessage(role="assistant", content=full_response)
            )

        # Final marker — orchestrator keys on this.
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
