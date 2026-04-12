"""LLM client adapter. Pipeline sees only this interface.

Supports two auth paths:
- Max: Agent SDK using Claude Code session auth (no API key)
- API key: Standard Anthropic SDK (pay-as-you-go)

Both expose the same streaming interface to the pipeline.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)


class LLMClient:
    """Thin adapter around Agent SDK or Messages API.

    For Max auth: uses ClaudeSDKClient with persistent session.
    The SDK manages multi-turn conversation state internally.
    """

    def __init__(self, model: str = "sonnet", auth_mode: str = "max", api_key: str = "", cli_path: str = ""):
        self.model = model
        self.auth_mode = auth_mode
        self.api_key = api_key
        self.cli_path = cli_path
        self._client = None
        self._stable_prompt: str | None = None
        self._conversation: list[dict] = []

    async def connect(self, stable_prompt: str) -> None:
        """Connect with stable context as the system prompt."""
        self._stable_prompt = stable_prompt
        self._conversation = []

        if self.auth_mode == "max":
            await self._connect_max()
        else:
            await self._connect_api_key()

    async def _connect_max(self) -> None:
        """Connect via Agent SDK using Max subscription auth."""
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

            options = ClaudeAgentOptions(
                system_prompt=self._stable_prompt,
                allowed_tools=[],
                model=self.model,
                cli_path=self.cli_path or None,
            )
            self._client = ClaudeSDKClient(options)
            await self._client.connect()
            log.info("Connected via Agent SDK (Max auth)")
        except ImportError:
            log.warning("claude_agent_sdk not installed, falling back to API key mode")
            self.auth_mode = "api_key"
            await self._connect_api_key()

    async def _connect_api_key(self) -> None:
        """Connect via standard Anthropic SDK."""
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
            log.info("Connected via Anthropic SDK (API key auth)")
        except ImportError:
            raise RuntimeError("anthropic package not installed")

    async def escalate(self, model: str, effort: str = "high") -> None:
        """Temporarily switch to a stronger model with extended thinking.

        Call before send() for the escalated turn, then call de_escalate()
        to restore the default model.
        """
        self._base_model = self.model
        self.model = model

        if self.auth_mode == "max" and self._client:
            try:
                await self._client.set_model(model)
                log.info(f"Escalated to {model} (effort={effort})")
            except Exception as e:
                log.warning(f"set_model failed: {e}")
        self._escalation_effort = effort

    async def de_escalate(self) -> None:
        """Restore the default model after an escalated turn."""
        if hasattr(self, "_base_model"):
            self.model = self._base_model
            if self.auth_mode == "max" and self._client:
                try:
                    await self._client.set_model(self._base_model)
                    log.info(f"De-escalated back to {self._base_model}")
                except Exception as e:
                    log.warning(f"set_model failed: {e}")
            del self._base_model
        self._escalation_effort = None

    async def send(
        self, user_message: str, memory_context: str | None = None
    ) -> AsyncIterator[dict]:
        """Send a message and yield streaming chunks.

        Per-turn memory fragments from Stage 2 are prepended to the
        user message, not injected into the system prompt.
        """
        if memory_context:
            enriched = f"[Relevant memory context]\n{memory_context}\n\n{user_message}"
        else:
            enriched = user_message

        if self.auth_mode == "max":
            async for chunk in self._send_max(enriched):
                yield chunk
        else:
            async for chunk in self._send_api_key(enriched):
                yield chunk

    async def _send_max(self, message: str) -> AsyncIterator[dict]:
        """Send via Agent SDK persistent session."""
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        await self._client.query(message)

        full_response = ""
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_response += block.text
                        yield {"type": "text", "chunk": block.text}

            elif isinstance(msg, ResultMessage):
                yield {
                    "type": "result",
                    "text": full_response,
                    "session_id": getattr(msg, "session_id", None),
                    "cost_usd": getattr(msg, "total_cost_usd", None),
                    "duration_ms": getattr(msg, "duration_ms", None),
                }

    async def _send_api_key(self, message: str) -> AsyncIterator[dict]:
        """Send via standard Anthropic Messages API with streaming."""
        MODEL_MAP = {
            "sonnet": "claude-sonnet-4-6-20260411",
            "opus": "claude-opus-4-6-20260411",
            "haiku": "claude-haiku-4-5-20251001",
        }
        model_id = MODEL_MAP.get(self.model, self.model)

        self._conversation.append({"role": "user", "content": message})

        # Build request kwargs
        effort = getattr(self, "_escalation_effort", None)
        kwargs = dict(
            model=model_id,
            max_tokens=16384 if effort else 8192,
            system=self._stable_prompt,
            messages=self._conversation,
        )

        # Extended thinking for escalated turns
        if effort:
            THINKING_BUDGET = {
                "low": 2048,
                "medium": 8192,
                "high": 32768,
                "max": 65536,
            }
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET.get(effort, 32768),
            }

        full_response = ""
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                full_response += text
                yield {"type": "delta", "text": text}

        self._conversation.append({"role": "assistant", "content": full_response})
        yield {"type": "result", "text": full_response}

    async def reconnect(self, stable_prompt: str) -> None:
        """Reconnect with updated stable context.

        Call when MEMORY.md changes, day rolls over, or after compaction.
        Starts a fresh session.
        """
        await self.disconnect()
        await self.connect(stable_prompt)

    async def disconnect(self) -> None:
        if self._client and self.auth_mode == "max":
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._conversation = []


class SingleTurnClient:
    """Short-lived client for pipeline stages (triage, reflection, compaction).

    Uses the query() function for simple one-shot interactions.
    """

    def __init__(self, model: str = "haiku", auth_mode: str = "max", api_key: str = "", cli_path: str = ""):
        self.model = model
        self.auth_mode = auth_mode
        self.api_key = api_key
        self.cli_path = cli_path

    async def query(self, system_prompt: str, user_message: str) -> str:
        """Single-turn query. Returns the full response text."""
        if self.auth_mode == "max":
            return await self._query_max(system_prompt, user_message)
        else:
            return await self._query_api_key(system_prompt, user_message)

    async def _query_max(self, system_prompt: str, user_message: str) -> str:
        """One-shot query via Agent SDK query() function."""
        try:
            from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
            from claude_agent_sdk import AssistantMessage, TextBlock
        except ImportError:
            log.warning("Agent SDK not available, falling back to API key")
            self.auth_mode = "api_key"
            return await self._query_api_key(system_prompt, user_message)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=[],
            model=self.model,
            cli_path=self.cli_path or None,
        )

        response = ""
        async for msg in sdk_query(prompt=user_message, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response += block.text

        return response

    async def _query_api_key(self, system_prompt: str, user_message: str) -> str:
        from anthropic import AsyncAnthropic

        MODEL_MAP = {
            "sonnet": "claude-sonnet-4-6-20260411",
            "opus": "claude-opus-4-6-20260411",
            "haiku": "claude-haiku-4-5-20251001",
        }
        model_id = MODEL_MAP.get(self.model, self.model)

        client = AsyncAnthropic(api_key=self.api_key)
        message = await client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text
