"""LLM client adapter. Pipeline sees only this interface.

Supports two auth paths:
- Max: Agent SDK using Claude Code session auth (no API key)
- API key: Standard Anthropic SDK (pay-as-you-go)

Both expose the same streaming interface to the pipeline.

Tool use (Max auth only). When the orchestrator passes a `tool_bundle`,
the SDK is configured with in-process MCP tools the model can invoke
mid-turn. The SDK handles the tool_use → tool_result round-trip
transparently — we just keep collecting TextBlocks and the model
weaves the tool's structured result into its reply.

API-key mode currently doesn't wire tools. Adding it would mean
managing the tool_use loop manually against the Messages API; not
worth the complexity until someone actually runs MicrowaveOS without
Max auth. The orchestrator surfaces this gap in the system prompt
so the model doesn't try to call tools that don't exist.
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

    def __init__(self, model: str = "sonnet", auth_mode: str = "max", api_key: str = "",
                 cli_path: str = "", output_dir: str = "", workspace_dir: str = "",
                 tool_bundle=None):
        self.model = model
        self.auth_mode = auth_mode
        self.api_key = api_key
        self.cli_path = cli_path
        self.output_dir = output_dir
        # Agent SDK cwd. Without this, the SDK runs in whatever directory
        # the bot process was launched from (typically the source repo),
        # so when the LLM writes a relative path like
        # `workspace/skills/foo/SKILL.md` it lands in the source tree
        # instead of the user's personal workspace. Pinning cwd to the
        # workspace's parent makes those relative writes resolve correctly.
        self.workspace_dir = workspace_dir
        # ToolBundle from src.tools.build_tools — None means no tools.
        # Stashed on the instance so reconnect() can rewire after a
        # stable-context refresh without the orchestrator re-passing it.
        self.tool_bundle = tool_bundle
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

            # In-process tools (if any). When tool_bundle is empty/None,
            # this collapses to the historical no-tool behavior — the
            # model treats responses as text-only and the pipeline still
            # extracts files from code fences after the fact.
            opts: dict = dict(
                system_prompt=self._stable_prompt,
                model=self.model,
                cli_path=self.cli_path or None,
            )
            # Pin the SDK's working directory to the user's workspace so
            # relative-path writes from tool calls land in the personal
            # workspace instead of wherever the bot was launched from.
            # The "parent" so a path like `workspace/skills/...` resolves
            # under `<home>/.microwaveos/workspace/skills/...`.
            if self.workspace_dir:
                from pathlib import Path as _Path
                ws = _Path(self.workspace_dir)
                opts["cwd"] = str(ws.parent)
                opts["add_dirs"] = [str(ws)]
            bundle = self.tool_bundle
            if bundle and not bundle.is_empty:
                opts["mcp_servers"] = bundle.mcp_servers
                opts["allowed_tools"] = list(bundle.allowed_tools)
                log.info(
                    "Tools enabled: %s",
                    ", ".join(bundle.allowed_tools),
                )
            else:
                opts["allowed_tools"] = []

            options = ClaudeAgentOptions(**opts)
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
        self,
        user_message: str,
        memory_context: str | None = None,
        images: list[tuple[bytes, str]] | None = None,
    ) -> AsyncIterator[dict]:
        """Send a message and yield streaming chunks.

        Per-turn memory fragments from Stage 2 are prepended to the
        user message, not injected into the system prompt.

        `images` is an optional list of (bytes, content_type) tuples.
        When present, the message is sent as a multimodal Anthropic
        content block array via the api_key path. The Agent SDK (max
        auth) currently has no clean multimodal surface — we log a
        warning and fall through to text-only there. Users on Max who
        need vision should switch AUTH_MODE=api_key.
        """
        if memory_context:
            enriched = f"[Relevant memory context]\n{memory_context}\n\n{user_message}"
        else:
            enriched = user_message

        if self.auth_mode == "max":
            if images:
                log.warning(
                    "LLMClient: %d image(s) received but Max auth path doesn't "
                    "support multimodal; sending text-only. Switch to "
                    "AUTH_MODE=api_key for vision.",
                    len(images),
                )
            async for chunk in self._send_max(enriched):
                yield chunk
        else:
            async for chunk in self._send_api_key(enriched, images=images):
                yield chunk

    async def _send_max(self, message: str) -> AsyncIterator[dict]:
        """Send via Agent SDK persistent session.

        Streams TextBlocks as they arrive. When a tool is wired in,
        the model may emit ToolUseBlocks that the SDK invokes
        transparently — we surface a `{"type": "tool_use"}` chunk for
        channel visibility (typing indicator, debug overlay) but the
        actual tool I/O happens inside the SDK and feeds a fresh
        AssistantMessage back into the same `receive_response()` stream.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        # ToolUseBlock isn't always exported in older SDK builds; pull
        # it lazily so we don't break clients that don't have it.
        try:
            from claude_agent_sdk import ToolUseBlock
        except ImportError:
            ToolUseBlock = None  # type: ignore[assignment]

        await self._client.query(message)

        full_response = ""
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_response += block.text
                        yield {"type": "text", "chunk": block.text}
                    elif ToolUseBlock is not None and isinstance(block, ToolUseBlock):
                        # Surface tool calls for channels that want to
                        # render a "running tool…" state. The SDK has
                        # already started executing — we don't need to
                        # do anything actionable here.
                        yield {
                            "type": "tool_use",
                            "tool": getattr(block, "name", "unknown"),
                            "tool_use_id": getattr(block, "id", None),
                        }
                    # Other block types (ThinkingBlock, ToolResultBlock
                    # echoes) are deliberately skipped — they're
                    # internal to the SDK loop and the user-visible
                    # response is conveyed via subsequent TextBlocks.

            elif isinstance(msg, ResultMessage):
                yield {
                    "type": "result",
                    "text": full_response,
                    "session_id": getattr(msg, "session_id", None),
                    "cost_usd": getattr(msg, "total_cost_usd", None),
                    "duration_ms": getattr(msg, "duration_ms", None),
                }

    async def _send_api_key(
        self,
        message: str,
        images: list[tuple[bytes, str]] | None = None,
    ) -> AsyncIterator[dict]:
        """Send via standard Anthropic Messages API with streaming.

        When `images` is present, the user content becomes a list of
        content blocks: image blocks first (so the text can refer to
        "this photo"), then a single text block. The conversation
        history stores the same shape so subsequent turns inherit the
        visual context.
        """
        import base64 as _base64

        MODEL_MAP = {
            "sonnet": "claude-sonnet-4-6-20260411",
            "opus": "claude-opus-4-6-20260411",
            "haiku": "claude-haiku-4-5-20251001",
        }
        model_id = MODEL_MAP.get(self.model, self.model)

        if images:
            content_blocks: list[dict] = []
            for img_bytes, ct in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": ct or "image/jpeg",
                        "data": _base64.b64encode(img_bytes).decode("ascii"),
                    },
                })
            content_blocks.append({"type": "text", "text": message})
            self._conversation.append({"role": "user", "content": content_blocks})
        else:
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

    def __init__(self, model: str = "haiku", auth_mode: str = "max", api_key: str = "",
                 cli_path: str = "", workspace_dir: str = ""):
        self.model = model
        self.auth_mode = auth_mode
        self.api_key = api_key
        self.cli_path = cli_path
        # Agent SDK cwd hint — same purpose as on LLMClient: pin relative
        # writes to the user's workspace, not the source repo. SingleTurn
        # callers (triage, reflection, scheduler one-shots, memory-health)
        # mostly don't write files, but setting cwd consistently means a
        # future caller that does will land in the right place.
        self.workspace_dir = workspace_dir

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

        opts: dict = dict(
            system_prompt=system_prompt,
            allowed_tools=[],
            model=self.model,
            cli_path=self.cli_path or None,
        )
        if self.workspace_dir:
            from pathlib import Path as _Path
            ws = _Path(self.workspace_dir)
            opts["cwd"] = str(ws.parent)
            opts["add_dirs"] = [str(ws)]
        options = ClaudeAgentOptions(**opts)

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
