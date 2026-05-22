"""Main-LLM factory — pick LLMClient (legacy) or LLMSession (new).

Reads the env override `LLM_STAGE_MAIN` and returns whichever
implementation matches. Both expose the same surface — connect,
reconnect, disconnect, escalate, de_escalate, send — so the
orchestrator stays oblivious to which one it's holding.

Phase C.3: env-gated cutover. With `LLM_STAGE_MAIN` empty (the
default), this returns the existing `LLMClient` — zero behavior
change. With `LLM_STAGE_MAIN=near:claude-sonnet-4-6` and
`NEAR_API_KEY` set, returns an `LLMSession` wired with provider-shape
tools from `build_provider_tools()`.

Why env-driven instead of a Config flag the orchestrator inspects:
the rest of the provider stack (selector, session factory) already
reads env directly. Mirroring it in Config and then plumbing through
the orchestrator would be three places to keep in sync; env is the
canonical surface.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def build_main_llm(config: Any):
    """Construct the main-LLM object the orchestrator holds in `self.llm`.

    Returns either an `LLMClient` (legacy Agent SDK / direct Anthropic
    path) or an `LLMSession` (new provider abstraction path). Both
    implement: connect, reconnect, disconnect, escalate, de_escalate,
    send.
    """
    override = os.environ.get("LLM_STAGE_MAIN", "").strip()

    if override:
        provider, _, model = override.partition(":")
        provider = provider.strip()
        model = model.strip()

        if provider == "near":
            return _build_near_session(config, model)

        # Unknown override → log and fall through to legacy. Better than
        # crashing startup; the user can fix env and restart.
        log.warning(
            "LLM_STAGE_MAIN=%r has unknown provider %r; falling back to legacy",
            override, provider,
        )

    return _build_legacy_client(config)


def _build_legacy_client(config: Any):
    """Construct the pre-Phase-C LLMClient. Unchanged behavior."""
    from src.llm.client import LLMClient
    from src.tools import build_tools

    tool_bundle = build_tools(config)
    if tool_bundle.allowed_tools:
        log.info(
            "[legacy] Registered %d MCP tool(s): %s",
            len(tool_bundle.allowed_tools),
            ", ".join(tool_bundle.allowed_tools),
        )

    return LLMClient(
        model=config.model_main,
        auth_mode=config.auth_mode,
        api_key=config.anthropic_api_key,
        cli_path=config.cli_path,
        output_dir=str(config.output_dir),
        workspace_dir=str(config.workspace_dir),
        tool_bundle=tool_bundle,
        builtin_tools=list(config.bot_builtin_tools),
    )


def _build_near_session(config: Any, model_override: str):
    """Construct an LLMSession backed by NEAR.

    Pulls tools from `build_provider_tools(config)` — same Instacart /
    GitHub handlers as the legacy path, just in (ToolDefinition,
    handler) shape. Built-in tools (Bash/Read/Write/etc.) are NOT
    available on the new path; the orchestrator's stable-context
    catalog should reflect that. Web-tool re-implementation comes
    in Phase C.4 alongside the SDK removal.
    """
    from src.llm.session import LLMSession
    from src.tools import build_provider_tools

    provider_tools = build_provider_tools(config)
    tool_defs = [pt.definition for pt in provider_tools]
    tool_handlers = {pt.definition.name: pt.handler for pt in provider_tools}

    if provider_tools:
        log.info(
            "[near] Registered %d tool(s): %s",
            len(provider_tools),
            ", ".join(pt.definition.name for pt in provider_tools),
        )
    if config.bot_builtin_tools:
        log.warning(
            "BOT_BUILTIN_TOOLS=%s is set but the NEAR path doesn't expose "
            "Agent SDK built-ins. The tools won't be available this turn. "
            "Native web tools land in Phase C.4.",
            ",".join(config.bot_builtin_tools),
        )

    # Default model resolution: env override wins; otherwise config.model_main.
    model = model_override or config.model_main

    # NEAR_API_KEY is checked at LLMSession construction time
    # (via _build_default_provider). If it's missing, that raises a
    # RuntimeError — same fail-fast posture as triage / reflection.
    return LLMSession(
        model=model,
        tools=tool_defs,
        tool_handlers=tool_handlers,
    )
