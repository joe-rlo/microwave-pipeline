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


def build_baa_llm(config: Any):
    """Construct the BAA LLM for the health PHI route, or return None.

    Returns an `LLMSession` backed by `BedrockProvider` when the health
    module is enabled, `HEALTH_BAA_PROVIDER=bedrock`, AWS credentials
    are present in env, and the BAA model IDs are configured. Returns
    None in any other case — the router uses `decline_phi` instead.

    No tools are wired on the BAA path. PHI responses are
    evidence-grounded via the spliced `[Evidence context]` block + the
    `health-qa` skill body. Cross-tool calls aren't part of the spec
    (and webfetch / instacart / github don't need to see PHI prompts).

    Lifecycle is per-turn (the orchestrator constructs, connects,
    sends, disconnects). That keeps PHI history isolated from the main
    pipeline's session — the privacy-correct default. Long-running
    BAA conversations across many turns are out of scope; the Health
    Profile system (future spec) handles cross-turn structured PHI
    via a different mechanism.
    """
    health = getattr(config, "health", None)
    if health is None or not health.phi_path_available:
        return None
    if health.baa_provider != "bedrock":
        log.warning(
            "BAA provider %r not supported yet; only 'bedrock' is wired",
            health.baa_provider,
        )
        return None

    region = os.environ.get("AWS_REGION", "").strip()
    if not region:
        log.error(
            "Health PHI route configured but AWS_REGION is not set; "
            "router will fall back to decline_phi"
        )
        return None

    from src.llm.providers.bedrock import BedrockProvider
    from src.llm.session import LLMSession

    try:
        provider = BedrockProvider(region=region)
    except RuntimeError as e:
        log.error("Failed to construct BedrockProvider: %s", e)
        return None

    return LLMSession(
        model=health.baa_model_main,
        provider=provider,
        tools=[],
        tool_handlers={},
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
