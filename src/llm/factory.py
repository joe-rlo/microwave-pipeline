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


def build_private_tee_llm(config: Any, *, complexity: str = "moderate"):
    """Construct the Private-TEE LLM for the general_private_tee route.

    Returns an `LLMSession` backed by `NEARProvider` against one of
    NEAR's nearai-hosted open-weight models running in TEE-attested
    isolation. Returns None when NEAR_API_KEY isn't set — the router
    will have already selected this path based on a user pref, so the
    caller falls back gracefully (orchestrator downgrades to the
    standard `general` path with a log warning).

    Model picks per spec §5:
      - simple/moderate → Qwen/Qwen3.5-122B-A10B  (cheap, 131K ctx)
      - complex         → openai/gpt-oss-120b      (best open-weight)

    No tools wired on this path (same reasoning as BAA — health
    responses are evidence-grounded). Per-turn lifecycle owned by the
    orchestrator.
    """
    api_key = os.environ.get("NEAR_API_KEY", "").strip()
    if not api_key:
        log.warning(
            "Private-TEE route requested but NEAR_API_KEY unset; "
            "orchestrator will fall back to standard general path"
        )
        return None
    base_url = (
        os.environ.get("NEAR_BASE_URL", "").strip()
        or "https://cloud-api.near.ai/v1"
    )

    from src.llm.providers.near import NEARProvider
    from src.llm.session import LLMSession

    if complexity == "complex":
        model = "openai/gpt-oss-120b"
    else:
        model = "Qwen/Qwen3.5-122B-A10B"

    provider = NEARProvider(api_key=api_key, base_url=base_url)
    return LLMSession(
        model=model,
        provider=provider,
        tools=[],
        tool_handlers={},
    )


def build_baa_llm(config: Any):
    """Construct the BAA LLM for the health PHI route, or return None.

    Returns an `LLMSession` backed by `BedrockProvider` when the health
    module is enabled, `HEALTH_BAA_PROVIDER=bedrock`, AWS credentials
    are present in env, and the BAA model IDs are configured. Returns
    None in any other case — the router uses `decline_phi` instead.

    Tool wiring policy: the BAA path is privacy-restricted on purpose.
    External-service tools (webfetch / instacart / github / blink) are
    NEVER registered here — letting the model call them on a PHI turn
    would leak personal health data outside the BAA boundary. Only
    tools whose handlers touch the user's own local encrypted profile
    DB (and therefore can't exfiltrate) are exposed. Currently:

      - health_profile_summary / _show / _audit (read-only)

    New tools intended for the BAA path must be added explicitly to
    `_BAA_ALLOWED_TOOLS` below — opt-in, not name-prefix magic. A new
    health_* tool that talks to an external API would NOT belong on
    this list even though its name shares the prefix.

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
    from src.tools import build_provider_tools

    try:
        provider = BedrockProvider(region=region)
    except RuntimeError as e:
        log.error("Failed to construct BedrockProvider: %s", e)
        return None

    # Filter the registered provider tools down to the PHI-safe allowlist.
    # See class docstring above for the policy.
    all_tools = build_provider_tools(config)
    baa_tools = [t for t in all_tools if t.definition.name in _BAA_ALLOWED_TOOLS]
    if baa_tools:
        log.info(
            "[baa] Registered %d BAA-safe tool(s): %s",
            len(baa_tools),
            ", ".join(t.definition.name for t in baa_tools),
        )

    return LLMSession(
        model=health.baa_model_main,
        provider=provider,
        tools=[t.definition for t in baa_tools],
        tool_handlers={t.definition.name: t.handler for t in baa_tools},
    )


# Allowlist of tool names that may be called from the BAA path. Membership
# means the tool's handler does NOT leak PHI to any external service —
# typically because it only reads the user's own local encrypted DB.
# Adding here is an explicit privacy assertion; don't expand without
# verifying the handler's network surface.
_BAA_ALLOWED_TOOLS = frozenset({
    "health_profile_summary",
    "health_profile_show",
    "health_profile_audit",
})


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
