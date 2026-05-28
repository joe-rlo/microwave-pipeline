"""Tool registry — dual-shape during the SDK-to-provider migration.

Two surfaces live here in parallel during Phase C:

1. **Agent SDK shape** (`build_tools(config) -> ToolBundle`) — the
   pre-existing path. Returns `SdkMcpTool` objects bundled into an
   in-process MCP server the SDK can mount. Used by `src/llm/client.py`'s
   Max-auth path.

2. **Provider shape** (`build_provider_tools(config) -> list[ProviderTool]`)
   — the Phase C path. Returns plain (`ToolDefinition`, `handler`) pairs
   for `src/llm/session.py`'s tool loop. Same JSON schemas, same
   underlying handlers — just unwrapped from the SDK envelope.

The provider-shape handlers raise exceptions on tool errors instead
of returning `{"is_error": True}` dicts; the session's loop catches
those and reports `is_error=True` to the model. Same semantics, less
ceremony.

Naming convention (SDK shape only). The Agent SDK exposes in-process
tools as `mcp__<server-name>__<tool-name>`. We use one server named
`microwave` so all SDK tools land at `mcp__microwave__<tool-name>`.
The provider shape uses bare names — `instacart_create_cart`, not
`mcp__microwave__instacart_create_cart`. That's because the provider
itself doesn't know about MCP — it talks to the model in OpenAI /
Anthropic native shape, where tool names are unqualified.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.llm.provider import ToolDefinition

log = logging.getLogger(__name__)


def _web_tools_disabled() -> bool:
    """True when the WEB_TOOLS_DISABLED env flag is set (escape hatch for
    isolated tests / paranoid deployments). Default is enabled."""
    return os.environ.get("WEB_TOOLS_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _file_tools_disabled() -> bool:
    """True when the FILE_TOOLS_DISABLED env flag is set. Default is enabled."""
    return os.environ.get("FILE_TOOLS_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _websearch_disabled() -> bool:
    """True when the WEBSEARCH_DISABLED env flag is set. Default is enabled."""
    return os.environ.get("WEBSEARCH_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Provider-shape handler: async (args) -> response text. Raises on error.
ProviderToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ProviderTool:
    """One tool, in the provider abstraction's shape.

    `definition` is what the provider sends to the model so it knows
    the tool exists. `handler` is what we run when the model calls it.
    """

    definition: ToolDefinition
    handler: ProviderToolHandler


# Single in-process MCP server name. Tools are exposed as
# `mcp__microwave__<name>` in the SDK's tool list.
MCP_SERVER_NAME = "microwave"


@dataclass
class ToolBundle:
    """What the orchestrator gets back from `build_tools(config)`.

    `mcp_servers` is the dict you hand to `ClaudeAgentOptions(mcp_servers=...)`.
    `allowed_tools` is the full list of qualified tool names — empty if
    no tools were registered. The LLM client unions this with anything
    else the pipeline wants to allow (e.g. built-in SDK tools, which we
    currently don't enable).
    """

    mcp_servers: dict[str, Any]
    allowed_tools: list[str]
    # Human-readable summary for the system prompt — tells the LLM
    # which tools exist and how to call them. Goes into the dynamic
    # context, not the stable prompt, because tool availability can
    # change between turns (skill activation, missing API keys).
    catalog_text: str

    @property
    def is_empty(self) -> bool:
        return not self.allowed_tools


def build_tools(config) -> ToolBundle:
    """Inspect config, register every tool whose prerequisites are met,
    and return a ToolBundle the orchestrator can plug into the LLM.

    Tools self-register based on what's configured. Instacart only
    appears if `INSTACART_API_KEY` is set; otherwise the LLM never
    sees it advertised and won't try to call it. Failing closed beats
    surfacing a tool that errors on every invocation.
    """
    try:
        from claude_agent_sdk import create_sdk_mcp_server
    except ImportError:
        log.info("claude_agent_sdk not available; tools disabled")
        return ToolBundle(mcp_servers={}, allowed_tools=[], catalog_text="")

    tools: list = []
    catalog_lines: list[str] = []

    # --- Instacart ---
    if getattr(config, "instacart_api_key", ""):
        from src.tools.instacart import build_instacart_tools, INSTACART_TOOL_DOCS

        tools.extend(build_instacart_tools(config))
        catalog_lines.append(INSTACART_TOOL_DOCS)
        log.info("Registered Instacart tools")
    else:
        log.debug("INSTACART_API_KEY not set; Instacart tool disabled")

    # --- GitHub ---
    if getattr(config, "github_token", ""):
        from src.tools.github import build_github_tools, GITHUB_TOOL_DOCS

        tools.extend(build_github_tools(config))
        catalog_lines.append(GITHUB_TOOL_DOCS)
        log.info("Registered GitHub tools")
    else:
        log.debug("GITHUB_TOKEN not set; GitHub tools disabled")

    # --- Web tools (native, replaces SDK built-ins) ---
    if not _web_tools_disabled():
        from src.tools.web import build_webfetch_sdk_tools, WEB_TOOL_DOCS

        web_tools = build_webfetch_sdk_tools()
        if web_tools:
            tools.extend(web_tools)
            catalog_lines.append(WEB_TOOL_DOCS)
            log.info("Registered native web tools (webfetch)")

    # --- File tools (native, replaces SDK Read; Phase C.5) ---
    workspace_dir = getattr(config, "workspace_dir", None)
    if workspace_dir is not None and not _file_tools_disabled():
        from src.tools.files import build_read_sdk_tools, WEBFETCH_TOOL_DOCS

        file_tools = build_read_sdk_tools(workspace_dir)
        if file_tools:
            tools.extend(file_tools)
            catalog_lines.append(WEBFETCH_TOOL_DOCS)
            log.info("Registered native file tools (read_file)")

    # --- Web search (native, replaces SDK WebSearch; Phase C.5 part 2) ---
    if not _websearch_disabled():
        from src.tools.search import build_websearch_sdk_tools, WEBSEARCH_TOOL_DOCS

        ws_tools = build_websearch_sdk_tools()
        if ws_tools:
            tools.extend(ws_tools)
            catalog_lines.append(WEBSEARCH_TOOL_DOCS)
            log.info("Registered native websearch tool")

    # --- Scheduler docs (no SDK tool, provider-path only) -----------------
    # The scheduler tool only exists on the provider path (build_provider_tools
    # below). But the LLM reads its tool catalog from this function's
    # `catalog_text`, so we splice the docs in here. No SDK MCP tool is
    # registered for it — when the SDK path is finally removed in Phase C.4,
    # this whole branch goes away cleanly.
    from src.tools.scheduler import SCHEDULER_TOOL_DOCS

    catalog_lines.append(SCHEDULER_TOOL_DOCS)

    # --- Blink docs (provider-path only, gated on creds file) -------------
    # Same provider-only pattern as the scheduler tool above. Gated so the
    # LLM doesn't see Blink tools advertised when there's no way to call
    # them (no creds → tool would error on every invocation).
    from src.tools import blink as blink_mod

    if blink_mod.credentials_available():
        catalog_lines.append(blink_mod.BLINK_TOOL_DOCS)

    # --- Health profile docs (provider-path only, gated on health enabled) -
    from src.tools import health_profile as hp_mod

    if hp_mod.health_module_available(config):
        catalog_lines.append(hp_mod.HEALTH_PROFILE_TOOL_DOCS)

    if not tools:
        # Catalog still ships even without SDK tools — the provider path
        # may have tools (like scheduler) that aren't SDK-registered.
        return ToolBundle(
            mcp_servers={},
            allowed_tools=[],
            catalog_text="\n\n".join(catalog_lines),
        )

    server = create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="0.1.0",
        tools=tools,
    )
    allowed = [_qualified_name(t) for t in tools]
    catalog_text = "\n\n".join(catalog_lines)
    return ToolBundle(
        mcp_servers={MCP_SERVER_NAME: server},
        allowed_tools=allowed,
        catalog_text=catalog_text,
    )


def _qualified_name(sdk_tool) -> str:
    """Turn an SdkMcpTool into its fully-qualified `mcp__server__name`."""
    # `SdkMcpTool` exposes `.name` as the bare tool name. The SDK
    # qualifies it with the server name when surfacing to the model.
    bare = getattr(sdk_tool, "name", None)
    if not bare:
        # Fallback for unexpected SDK shapes — should never hit in
        # practice, but better than crashing the orchestrator.
        bare = getattr(sdk_tool, "__name__", "unknown")
    return f"mcp__{MCP_SERVER_NAME}__{bare}"


# === Provider-shape registry ================================================
#
# Returns the same logical tools as build_tools() but in the shape
# LLMSession's tool loop consumes: (ToolDefinition, async handler).
# This is the Phase C path; build_tools() above stays for the Agent
# SDK path until C.4 drops it entirely.


def build_provider_tools(config) -> list[ProviderTool]:
    """Build (ToolDefinition, handler) pairs from configured tools."""
    out: list[ProviderTool] = []

    # --- Instacart ---
    if getattr(config, "instacart_api_key", ""):
        from src.tools import instacart as instacart_mod

        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="instacart_create_cart",
                    description=(
                        "Build a Shop with Instacart cart from a list of "
                        "items and return a checkout URL."
                    ),
                    input_schema=instacart_mod.INSTACART_CREATE_CART_SCHEMA,
                ),
                handler=_instacart_handler(config),
            )
        )

    # --- Web tools (native; no env key required) ---
    if not _web_tools_disabled():
        from src.tools import web as web_mod

        async def _webfetch(args: dict[str, Any]) -> str:
            return _unwrap_mcp_result(
                await web_mod._handle_webfetch(args),
                tool_name="webfetch",
            )

        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="webfetch",
                    description=(
                        "Fetch a public web page and return its content as "
                        "plain text. Use when the user shares a URL or you "
                        "need information from a specific public page."
                    ),
                    input_schema=web_mod.WEBFETCH_SCHEMA,
                ),
                handler=_webfetch,
            )
        )

    # --- File tools (native; no env key required; workspace-sandboxed) ---
    workspace_dir = getattr(config, "workspace_dir", None)
    if workspace_dir is not None and not _file_tools_disabled():
        from src.tools import files as files_mod

        # Capture workspace_dir in closure so the handler doesn't need
        # to re-read config at call time.
        _ws = workspace_dir

        async def _read_file(args: dict[str, Any]) -> str:
            return _unwrap_mcp_result(
                await files_mod._handle_read_file(args, workspace_dir=_ws),
                tool_name="read_file",
            )

        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="read_file",
                    description=(
                        "Read a text file from the user's workspace. Paths "
                        "are resolved against the workspace directory; "
                        "absolute paths outside the workspace are rejected."
                    ),
                    input_schema=files_mod.READ_FILE_SCHEMA,
                ),
                handler=_read_file,
            )
        )

    # --- Web search (native; pluggable backend; default DDG) ---
    if not _websearch_disabled():
        from src.tools import search as search_mod

        async def _websearch(args: dict[str, Any]) -> str:
            return _unwrap_mcp_result(
                await search_mod._handle_websearch(args),
                tool_name="websearch",
            )

        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="websearch",
                    description=(
                        "Search the public web. Returns up to N "
                        "{title, url, snippet} results. Backend is "
                        "pluggable via WEBSEARCH_BACKEND env (default ddg)."
                    ),
                    input_schema=search_mod.WEBSEARCH_SCHEMA,
                ),
                handler=_websearch,
            )
        )

    # --- Health profile (gated on HEALTH_MODULE_ENABLED) ---
    # All three handlers are read-only; destructive ops stay behind the
    # `/profile clear` slash command which requires an explicit phrase.
    from src.tools import health_profile as hp_mod

    if hp_mod.health_module_available(config):
        def _make_hp_handler(fn):
            async def call(args: dict[str, Any]) -> str:
                return await fn(args, config=config)
            return call

        out.extend([
            ProviderTool(
                definition=ToolDefinition(
                    name="health_profile_summary",
                    description="Read a high-level snapshot of the user's health profile: section counts + last updated.",
                    input_schema=hp_mod.HEALTH_PROFILE_SUMMARY_SCHEMA,
                ),
                handler=_make_hp_handler(hp_mod._handle_summary),
            ),
            ProviderTool(
                definition=ToolDefinition(
                    name="health_profile_show",
                    description="Get the detailed contents of one profile section (medications, conditions, allergies, etc.).",
                    input_schema=hp_mod.HEALTH_PROFILE_SHOW_SCHEMA,
                ),
                handler=_make_hp_handler(hp_mod._handle_show),
            ),
            ProviderTool(
                definition=ToolDefinition(
                    name="health_profile_audit",
                    description="Recent change-log entries for the profile (op + section + timestamp).",
                    input_schema=hp_mod.HEALTH_PROFILE_AUDIT_SCHEMA,
                ),
                handler=_make_hp_handler(hp_mod._handle_audit),
            ),
        ])

    # --- Blink (gated on credentials file presence) ---
    from src.tools import blink as blink_mod

    if blink_mod.credentials_available():
        out.extend([
            ProviderTool(
                definition=ToolDefinition(
                    name="blink_status",
                    description="Read Blink system snapshot (networks armed, cameras battery/wifi/temp, sync modules).",
                    input_schema=blink_mod.BLINK_STATUS_SCHEMA,
                ),
                handler=blink_mod._handle_status,
            ),
            ProviderTool(
                definition=ToolDefinition(
                    name="blink_arm",
                    description="Arm a Blink network by name (turns motion detection ON).",
                    input_schema=blink_mod.BLINK_ARM_SCHEMA,
                ),
                handler=blink_mod._handle_arm,
            ),
            ProviderTool(
                definition=ToolDefinition(
                    name="blink_disarm",
                    description="Disarm a Blink network by name (turns motion detection OFF).",
                    input_schema=blink_mod.BLINK_DISARM_SCHEMA,
                ),
                handler=blink_mod._handle_disarm,
            ),
            ProviderTool(
                definition=ToolDefinition(
                    name="blink_snap",
                    description="Trigger a fresh thumbnail capture on a Blink camera by name.",
                    input_schema=blink_mod.BLINK_SNAP_SCHEMA,
                ),
                handler=blink_mod._handle_snap,
            ),
        ])

    # --- Scheduler (always on — same DB the daemon already uses) ---
    from src.tools import scheduler as sched_mod

    def _make_sched_handler(fn):
        async def call(args: dict[str, Any]) -> str:
            return await fn(args, config=config)
        return call

    out.extend([
        ProviderTool(
            definition=ToolDefinition(
                name="scheduler_list",
                description="List all scheduled jobs (name, cron, mode, channel, enabled, last-run).",
                input_schema=sched_mod.SCHEDULER_LIST_SCHEMA,
            ),
            handler=_make_sched_handler(sched_mod._handle_list),
        ),
        ProviderTool(
            definition=ToolDefinition(
                name="scheduler_get",
                description="Get full details on one scheduled job by name.",
                input_schema=sched_mod.SCHEDULER_GET_SCHEMA,
            ),
            handler=_make_sched_handler(sched_mod._handle_get),
        ),
        ProviderTool(
            definition=ToolDefinition(
                name="scheduler_add",
                description="Create a new scheduled job (mode: llm | direct | script).",
                input_schema=sched_mod.SCHEDULER_ADD_SCHEMA,
            ),
            handler=_make_sched_handler(sched_mod._handle_add),
        ),
        ProviderTool(
            definition=ToolDefinition(
                name="scheduler_remove",
                description="Delete a scheduled job by name.",
                input_schema=sched_mod.SCHEDULER_REMOVE_SCHEMA,
            ),
            handler=_make_sched_handler(sched_mod._handle_remove),
        ),
        ProviderTool(
            definition=ToolDefinition(
                name="scheduler_set_enabled",
                description="Enable or disable a scheduled job by name (keeps the row).",
                input_schema=sched_mod.SCHEDULER_SET_ENABLED_SCHEMA,
            ),
            handler=_make_sched_handler(sched_mod._handle_set_enabled),
        ),
    ])

    # --- GitHub ---
    if getattr(config, "github_token", ""):
        from src.tools import github as github_mod

        token = config.github_token
        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="github_list_repos",
                    description=(
                        "List GitHub repos visible to the authenticated user."
                    ),
                    input_schema=github_mod.LIST_REPOS_SCHEMA,
                ),
                handler=_github_handler(github_mod._handle_list_repos, token),
            )
        )
        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="github_repo_summary",
                    description=(
                        "Get a composite snapshot of one GitHub repo "
                        "(metadata + README + commits + PRs/issues + langs)."
                    ),
                    input_schema=github_mod.REPO_SUMMARY_SCHEMA,
                ),
                handler=_github_handler(github_mod._handle_repo_summary, token),
            )
        )
        out.append(
            ProviderTool(
                definition=ToolDefinition(
                    name="github_recent_activity",
                    description=(
                        "Recent push/PR/issue/release events for the "
                        "authenticated user across all repos."
                    ),
                    input_schema=github_mod.RECENT_ACTIVITY_SCHEMA,
                ),
                handler=_github_handler(github_mod._handle_recent_activity, token),
            )
        )

    return out


def _instacart_handler(config) -> ProviderToolHandler:
    """Wrap the existing instacart `_handle_create_cart` for provider shape."""
    from src.tools.instacart import _handle_create_cart

    api_key = config.instacart_api_key
    linkback = getattr(config, "instacart_partner_linkback_url", "") or None

    async def call(args: dict[str, Any]) -> str:
        mcp_result = await _handle_create_cart(args, api_key, linkback)
        return _unwrap_mcp_result(mcp_result, tool_name="instacart_create_cart")

    return call


def _github_handler(
    underlying: Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]],
    token: str,
) -> ProviderToolHandler:
    """Wrap a github `_handle_*` so its (args, token) -> mcp-dict shape
    looks like the provider's (args) -> str."""

    async def call(args: dict[str, Any]) -> str:
        mcp_result = await underlying(args, token)
        return _unwrap_mcp_result(mcp_result, tool_name=underlying.__name__)

    return call


def _unwrap_mcp_result(mcp_result: dict[str, Any], *, tool_name: str) -> str:
    """Turn an MCP-shaped tool response into a plain string.

    MCP shape:
      success: {"content": [{"type": "text", "text": <json or message>}]}
      error:   {"content": [{"type": "text", "text": <message>}], "is_error": True}

    For errors we raise — the LLMSession's tool loop catches and reports
    `is_error=True` to the model. For success we return the text payload
    verbatim (already JSON-encoded when the tool wanted to return
    structured data, which the model parses on the other side).
    """
    content = mcp_result.get("content") or []
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(t for t in text_parts if t)

    if mcp_result.get("is_error"):
        raise RuntimeError(text or f"Tool {tool_name!r} returned an error")

    return text
