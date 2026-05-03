"""Agent SDK tool registry.

This is the layer where we cross from "MicrowaveOS owns the data" to
"the LLM can take an action." Each module here exposes one or more
`SdkMcpTool` objects (created via `claude_agent_sdk.tool`) that the
LLM can invoke during a turn.

The registry pattern keeps tool wiring out of the orchestrator: the
orchestrator asks `build_mcp_servers(config)` and gets back a dict of
named MCP server configs ready to pass into `ClaudeAgentOptions`,
plus an `allowed_tools` list of fully-qualified tool names.

Naming convention. The Agent SDK exposes in-process tools as
`mcp__<server-name>__<tool-name>`. We use a single server named
`microwave` so all tools land at `mcp__microwave__<tool-name>`. One
server is enough until we want per-domain sandboxing (e.g. revoking
just file-system tools).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


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

    if not tools:
        return ToolBundle(mcp_servers={}, allowed_tools=[], catalog_text="")

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
