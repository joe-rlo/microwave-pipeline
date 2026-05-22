"""Agent SDK tool wrappers for GitHub.

Three tools, all read-only:

- `github_list_repos` — list repos visible to the authenticated user.
  Cheap, single-call. Good first step for "walk my repos".
- `github_repo_summary` — composite snapshot of one repo (metadata +
  README excerpt + recent commits + open PRs/issues + languages). One
  tool call → enough context for the LLM to write a useful summary.
- `github_recent_activity` — flattened events feed for the authenticated
  user. Answers "what have I been working on lately?" across all repos.

Why three tools instead of one big "walk_repos" tool: each call is small
and composable, so the LLM can decide its own breadth (list 20 repos,
deep-dive on the 3 most-pushed) instead of us guessing. Mirrors how a
human would explore the API.

The PAT lives in env (`GITHUB_TOKEN`), never in `gh auth` — so enabling
this tool doesn't touch the user's local `gh` state.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.integrations.github import GitHubClient, GitHubError

log = logging.getLogger(__name__)


# What the LLM reads in its system context.
GITHUB_TOOL_DOCS = """\
**github_list_repos** — List GitHub repos visible to the authenticated user (sorted by last push by default).

When to use:
- "What repos do I have?", "show me my projects", "walk my repos".
- As a first step before drilling into a specific repo with `github_repo_summary`.

How to use:
- `visibility`: "all" (default), "public", or "private".
- `sort`: "pushed" (default), "updated", "created", or "full_name".
- `limit`: defaults to 30 (max 100). Keep it small unless the user asked for everything.

**github_repo_summary** — Get a composite snapshot of one repo: metadata + README excerpt + recent commits + open PRs/issues + language breakdown.

When to use:
- After `github_list_repos`, when the user wants details on a specific repo.
- "What's [repo] about?", "summarize my [repo] project", "what's open on [repo]?".

How to use:
- `repo`: "owner/name" — e.g. "octocat/Hello-World". Required.
- Returns enough to write a paragraph-level summary without further calls. README is truncated to ~2000 chars.

**github_recent_activity** — Recent push/PR/issue/release events for the authenticated user across all repos.

When to use:
- "What have I been doing lately?", "what did I push this week?", "recap my GitHub activity".
- Cross-repo timeline — for single-repo recency, use `github_repo_summary` (it includes recent commits).

How to use:
- `limit`: defaults to 30, max 100. GitHub only retains ~90 days of events server-side.

Don't fabricate repo names, PR titles, or commit SHAs. If a tool errors, say so plainly.
"""


LIST_REPOS_SCHEMA = {
    "type": "object",
    "properties": {
        "visibility": {
            "type": "string",
            "enum": ["all", "public", "private"],
            "description": "Filter by visibility. Defaults to 'all'.",
        },
        "sort": {
            "type": "string",
            "enum": ["created", "updated", "pushed", "full_name"],
            "description": "Sort order. Defaults to 'pushed' (most recently pushed first).",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max repos to return. Defaults to 30.",
        },
    },
    "additionalProperties": False,
}


REPO_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": {
            "type": "string",
            "description": "Full repo name in 'owner/name' format (e.g. 'octocat/Hello-World').",
        },
    },
    "required": ["repo"],
    "additionalProperties": False,
}


RECENT_ACTIVITY_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max events to return. Defaults to 30.",
        },
    },
    "additionalProperties": False,
}


def build_github_tools(config) -> list:
    """Build the SdkMcpTool list for GitHub. Returns [] if SDK unavailable."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    token = getattr(config, "github_token", "")

    @tool(
        name="github_list_repos",
        description=(
            "List GitHub repos visible to the authenticated user, sorted by "
            "last push by default. Use as the first step when the user asks "
            "to walk through their projects."
        ),
        input_schema=LIST_REPOS_SCHEMA,
    )
    async def github_list_repos(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_list_repos(args, token)

    @tool(
        name="github_repo_summary",
        description=(
            "Get a composite snapshot of one GitHub repo — metadata, README "
            "excerpt, recent commits, open PRs/issues, language breakdown. "
            "Use after listing repos to drill into a specific one."
        ),
        input_schema=REPO_SUMMARY_SCHEMA,
    )
    async def github_repo_summary(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_repo_summary(args, token)

    @tool(
        name="github_recent_activity",
        description=(
            "Recent push/PR/issue/release events for the authenticated user "
            "across all repos. Use for 'what have I been working on lately' "
            "questions."
        ),
        input_schema=RECENT_ACTIVITY_SCHEMA,
    )
    async def github_recent_activity(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_recent_activity(args, token)

    return [github_list_repos, github_repo_summary, github_recent_activity]


# --- Handlers ---


async def _handle_list_repos(args: dict[str, Any], token: str) -> dict[str, Any]:
    visibility = args.get("visibility") or "all"
    sort = args.get("sort") or "pushed"
    limit = int(args.get("limit") or 30)

    client = GitHubClient(token=token)
    try:
        repos = await client.list_user_repos(
            visibility=visibility, sort=sort, limit=limit
        )
    except GitHubError as e:
        log.warning("github_list_repos failed: %s", e)
        return _error(f"GitHub API error: {e}")
    except Exception as e:
        log.exception("Unexpected github_list_repos failure")
        return _error(f"Unexpected error listing repos: {e}")

    payload = {
        "count": len(repos),
        "repos": [r.to_payload() for r in repos],
    }
    return _ok(payload)


async def _handle_repo_summary(args: dict[str, Any], token: str) -> dict[str, Any]:
    repo_name = args.get("repo")
    if not isinstance(repo_name, str) or "/" not in repo_name:
        return _error("`repo` must be in 'owner/name' format.")

    client = GitHubClient(token=token)
    try:
        summary = await client.get_repo_summary(repo_name)
    except GitHubError as e:
        log.warning("github_repo_summary(%s) failed: %s", repo_name, e)
        return _error(f"GitHub API error: {e}")
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected github_repo_summary failure")
        return _error(f"Unexpected error summarizing repo: {e}")

    return _ok(summary.to_payload())


async def _handle_recent_activity(args: dict[str, Any], token: str) -> dict[str, Any]:
    limit = int(args.get("limit") or 30)

    client = GitHubClient(token=token)
    try:
        user = await client.get_authenticated_user()
        username = user.get("login")
        if not isinstance(username, str) or not username:
            return _error("Couldn't determine authenticated GitHub username.")
        events = await client.list_recent_activity(username, limit=limit)
    except GitHubError as e:
        log.warning("github_recent_activity failed: %s", e)
        return _error(f"GitHub API error: {e}")
    except Exception as e:
        log.exception("Unexpected github_recent_activity failure")
        return _error(f"Unexpected error fetching activity: {e}")

    payload = {
        "username": username,
        "count": len(events),
        "events": [e.to_payload() for e in events],
    }
    return _ok(payload)


# --- MCP shape helpers ---


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }


def _error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }
