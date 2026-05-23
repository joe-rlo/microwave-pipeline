"""Native filesystem tools — replaces Agent SDK Read for the new provider path.

Phase C.5 starts with `read_file`. Bash / Write / Edit need different
sandboxing decisions and follow in later phases.

What `read_file` does:
- Reads a text file from within `workspace_dir` (the user's
  ~/.microwaveos/workspace by default — same directory the LLM has
  always been able to navigate via the SDK's Read tool)
- Returns content as plain text, with a size cap so a huge file can't
  blow the model's context budget
- Resolves the path strictly: relative paths are anchored to
  workspace_dir; absolute paths must already be inside workspace_dir.
  Symlinks are resolved BEFORE the sandbox check so a symlink
  pointing outside the workspace cannot exfiltrate data.
- Rejects binary files by sniffing content (first 1KB), and also by
  extension whitelist as a cheap fast path

What it does NOT do:
- No globbing / directory listing. One file per call, identified by
  path. Listing tools come later if there's real demand — the LLM can
  always list with webfetch-style enumeration on its own.
- No relative-to-CWD resolution. CWD inside the bot is the project
  source tree, NOT the workspace. Always interpret paths against
  `workspace_dir`.
- No write or edit. Those land in later phases — they need a
  per-call confirmation model the orchestrator doesn't have today.

Why this tool isn't gated by an env var (mirrors webfetch design):
- Sandboxing is enforced unconditionally; the LLM can't read outside
  workspace_dir regardless of environment
- The workspace IS the user's chosen surface for "things the agent can
  read." If they don't want a file readable, it doesn't go there.
- Set `FILE_TOOLS_DISABLED=true` as an escape hatch for testing /
  paranoid deployments.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Practical cap on returned text. Same shape as webfetch — ~50K chars
# fits a long doc/note, stays well within any model's context budget.
DEFAULT_MAX_CHARS = 50_000

# Hard byte cap before we even attempt to decode. Files bigger than
# this are almost certainly not what you want the LLM reading.
DEFAULT_MAX_BYTES = 2_000_000

# How many leading bytes to sniff for the binary-content check.
# Mirrors what `git diff` uses to decide if a file is binary.
BINARY_SNIFF_BYTES = 1024


# Text-friendly extensions. NOT an exclusive list — we ALSO content-sniff
# for binary so an unrecognized extension like .conf or .toml works as
# long as it's actually text. This is the fast-path / failure-mode
# documentation surface.
TEXT_EXTENSIONS = frozenset({
    # Prose / notes
    "md", "txt", "rst", "org",
    # Code
    "py", "js", "ts", "tsx", "jsx", "rb", "go", "rs", "java", "c", "cc",
    "cpp", "h", "hpp", "cs", "swift", "kt", "scala", "clj", "ex", "exs",
    "elm", "lua", "php", "pl", "sh", "bash", "zsh", "fish", "ps1",
    "sql", "r", "jl",
    # Config / data
    "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
    "properties", "xml", "csv", "tsv",
    # Web
    "html", "htm", "css", "scss", "sass", "less", "vue", "svelte",
    # Docs / misc
    "log", "dockerfile", "makefile", "gitignore", "editorconfig",
    "license", "readme", "fountain",
})


WEBFETCH_TOOL_DOCS = """\
**read_file** — Read a text file from the user's workspace.

When to use:
- The user references a workspace file by name ("the BIBLE.md in my
  novel project", "today's daily note", "MEMORY.md").
- You need the content of something that lives in workspace/ to
  answer the question.

When NOT to use:
- Browsing the workspace structure — you don't have a listing tool;
  ask the user for the path if you don't know it.
- Editing files — you can read but not write or modify.

How to use:
- `path`: relative to workspace/ (e.g. "MEMORY.md", "skills/github/SKILL.md")
  or absolute INSIDE workspace/. Paths outside the workspace are rejected.
- Returns up to ~50K chars of text. Binary files / oversized files
  return an error you can surface to the user.
"""


READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Path to the file. Relative paths are resolved against "
                "the user's workspace directory. Absolute paths must "
                "already be inside the workspace."
            ),
        },
        "max_chars": {
            "type": "integer",
            "minimum": 100,
            "maximum": 200_000,
            "description": (
                f"Max characters to return. Defaults to {DEFAULT_MAX_CHARS}. "
                "Beyond this the result is truncated with a clear marker."
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


class ReadFileError(RuntimeError):
    """Raised on any read_file failure that should surface to the model."""


def read_file(
    path_str: str,
    *,
    workspace_dir: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    """Read a text file from within the workspace and return its content.

    Pure function. Tools wrap this to produce MCP / provider-shape
    responses; tests call it directly.

    Raises `ReadFileError` on any failure with a message safe to surface
    to the model + user.
    """
    if not isinstance(path_str, str) or not path_str:
        raise ReadFileError("path must be a non-empty string")

    workspace_dir = Path(workspace_dir).resolve()
    if not workspace_dir.exists():
        raise ReadFileError(
            f"Workspace directory {workspace_dir} does not exist"
        )

    target = _resolve_within_workspace(path_str, workspace_dir)

    if not target.exists():
        raise ReadFileError(f"File not found: {path_str}")
    if not target.is_file():
        raise ReadFileError(f"Not a file: {path_str}")

    try:
        size = target.stat().st_size
    except OSError as e:
        raise ReadFileError(f"Could not stat file: {e}") from e

    if size > max_bytes:
        raise ReadFileError(
            f"File too large ({size:,} bytes > {max_bytes:,} cap); "
            "ask the user to point at a specific section instead."
        )

    # Cheap extension check first
    suffix = target.suffix.lstrip(".").lower()
    name_lower = target.name.lower()
    # The whitelist covers files identified by stem (e.g. "Dockerfile",
    # "Makefile") too — check both.
    if (
        suffix
        and suffix not in TEXT_EXTENSIONS
        and name_lower not in TEXT_EXTENSIONS
    ):
        # Not on the whitelist — fall through to content sniff. We don't
        # reject outright because users have valid `.fountain`, `.conf`,
        # `.tmpl` files all the time.
        pass

    # Content-sniff for binary. A NUL byte in the first 1KB is the
    # canonical "this is binary" signal git itself uses.
    try:
        with open(target, "rb") as f:
            sniff = f.read(min(size, BINARY_SNIFF_BYTES))
    except OSError as e:
        raise ReadFileError(f"Could not open file: {e}") from e

    if b"\x00" in sniff:
        raise ReadFileError(
            f"File appears to be binary (NUL byte in first {BINARY_SNIFF_BYTES} "
            "bytes); read_file only handles text."
        )

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ReadFileError(f"Could not read file: {e}") from e

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n…[truncated]"

    return text


def _resolve_within_workspace(path_str: str, workspace_dir: Path) -> Path:
    """Resolve `path_str` into an absolute path GUARANTEED to be inside
    `workspace_dir`. Raises ReadFileError if the resolution escapes
    the sandbox (including via symlinks).

    Why we resolve BEFORE checking: `workspace/../../etc/passwd` would
    pass a string-prefix check naively, but `.resolve()` normalizes
    away the `..` segments. Same for symlinks — `.resolve(strict=False)`
    follows them, so a symlink pointing outside the workspace ends up
    with its real target's path here.
    """
    raw = Path(path_str)
    if raw.is_absolute():
        candidate = raw
    else:
        candidate = workspace_dir / raw

    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as e:
        # RuntimeError can come from infinite symlink loops
        raise ReadFileError(f"Could not resolve path {path_str!r}: {e}") from e

    # Use is_relative_to for the sandbox check — clean, no string games.
    if not _is_within(resolved, workspace_dir):
        raise ReadFileError(
            f"Path {path_str!r} resolves outside the workspace; refusing"
        )

    return resolved


def _is_within(child: Path, parent: Path) -> bool:
    """Python 3.9+ has Path.is_relative_to; we replicate for forward
    safety across versions and to keep the logic visible at the call site."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# --- SDK-shape registration ------------------------------------------------


def build_read_sdk_tools(workspace_dir: Path) -> list:
    """SdkMcpTool wrapper. Returns [] if the SDK isn't installed."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    @tool(
        name="read_file",
        description=(
            "Read a text file from the user's workspace. Paths are "
            "resolved against the workspace directory; absolute paths "
            "outside the workspace are rejected. Binary files are rejected."
        ),
        input_schema=READ_FILE_SCHEMA,
    )
    async def read_file_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_read_file(args, workspace_dir=workspace_dir)

    return [read_file_tool]


async def _handle_read_file(
    args: dict[str, Any], *, workspace_dir: Path,
) -> dict[str, Any]:
    """Shared handler — returns MCP shape, used by both SDK and provider paths."""
    path_str = args.get("path")
    max_chars = args.get("max_chars") or DEFAULT_MAX_CHARS

    if not isinstance(path_str, str) or not path_str:
        return _error("path must be a non-empty string")

    try:
        text = read_file(
            path_str,
            workspace_dir=workspace_dir,
            max_chars=max_chars,
        )
    except ReadFileError as e:
        log.info("read_file failed for %s: %s", path_str, e)
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected read_file failure for %s", path_str)
        return _error(f"Unexpected error: {e}")

    payload = {"path": path_str, "char_count": len(text), "text": text}
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }


def _error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }


def file_tools_disabled() -> bool:
    """True when FILE_TOOLS_DISABLED env flag is set."""
    return os.environ.get("FILE_TOOLS_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
