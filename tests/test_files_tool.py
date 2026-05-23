"""Tests for the native read_file tool (Phase C.5).

Coverage:
- read_file happy path on workspace files
- Sandbox: relative path resolves into workspace
- Sandbox escape: ../ rejected, absolute outside rejected
- Sandbox escape via symlink to outside-workspace rejected
- Binary detection (NUL byte in first 1KB)
- Size cap (byte-level pre-check + char-level truncation)
- Missing file / not-a-file rejected
- Empty path / non-string path rejected
- Handler returns MCP shape; success/error paths
- Provider bridge unwrap works (success + error)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.tools.files import (
    DEFAULT_MAX_CHARS,
    ReadFileError,
    _handle_read_file,
    _is_within,
    _resolve_within_workspace,
    file_tools_disabled,
    read_file,
)


# --- helpers ---


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tmp workspace dir with a couple of files."""
    (tmp_path / "MEMORY.md").write_text("# Memory\n- thing")
    (tmp_path / "IDENTITY.md").write_text("You are an assistant.")
    sub = tmp_path / "skills" / "github"
    sub.mkdir(parents=True)
    (sub / "SKILL.md").write_text("---\nname: github\n---\n# tool docs")
    return tmp_path


# --- _is_within ---


class TestIsWithin:
    def test_child_inside(self, tmp_path):
        assert _is_within(tmp_path / "foo" / "bar", tmp_path) is True

    def test_child_equal_to_parent(self, tmp_path):
        # A path is trivially relative to itself.
        assert _is_within(tmp_path, tmp_path) is True

    def test_sibling_outside(self, tmp_path):
        outside = tmp_path.parent / "other"
        assert _is_within(outside, tmp_path) is False


# --- _resolve_within_workspace ---


class TestResolve:
    def test_relative_path_anchored_to_workspace(self, workspace):
        out = _resolve_within_workspace("MEMORY.md", workspace)
        assert out == (workspace / "MEMORY.md").resolve()

    def test_absolute_path_inside_workspace_allowed(self, workspace):
        abs_path = str(workspace / "MEMORY.md")
        out = _resolve_within_workspace(abs_path, workspace)
        assert out == (workspace / "MEMORY.md").resolve()

    def test_dotdot_escape_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="outside the workspace"):
            _resolve_within_workspace("../escaped.md", workspace)

    def test_absolute_outside_rejected(self, workspace):
        # /etc/passwd is canonical "should never be readable"
        with pytest.raises(ReadFileError, match="outside the workspace"):
            _resolve_within_workspace("/etc/passwd", workspace)

    def test_symlink_pointing_outside_rejected(self, workspace, tmp_path):
        # Create a file outside workspace
        outside_dir = tmp_path.parent / "outside_workspace_for_test"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret"
        outside_file.write_text("don't read me")

        # Symlink from inside workspace → outside file
        link = workspace / "exfil_link"
        try:
            link.symlink_to(outside_file)
        except OSError:
            pytest.skip("symlinks not supported on this filesystem")

        # Even though `exfil_link` lives under workspace, .resolve()
        # follows the link to the real target — which is outside.
        with pytest.raises(ReadFileError, match="outside the workspace"):
            _resolve_within_workspace("exfil_link", workspace)

        # Cleanup
        outside_file.unlink()
        outside_dir.rmdir()

    def test_nested_path_ok(self, workspace):
        out = _resolve_within_workspace("skills/github/SKILL.md", workspace)
        assert out == (workspace / "skills" / "github" / "SKILL.md").resolve()


# --- read_file ---


class TestReadFile:
    def test_happy_path(self, workspace):
        text = read_file("MEMORY.md", workspace_dir=workspace)
        assert "Memory" in text
        assert "thing" in text

    def test_absolute_within_workspace(self, workspace):
        abs_path = str(workspace / "MEMORY.md")
        text = read_file(abs_path, workspace_dir=workspace)
        assert "Memory" in text

    def test_nested_file(self, workspace):
        text = read_file("skills/github/SKILL.md", workspace_dir=workspace)
        assert "github" in text

    def test_empty_path_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="non-empty"):
            read_file("", workspace_dir=workspace)

    def test_non_string_path_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="non-empty"):
            read_file(None, workspace_dir=workspace)  # type: ignore[arg-type]

    def test_missing_file_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="not found"):
            read_file("nope.md", workspace_dir=workspace)

    def test_directory_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="Not a file"):
            read_file("skills", workspace_dir=workspace)

    def test_workspace_dir_missing(self, tmp_path):
        with pytest.raises(ReadFileError, match="does not exist"):
            read_file("x.md", workspace_dir=tmp_path / "does-not-exist")

    def test_dotdot_escape_rejected(self, workspace):
        with pytest.raises(ReadFileError, match="outside"):
            read_file("../escape.md", workspace_dir=workspace)

    def test_size_cap_rejects_oversized(self, workspace):
        big = workspace / "big.txt"
        big.write_text("x" * 5_000_000)  # 5 MB
        with pytest.raises(ReadFileError, match="too large"):
            read_file("big.txt", workspace_dir=workspace, max_bytes=1_000_000)

    def test_max_chars_truncates(self, workspace):
        med = workspace / "med.md"
        med.write_text("a" * 10_000)
        text = read_file("med.md", workspace_dir=workspace, max_chars=500)
        assert text.endswith("[truncated]")
        assert len(text) < 1000

    def test_binary_file_rejected(self, workspace):
        bin_file = workspace / "blob.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03" * 100)
        with pytest.raises(ReadFileError, match="binary"):
            read_file("blob.bin", workspace_dir=workspace)

    def test_unknown_extension_text_file_accepted(self, workspace):
        # .conf isn't in TEXT_EXTENSIONS but content-sniff passes;
        # should be readable.
        cfg = workspace / "thing.conf"
        cfg.write_text("[section]\nkey=value\n")
        text = read_file("thing.conf", workspace_dir=workspace)
        assert "key=value" in text

    def test_utf8_with_invalid_bytes_replaced(self, workspace):
        # The fallback `errors='replace'` handles partially-corrupted
        # utf-8 (rare in practice, but real) without crashing.
        f = workspace / "bad.md"
        f.write_bytes(b"hello \xff\xfe world")
        text = read_file("bad.md", workspace_dir=workspace)
        assert "hello" in text
        assert "world" in text


# --- _handle_read_file (MCP shape) ---


@pytest.mark.asyncio
class TestHandlerMcpShape:
    async def test_success_returns_payload(self, workspace):
        result = await _handle_read_file(
            {"path": "MEMORY.md"}, workspace_dir=workspace,
        )
        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["path"] == "MEMORY.md"
        assert "Memory" in payload["text"]
        assert payload["char_count"] > 0

    async def test_missing_file_returns_is_error(self, workspace):
        result = await _handle_read_file(
            {"path": "nope.md"}, workspace_dir=workspace,
        )
        assert result.get("is_error") is True
        assert "not found" in result["content"][0]["text"]

    async def test_empty_path_returns_is_error(self, workspace):
        result = await _handle_read_file({}, workspace_dir=workspace)
        assert result.get("is_error") is True


# --- Provider bridge unwrap ---


@pytest.mark.asyncio
class TestProviderBridge:
    async def test_handler_returns_text_via_unwrap(self, workspace):
        from src.tools import _unwrap_mcp_result

        mcp = await _handle_read_file(
            {"path": "IDENTITY.md"}, workspace_dir=workspace,
        )
        text = _unwrap_mcp_result(mcp, tool_name="read_file")
        payload = json.loads(text)
        assert payload["path"] == "IDENTITY.md"
        assert "assistant" in payload["text"].lower()

    async def test_handler_error_raises_via_unwrap(self, workspace):
        from src.tools import _unwrap_mcp_result

        mcp = await _handle_read_file(
            {"path": "../escape"}, workspace_dir=workspace,
        )
        with pytest.raises(RuntimeError, match="outside"):
            _unwrap_mcp_result(mcp, tool_name="read_file")


# --- Env flag ---


class TestEnvFlag:
    def test_disabled_when_set(self, monkeypatch):
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "true")
        assert file_tools_disabled() is True

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("FILE_TOOLS_DISABLED", raising=False)
        assert file_tools_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("FILE_TOOLS_DISABLED", val)
        assert file_tools_disabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("FILE_TOOLS_DISABLED", val)
        assert file_tools_disabled() is False
