"""/why command tests — path abbreviation, output shape, verbose flag.

The handler doesn't make any I/O calls — it formats a cached
SearchResult into a string. Tests cover:
- Empty cache returns an informative message
- Non-empty cache renders top-5 fragments
- Verbose flag (-v / scores / verbose) reveals scores
- Path abbreviation trims absolute paths to workspace-relative form
- Non-filesystem sources (session:abc:summary) pass through unchanged
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.pipeline.why import (
    _abbreviate_source,
    _snippet,
    handle_why_command,
)
from src.session.models import MemoryFragment, SearchResult


def _frag(content="hello world", source="/x/workspace/MEMORY.md", score=0.8):
    return MemoryFragment(
        id=1, content=content, source=source,
        timestamp=datetime(2026, 5, 13), score=score,
    )


def _orch_with_search(fragments, workspace=Path("/x/workspace")):
    """Build a minimal orchestrator-shaped object with a cached search."""
    orch = SimpleNamespace()
    orch.config = SimpleNamespace(workspace_dir=workspace)
    orch._last_search_result = SearchResult(
        fragments=list(fragments), strategy_used="hybrid", search_time_ms=42,
    )
    return orch


class TestSnippet:
    def test_short_content_unchanged(self):
        assert _snippet("Joe is a builder") == "Joe is a builder"

    def test_collapses_newlines(self):
        assert _snippet("line one\nline two\nline three") == "line one line two line three"

    def test_truncates_with_ellipsis(self):
        long = "x" * 200
        out = _snippet(long)
        assert len(out) <= 100
        assert out.endswith("…")

    def test_handles_empty(self):
        assert _snippet("") == ""


class TestAbbreviateSource:
    def test_workspace_relative(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        memory = ws / "MEMORY.md"
        memory.write_text("x")
        out = _abbreviate_source(str(memory), ws)
        assert out == "workspace/MEMORY.md"

    def test_nested_workspace_path(self, tmp_path):
        ws = tmp_path / "workspace"
        (ws / "projects" / "blog").mkdir(parents=True)
        target = ws / "projects" / "blog" / "BIBLE.md"
        target.write_text("x")
        out = _abbreviate_source(str(target), ws)
        assert out == "workspace/projects/blog/BIBLE.md"

    def test_non_filesystem_source_passes_through(self, tmp_path):
        """Session summary IDs like 'session:abc123:summary' are not
        paths; rendering them unchanged is more useful than treating
        them as broken paths."""
        out = _abbreviate_source("session:abc123:summary", tmp_path)
        assert out == "session:abc123:summary"

    def test_outside_workspace_shows_last_two_components(self, tmp_path):
        """Paths outside the workspace (system files, attachments)
        should still be recognizable — show last 2 components, not
        the full absolute path."""
        out = _abbreviate_source(
            "/Users/joe/some/elsewhere/file.txt", tmp_path / "workspace",
        )
        assert out == ".../elsewhere/file.txt"

    def test_empty_source(self, tmp_path):
        assert _abbreviate_source("", tmp_path) == "(unknown source)"


class TestHandleWhyCommand:
    @pytest.mark.asyncio
    async def test_not_a_why_command_returns_none(self):
        orch = _orch_with_search([])
        assert await handle_why_command("hello there", orch) is None
        assert await handle_why_command("/skill foo", orch) is None
        assert await handle_why_command("", orch) is None

    @pytest.mark.asyncio
    async def test_no_cache_yet(self):
        """Fresh orchestrator hasn't processed a turn yet."""
        orch = SimpleNamespace(config=SimpleNamespace(workspace_dir=Path("/x")))
        orch._last_search_result = None
        out = await handle_why_command("/why", orch)
        assert "No retrieval cached" in out

    @pytest.mark.asyncio
    async def test_empty_fragments(self):
        """Search ran but returned nothing — distinct from 'no cache'."""
        orch = _orch_with_search([])
        out = await handle_why_command("/why", orch)
        assert "empty" in out.lower()
        assert "hybrid" in out  # strategy surfaced for tuning

    @pytest.mark.asyncio
    async def test_renders_fragments(self):
        orch = _orch_with_search([
            _frag(content="Joe's primary project is MicrowaveOS",
                  source="/x/workspace/MEMORY.md", score=0.82),
            _frag(content="Sarah's character voice is sharp",
                  source="/x/workspace/projects/novel/BIBLE.md", score=0.74),
        ], workspace=Path("/x/workspace"))
        out = await handle_why_command("/why", orch)
        assert "why: last turn retrieved" in out
        assert "workspace/MEMORY.md" in out
        assert "Joe's primary project" in out
        assert "Sarah's character voice" in out

    @pytest.mark.asyncio
    async def test_default_hides_scores(self):
        """Score values without prior calibration on what they mean
        are noise — default output is snippets only."""
        orch = _orch_with_search([_frag(score=0.82)])
        out = await handle_why_command("/why", orch)
        assert "[0.82]" not in out
        assert "0.82" not in out  # not even loose

    @pytest.mark.asyncio
    async def test_verbose_dash_v_shows_scores(self):
        orch = _orch_with_search([_frag(score=0.82)])
        out = await handle_why_command("/why -v", orch)
        assert "[0.82]" in out

    @pytest.mark.asyncio
    async def test_verbose_scores_alias(self):
        """Signal/Telegram convention: `/why scores` is the verbose
        form because shell-style `-v` flags don't fit mobile UX."""
        orch = _orch_with_search([_frag(score=0.95)])
        out = await handle_why_command("/why scores", orch)
        assert "[0.95]" in out

    @pytest.mark.asyncio
    async def test_verbose_alias_form(self):
        orch = _orch_with_search([_frag(score=0.61)])
        out = await handle_why_command("/why verbose", orch)
        assert "[0.61]" in out

    @pytest.mark.asyncio
    async def test_caps_at_five_fragments(self):
        """Even if search returned 20 fragments, /why shows top 5."""
        many = [_frag(content=f"frag {i}", source=f"/x/workspace/f{i}.md") for i in range(20)]
        orch = _orch_with_search(many, workspace=Path("/x/workspace"))
        out = await handle_why_command("/why", orch)
        assert "frag 0" in out
        assert "frag 4" in out
        assert "frag 5" not in out  # cut off
        assert "frag 19" not in out

    @pytest.mark.asyncio
    async def test_case_insensitive_command(self):
        """`/WHY` and `/Why` should work just like `/why`."""
        orch = _orch_with_search([_frag()])
        assert await handle_why_command("/WHY", orch) is not None
        assert await handle_why_command("/Why", orch) is not None
