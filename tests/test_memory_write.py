"""Tests for the `remember` memory-write tool (src/tools/memory_write.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import memory_write as mw
from src.tools import build_provider_tools


def _cfg(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(workspace_dir=tmp_path)


class TestRememberHandler:
    @pytest.mark.asyncio
    async def test_long_term_appends_to_memory_md(self, tmp_path):
        res = await mw._handle_remember({"fact": "Joe ships on Fridays"}, config=_cfg(tmp_path))
        assert not res.get("is_error")
        assert (tmp_path / "MEMORY.md").read_text().strip() == "Joe ships on Fridays"

    @pytest.mark.asyncio
    async def test_long_term_is_default_scope(self, tmp_path):
        await mw._handle_remember({"fact": "no scope given"}, config=_cfg(tmp_path))
        assert "no scope given" in (tmp_path / "MEMORY.md").read_text()

    @pytest.mark.asyncio
    async def test_append_not_overwrite(self, tmp_path):
        cfg = _cfg(tmp_path)
        await mw._handle_remember({"fact": "first"}, config=cfg)
        await mw._handle_remember({"fact": "second"}, config=cfg)
        body = (tmp_path / "MEMORY.md").read_text()
        assert "first" in body and "second" in body

    @pytest.mark.asyncio
    async def test_daily_scope_creates_note(self, tmp_path):
        res = await mw._handle_remember(
            {"fact": "watched the turntable", "scope": "daily"}, config=_cfg(tmp_path)
        )
        assert not res.get("is_error")
        notes = list((tmp_path / "memory").glob("*.md"))
        assert len(notes) == 1
        assert "watched the turntable" in notes[0].read_text()

    @pytest.mark.asyncio
    async def test_empty_fact_errors(self, tmp_path):
        res = await mw._handle_remember({"fact": "   "}, config=_cfg(tmp_path))
        assert res.get("is_error") is True

    @pytest.mark.asyncio
    async def test_bad_scope_errors(self, tmp_path):
        res = await mw._handle_remember(
            {"fact": "x", "scope": "wherever"}, config=_cfg(tmp_path)
        )
        assert res.get("is_error") is True

    @pytest.mark.asyncio
    async def test_too_long_errors(self, tmp_path):
        res = await mw._handle_remember(
            {"fact": "x" * (mw.MAX_FACT_CHARS + 1)}, config=_cfg(tmp_path)
        )
        assert res.get("is_error") is True
        # nothing written
        assert not (tmp_path / "MEMORY.md").exists()

    @pytest.mark.asyncio
    async def test_no_workspace_errors(self):
        res = await mw._handle_remember({"fact": "x"}, config=SimpleNamespace(workspace_dir=None))
        assert res.get("is_error") is True


class TestRegistration:
    def test_registered_by_default(self, tmp_path):
        names = [t.definition.name for t in build_provider_tools(_cfg(tmp_path))]
        assert "remember" in names

    def test_disabled_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_WRITE_DISABLED", "true")
        names = [t.definition.name for t in build_provider_tools(_cfg(tmp_path))]
        assert "remember" not in names
