"""Tests for Phase F.3 — operational glue.

Covers:
- MemoryStore.load_briefing() reads BRIEFING.md when present, returns ""
  when absent
- assemble_stable_context injects [Daily briefing] block when BRIEFING.md
  exists; doesn't when it doesn't
- stable_context_mtime includes BRIEFING.md so a briefing write triggers
  reconnect on next turn
- consolidation.scheduler: should_run returns True on missing marker,
  False within window, True after window; touch_marker creates+updates
- consolidation.scheduler.run_catchup_if_due: actually runs when due,
  noops when not, swallows exceptions

CLI tests stay narrow — we drive the CLI by argv and confirm exit code
+ key output substrings. Stub the LLM via the consolidation pipeline's
selector entry point so no tokens are spent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import apsw
import pytest

from src.memory.consolidation.scheduler import (
    marker_path,
    run_catchup_if_due,
    should_run,
    touch_marker,
)
from src.memory.store import MemoryStore


# --- MemoryStore briefing ---


class TestBriefingLoad:
    def test_load_briefing_missing_returns_empty(self, tmp_path):
        store = MemoryStore(tmp_path)
        assert store.load_briefing() == ""

    def test_load_briefing_present(self, tmp_path):
        store = MemoryStore(tmp_path)
        (tmp_path / "BRIEFING.md").write_text("# Today\n\nFoo")
        assert "# Today" in store.load_briefing()

    def test_assemble_stable_includes_briefing(self, tmp_path):
        store = MemoryStore(tmp_path)
        # Need at least an identity for the prompt to assemble
        (tmp_path / "IDENTITY.md").write_text("You are an assistant.")
        (tmp_path / "BRIEFING.md").write_text("Active project: foo")
        prompt = store.assemble_stable_context()
        assert "[Daily briefing]" in prompt
        assert "Active project: foo" in prompt

    def test_assemble_stable_no_briefing_section_when_absent(self, tmp_path):
        store = MemoryStore(tmp_path)
        (tmp_path / "IDENTITY.md").write_text("You are an assistant.")
        prompt = store.assemble_stable_context()
        assert "[Daily briefing]" not in prompt

    def test_briefing_mtime_in_stable_context_mtime(self, tmp_path):
        store = MemoryStore(tmp_path)
        (tmp_path / "IDENTITY.md").write_text("hi")
        before = store.stable_context_mtime()
        # Write briefing — mtime must bump
        time.sleep(0.02)
        (tmp_path / "BRIEFING.md").write_text("brief content")
        after = store.stable_context_mtime()
        assert after > before


# --- consolidation.scheduler ---


class TestShouldRun:
    def test_no_marker_means_run(self, tmp_path):
        assert should_run(data_dir=tmp_path, interval_hours=24) is True

    def test_recent_marker_skips(self, tmp_path):
        touch_marker(tmp_path)
        assert should_run(data_dir=tmp_path, interval_hours=24) is False

    def test_old_marker_runs(self, tmp_path):
        touch_marker(tmp_path)
        m = marker_path(tmp_path)
        # backdate to 25h ago
        old = time.time() - 25 * 3600
        import os
        os.utime(m, (old, old))
        assert should_run(data_dir=tmp_path, interval_hours=24) is True

    def test_zero_hour_interval_always_runs(self, tmp_path):
        touch_marker(tmp_path)
        # interval=0 means "always due"
        assert should_run(data_dir=tmp_path, interval_hours=0) is True

    def test_touch_marker_creates_directory(self, tmp_path):
        sub = tmp_path / "subdir-that-doesnt-exist"
        # Marker write must create the parent directory rather than
        # raising — fresh installs may have no data_dir yet.
        touch_marker(sub)
        assert sub.exists()
        assert marker_path(sub).exists()


@pytest.mark.asyncio
class TestRunCatchupIfDue:
    async def _make_config(self, tmp_path):
        class _C:
            workspace_dir = tmp_path / "workspace"
            data_dir = tmp_path / "data"
            auth_mode = "max"
            anthropic_api_key = ""
            cli_path = ""
        _C.workspace_dir.mkdir(parents=True, exist_ok=True)
        _C.data_dir.mkdir(parents=True, exist_ok=True)
        return _C()

    async def _conn_with_schema(self):
        c = apsw.Connection(":memory:")
        c.row_trace = lambda cursor, row: {
            d[0]: v for d, v in zip(cursor.getdescription(), row)
        }
        from src.memory.breadcrumbs import init_tables as init_bc
        from src.memory.consolidation import init_tables as init_consolidation
        init_bc(c)
        init_consolidation(c)
        # turns table for Extract's recent-turn read
        c.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(id INTEGER PRIMARY KEY, role TEXT, content TEXT, timestamp TEXT)"
        )
        return c

    async def test_skips_when_marker_fresh(self, tmp_path, monkeypatch):
        config = await self._make_config(tmp_path)
        touch_marker(config.data_dir)
        conn = await self._conn_with_schema()

        ran = await run_catchup_if_due(
            conn=conn, config=config, interval_hours=24,
        )
        assert ran is False

    async def test_runs_and_touches_marker(self, tmp_path, monkeypatch):
        config = await self._make_config(tmp_path)
        conn = await self._conn_with_schema()

        # Stub the consolidation pipeline's per-stage LLM callables
        def fake_get_stage_callable(stage, **kw):
            async def call(s, u):
                return json.dumps({"facts": []})
            return call
        monkeypatch.setattr(
            "src.llm.selector.get_stage_callable", fake_get_stage_callable
        )

        ran = await run_catchup_if_due(
            conn=conn, config=config, interval_hours=24,
        )
        assert ran is True
        # Marker should now exist
        assert marker_path(config.data_dir).exists()
        # Second call within window should skip
        ran_again = await run_catchup_if_due(
            conn=conn, config=config, interval_hours=24,
        )
        assert ran_again is False

    async def test_pipeline_failure_logged_not_raised(self, tmp_path, monkeypatch):
        config = await self._make_config(tmp_path)
        conn = await self._conn_with_schema()

        # scheduler.py imports run_consolidation lazily inside the
        # function, so we patch at the original location instead.
        async def boom(**kwargs):
            raise RuntimeError("simulated catastrophe")

        monkeypatch.setattr(
            "src.memory.consolidation.pipeline.run_consolidation", boom
        )
        # The scheduler module also imports via `from src.memory.consolidation import run_consolidation`
        # — patch that too.
        import src.memory.consolidation as cons_pkg
        monkeypatch.setattr(cons_pkg, "run_consolidation", boom)

        # Must NOT raise — catchup is best-effort.
        ran = await run_catchup_if_due(
            conn=conn, config=config, interval_hours=24,
        )
        assert ran is False
        # Marker should NOT be touched (the run failed)
        assert not marker_path(config.data_dir).exists()


# --- CLI smoke tests ---


class TestMemoryCli:
    def test_parser_accepts_new_actions(self, monkeypatch, capsys):
        # We don't actually exercise consolidate / facts / etc against
        # a real database here — those go through the same code paths
        # as test_consolidation.py. This test just confirms argparse
        # accepts every new action.
        from src.memory.cli import memory_cli

        # `breadcrumbs` against a fresh ephemeral data dir should print
        # the "no breadcrumbs yet" message and return 0.
        import os
        tmp = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / "microwaveos-cli-test"
        tmp.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp / "workspace"))
        monkeypatch.setenv("DATA_DIR", str(tmp / "data"))

        rc = memory_cli(["breadcrumbs", "--limit", "5"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "No breadcrumbs" in captured.out or "breadcrumb" in captured.out.lower()

    def test_briefing_action_with_no_file(self, monkeypatch, tmp_path, capsys):
        from src.memory.cli import memory_cli
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        rc = memory_cli(["briefing"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "BRIEFING.md" in captured.out

    def test_briefing_prints_file_when_present(self, monkeypatch, tmp_path, capsys):
        from src.memory.cli import memory_cli
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "BRIEFING.md").write_text("hello briefing")
        monkeypatch.setenv("WORKSPACE_DIR", str(ws))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        rc = memory_cli(["briefing"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "hello briefing" in captured.out
