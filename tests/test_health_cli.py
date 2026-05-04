"""Health CLI smoke tests — exercise each subcommand against a tmp
workspace + mocked retrieval.

We don't test the dispatcher's argparse plumbing exhaustively; we do
test that each subcommand's effect is correct (status reads config,
install-skill copies the seed dir, audit list reads from the audit DB).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.health.cli import health_cli


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """Point WORKSPACE_DIR + DATA_DIR at temp dirs and return a
    helper-bound config so each test starts clean."""
    workspace = tmp_path / "ws"
    data = tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("HEALTH_MODULE_ENABLED", "true")
    return workspace, data


class TestStatusCommand:
    def test_status_runs_clean(self, isolated_workspace, capsys):
        rc = health_cli(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Module enabled: True" in out
        assert "BAA provider:" in out
        assert "Retrieval sources:" in out
        # Phase 3+ sources annotated as no-impl
        assert "(no impl yet — Phase 3+)" in out

    def test_status_shows_skill_missing(self, isolated_workspace, capsys):
        rc = health_cli(["status"])
        out = capsys.readouterr().out
        assert "health-qa skill NOT installed" in out
        assert "install-skill" in out


class TestInstallSkill:
    def test_install_when_missing(self, isolated_workspace, capsys):
        workspace, _ = isolated_workspace
        rc = health_cli(["install-skill"])
        assert rc == 0
        target = workspace / "skills" / "health-qa" / "SKILL.md"
        assert target.is_file()
        # Sanity: the spec content actually copied
        assert "Health Q&A" in target.read_text()
        assert "Hard rules" in target.read_text()

    def test_install_idempotent(self, isolated_workspace, capsys):
        workspace, _ = isolated_workspace
        # First install
        health_cli(["install-skill"])
        target = workspace / "skills" / "health-qa" / "SKILL.md"
        # Modify the installed copy to verify second call doesn't clobber
        target.write_text("# my edits", encoding="utf-8")
        rc = health_cli(["install-skill"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "already installed" in out
        # User edits preserved
        assert target.read_text() == "# my edits"


class TestAuditList:
    def test_empty_audit(self, isolated_workspace, capsys):
        rc = health_cli(["audit", "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No audit rows yet" in out

    def test_audit_list_after_writes(self, isolated_workspace, capsys):
        from src.health.audit import HealthAuditRow, HealthAuditWriter

        _, data_dir = isolated_workspace
        # Use the same db_path the CLI will use (db_path is data_dir/memory.db)
        db_path = data_dir / "memory.db"
        w = HealthAuditWriter(db_path)
        w.connect()
        w.write(HealthAuditRow(
            route="general",
            triage_phi_class="general",
            triage_health_topic="diabetes",
            sources_returned=[{"name": "pubmed", "count": 2}],
            llm_provider="anthropic",
            llm_model="sonnet",
            latency_ms=850,
        ))
        w.close()

        rc = health_cli(["audit", "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "route=general" in out
        assert "diabetes" in out
        assert "pubmed:2" in out


class TestRetrieveCommand:
    """`retrieve` is a sync entrypoint that wraps an async coroutine in
    asyncio.run. Tests stay sync — pytest-asyncio's loop would conflict
    with the CLI's own asyncio.run."""

    def test_retrieve_no_results_returns_zero(self, isolated_workspace, capsys):
        from src.health.retrieval.orchestrator import RetrievalOrchestrator

        async def empty_search(*a, **kw):
            return []

        with patch.object(RetrievalOrchestrator, "search", empty_search):
            rc = health_cli(["retrieve", "metformin"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No results" in out

    def test_retrieve_prints_results(self, isolated_workspace, capsys):
        from datetime import date
        from src.health.retrieval.base import Evidence
        from src.health.retrieval.orchestrator import RetrievalOrchestrator

        async def fake_search(self, query, topic=None, max_results=8):
            return [Evidence(
                source="pubmed",
                title="Effect of metformin",
                snippet="A randomized trial of 5,000 patients...",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                published=date(2024, 3, 15),
            )]

        with patch.object(RetrievalOrchestrator, "search", fake_search):
            rc = health_cli(["retrieve", "metformin", "--topic", "diabetes"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[1] pubmed" in out
        assert "Effect of metformin" in out
        assert "https://pubmed.ncbi.nlm.nih.gov/12345/" in out
        assert "2024-03-15" in out
