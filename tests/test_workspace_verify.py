"""Boot-time workspace invariant tests.

The verify_workspace hook runs early in Orchestrator.start() so any
drift between WORKSPACE_DIR and the actual disk layout is loud at
startup rather than confusing mid-conversation. Pin the contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.orchestrator import _verify_workspace


def _config(tmp_path):
    """Minimal Config-shaped object exposing the three attrs
    _verify_workspace reads."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = MagicMock()
    cfg.workspace_dir = workspace
    cfg.identity_path = workspace / "IDENTITY.md"
    cfg.memory_path = workspace / "MEMORY.md"
    return cfg, workspace


class TestVerifyWorkspace:
    def test_passes_when_identity_and_memory_present(self, tmp_path, caplog):
        cfg, ws = _config(tmp_path)
        (ws / "IDENTITY.md").write_text("# You are Microwave.")
        (ws / "MEMORY.md").write_text("- Dog: Biscuit")
        # Should not raise
        with caplog.at_level("INFO"):
            _verify_workspace(cfg)
        # Banner emitted with resolved paths — useful for diagnosing
        # cwd drift in production logs
        assert any("workspace = " in r.message for r in caplog.records)
        assert any("cwd       = " in r.message for r in caplog.records)

    def test_missing_identity_is_fatal(self, tmp_path):
        """IDENTITY.md anchors the system prompt's voice. Running
        without it produces a generic AI-assistant tone that surprises
        users — fail loud at boot rather than ship a broken persona."""
        cfg, _ = _config(tmp_path)
        # No IDENTITY.md created
        with pytest.raises(RuntimeError, match="IDENTITY.md"):
            _verify_workspace(cfg)

    def test_missing_memory_warns_does_not_crash(self, tmp_path, caplog):
        """MEMORY.md is recommended but optional — a clean install
        without it should still boot, just with a warning."""
        cfg, ws = _config(tmp_path)
        (ws / "IDENTITY.md").write_text("# You are Microwave.")
        # No MEMORY.md
        with caplog.at_level("WARNING"):
            _verify_workspace(cfg)
        assert any("MEMORY.md not present" in r.message for r in caplog.records)

    def test_error_message_points_at_remedy(self, tmp_path):
        """When we crash, the message should tell the user how to fix
        it — not just 'file missing'."""
        cfg, _ = _config(tmp_path)
        with pytest.raises(RuntimeError) as exc:
            _verify_workspace(cfg)
        msg = str(exc.value)
        assert "Customization" in msg or "voice/persona" in msg
