"""Health disclaimer loader tests."""

from __future__ import annotations

from pathlib import Path

from src.health.disclaimers import (
    DEFAULT_HEALTH_DISCLAIMER,
    load_health_channel_rules,
)


class TestLoadHealthChannelRules:
    def test_default_when_no_override(self, tmp_path: Path):
        # Empty workspace — default returned
        result = load_health_channel_rules(tmp_path)
        assert result == DEFAULT_HEALTH_DISCLAIMER

    def test_user_override(self, tmp_path: Path):
        channels_dir = tmp_path / "channels"
        channels_dir.mkdir()
        custom = "# My custom health disclaimer\n\n- Always remind about ER."
        (channels_dir / "health.md").write_text(custom, encoding="utf-8")

        result = load_health_channel_rules(tmp_path)
        assert result == custom

    def test_empty_override_falls_back_to_default(self, tmp_path: Path):
        """A blank file shouldn't override with nothing — that'd silently
        suppress the disclaimer footer, a real safety regression."""
        channels_dir = tmp_path / "channels"
        channels_dir.mkdir()
        (channels_dir / "health.md").write_text("   \n\n  ", encoding="utf-8")

        result = load_health_channel_rules(tmp_path)
        assert result == DEFAULT_HEALTH_DISCLAIMER

    def test_default_includes_emergency_framing(self):
        """Smoke check on the default: an emergency-services pointer
        is non-negotiable for the footer."""
        assert "emergency services" in DEFAULT_HEALTH_DISCLAIMER.lower()
        assert "not medical advice" in DEFAULT_HEALTH_DISCLAIMER.lower()
