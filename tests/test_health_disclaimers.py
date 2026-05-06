"""Health disclaimer loader tests."""

from __future__ import annotations

from pathlib import Path

from src.health.disclaimers import (
    DEFAULT_HEALTH_DISCLAIMER,
    EMPTY_RETRIEVAL_RELAXATION,
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


class TestEmptyRetrievalRelaxation:
    """The relaxation block is the carve-out that lets the bot answer
    benign general questions when retrieval came back empty. Pin the
    safety floors that still apply, since the whole point is they don't
    relax with the evidence rule."""

    def test_explicitly_overrides_evidence_rule(self):
        """The skill body says 'use only provided evidence'; this block
        must clearly mark itself as the override or the model gets
        contradictory instructions."""
        text = EMPTY_RETRIEVAL_RELAXATION.lower()
        assert "carve-out" in text or "override" in text or "relaxed" in text
        assert "this turn" in text

    def test_safety_floors_preserved(self):
        """Each floor that the spec marked non-negotiable for ANY
        health-routed answer must still be named here."""
        text = EMPTY_RETRIEVAL_RELAXATION.lower()
        # Dose / start-stop / diagnosis floors
        assert "dose" in text
        assert "start" in text and "stop" in text
        # Emergency framing
        assert "emergency" in text
        # Disclaimer footer still applies
        assert "disclaimer" in text or "footer" in text

    def test_brevity_guidance(self):
        """The relaxation only authorizes brief answers — without this,
        the model could still fill 500 words from training. The cap
        keeps it honest."""
        text = EMPTY_RETRIEVAL_RELAXATION.lower()
        assert "brief" in text or "1 to 3 sentences" in text or "1-3 sentences" in text

    def test_high_stakes_still_declines(self):
        """Even the relaxation has its own carve-out: high-stakes
        questions (specific contraindications, dosing, diagnosis,
        mental-health crisis) still get refused. Pin that."""
        text = EMPTY_RETRIEVAL_RELAXATION.lower()
        assert "high-stakes" in text or "high stakes" in text
        assert "diagnosis" in text or "contraindications" in text
