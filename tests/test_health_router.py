"""Health router decision tests.

Pure-function unit tests — no I/O, no mocks. The router's whole job
is mapping (triage, config) → routing decision; getting the decision
table right (especially the personal-without-BAA fallback) is what
keeps PHI from leaking to the standard API path.
"""

from __future__ import annotations

import pytest

from src.health.config import HealthConfig
from src.health.router import (
    DECLINE_PHI_MESSAGE,
    HealthRoute,
    route,
)
from src.session.models import TriageResult


def _triage(phi_class: str, **kw) -> TriageResult:
    """Build a TriageResult with sensible defaults; only the health
    fields matter for routing decisions."""
    return TriageResult(
        intent=kw.get("intent", "question"),
        complexity=kw.get("complexity", "moderate"),
        phi_class=phi_class,
        health_topic=kw.get("health_topic"),
    )


class TestRouterDisabled:
    def test_module_disabled_skips_regardless_of_phi_class(self):
        cfg = HealthConfig(enabled=False)
        for cls in ("none", "general", "personal", "unknown"):
            r = route(_triage(cls), cfg)
            assert r.path == "skip", f"phi_class={cls} should skip when disabled"
            assert r.use_baa_llm is False
            assert r.enable_retrieval is False
            assert r.require_disclaimer is False
            assert r.is_health is False


class TestRouterEnabledNonPhi:
    def test_phi_class_none_skips(self):
        cfg = HealthConfig(enabled=True)
        r = route(_triage("none"), cfg)
        assert r.path == "skip"
        assert r.is_health is False

    def test_phi_class_general_routes_to_general(self):
        cfg = HealthConfig(enabled=True)
        r = route(_triage("general"), cfg)
        assert r.path == "general"
        assert r.use_baa_llm is False  # general doesn't need BAA
        assert r.enable_retrieval is True
        assert r.require_disclaimer is True
        assert r.is_health is True


class TestRouterPersonalWithBaa:
    """When BAA is configured (Phase 2 reality), personal/unknown
    classifications route through the BAA LLM."""

    def _baa_cfg(self) -> HealthConfig:
        return HealthConfig(
            enabled=True,
            baa_provider="bedrock",
            baa_model_main="anthropic.claude-sonnet-4-20250514-v1:0",
        )

    def test_personal_routes_to_phi(self):
        r = route(_triage("personal"), self._baa_cfg())
        assert r.path == "phi"
        assert r.use_baa_llm is True
        assert r.enable_retrieval is True
        assert r.require_disclaimer is True

    def test_unknown_routes_to_phi(self):
        """Unknown is fail-safe — when triage isn't sure, treat as PHI.
        Verifying this explicitly because unknown is the case Haiku
        defaults to under uncertainty per the prompt."""
        r = route(_triage("unknown"), self._baa_cfg())
        assert r.path == "phi"
        assert r.use_baa_llm is True

    def test_reason_includes_phi_class(self):
        """The audit log uses the reason to distinguish 'personal' from
        'unknown' even though both go to the same path."""
        r = route(_triage("unknown"), self._baa_cfg())
        assert "phi_class=unknown" in r.reason


class TestRouterPersonalWithoutBaa:
    """Phase 1 reality: BAA isn't wired, so personal queries decline."""

    def test_personal_without_baa_declines(self):
        cfg = HealthConfig(enabled=True, baa_provider="none")
        r = route(_triage("personal"), cfg)
        assert r.path == "decline_phi"
        assert r.use_baa_llm is False
        assert r.enable_retrieval is False
        # decline has its own message; no disclaimer footer needed
        assert r.require_disclaimer is False
        assert r.is_health is True  # audit still treats as health-touched

    def test_unknown_without_baa_declines(self):
        cfg = HealthConfig(enabled=True, baa_provider="none")
        r = route(_triage("unknown"), cfg)
        assert r.path == "decline_phi"

    def test_baa_provider_set_but_no_model_declines(self):
        """phi_path_available also requires a model name. Provider alone
        doesn't unlock the PHI path."""
        cfg = HealthConfig(
            enabled=True,
            baa_provider="bedrock",
            baa_model_main="",  # incomplete
        )
        r = route(_triage("personal"), cfg)
        assert r.path == "decline_phi"

    def test_decline_message_is_actionable(self):
        """The user-facing message should give concrete next steps,
        not just 'sorry'. Two paths: rephrase abstractly, or enable BAA."""
        assert "rephras" in DECLINE_PHI_MESSAGE.lower()
        assert "BAA" in DECLINE_PHI_MESSAGE
        assert "emergency" in DECLINE_PHI_MESSAGE.lower()


class TestRouterImmutability:
    """HealthRoute is frozen so callers can't mutate the routing decision
    after the fact (e.g., flipping use_baa_llm partway through the
    orchestrator). Defensive against subtle bugs in long pipelines."""

    def test_route_is_frozen(self):
        cfg = HealthConfig(enabled=True)
        r = route(_triage("general"), cfg)
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            r.use_baa_llm = True  # type: ignore[misc]
