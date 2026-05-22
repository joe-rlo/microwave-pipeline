"""Conditional reflection — simple-tier skip + deep-tier variant.

Phase 1.1 of microwave-pipeline-improvements.md. Tests verify:
- simple_hedge_check detects hedge tokens via regex without a model call
- The "deep" variant uses the unsupported-claims prompt
- Reflection results carry a `path` field so /debug shows which lane fired
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.pipeline.reflection import (
    REFLECTION_PROMPT_DEEP,
    REFLECTION_PROMPT_NORMAL,
    _HEDGE_PATTERNS,
    reflect,
    simple_hedge_check,
)


class TestHedgePatterns:
    """The regex catches the hedge tokens we care about and doesn't
    false-positive on word fragments."""

    def test_perhaps(self):
        assert _HEDGE_PATTERNS.search("perhaps that's right")

    def test_i_think(self):
        assert _HEDGE_PATTERNS.search("I think we should go")

    def test_might_be(self):
        assert _HEDGE_PATTERNS.search("It might be the case")

    def test_it_seems(self):
        assert _HEDGE_PATTERNS.search("it seems likely")

    def test_im_not_sure(self):
        assert _HEDGE_PATTERNS.search("I'm not sure but")
        assert _HEDGE_PATTERNS.search("Im not sure but")

    def test_case_insensitive(self):
        assert _HEDGE_PATTERNS.search("PERHAPS")
        assert _HEDGE_PATTERNS.search("Maybe")

    def test_word_boundaries_protect_against_substrings(self):
        """'thinker' must not match 'think'. The whole point of word
        boundaries — otherwise the regex would false-positive on any
        text containing common English."""
        assert not _HEDGE_PATTERNS.search("the thinker statue")
        assert not _HEDGE_PATTERNS.search("rethinking the approach")

    def test_no_hedge_in_clean_response(self):
        clean = "The capital of France is Paris."
        assert not _HEDGE_PATTERNS.search(clean)


class TestSimpleHedgeCheck:
    """The regex path produces a ReflectionResult that's drop-in
    compatible with the model-call path."""

    def test_clean_response(self):
        result = simple_hedge_check("Done.")
        assert result.hedging_detected is False
        assert result.action == "deliver"  # never triggers re-search
        assert result.confidence == 0.85
        assert result.path == "skipped"

    def test_hedged_response_flagged(self):
        result = simple_hedge_check("Perhaps you could try that.")
        assert result.hedging_detected is True
        assert result.action == "deliver"  # still delivers; just flagged
        assert result.confidence == 0.65
        assert result.path == "skipped"

    def test_never_triggers_re_search(self):
        """A simple-tier turn doesn't have richer context to retrieve.
        Even if hedged, action must stay 'deliver' — re-searching here
        would defeat the latency-win purpose of the skip path."""
        result = simple_hedge_check("I think perhaps maybe possibly")
        assert result.action == "deliver"
        assert result.memory_gap is None

    def test_empty_response(self):
        """Defensive: empty input shouldn't crash the regex."""
        result = simple_hedge_check("")
        assert result.hedging_detected is False
        assert result.path == "skipped"


class TestReflectionVariants:
    """The deep variant uses the unsupported-claims prompt; the normal
    variant uses the original prompt. Both stamp the path field."""

    def test_normal_uses_normal_prompt(self):
        # Sanity check: the constants are distinct
        assert REFLECTION_PROMPT_NORMAL != REFLECTION_PROMPT_DEEP

    def test_deep_prompt_adds_unsupported_claims_check(self):
        """The deep variant's defining feature — unsupported-claims
        scrutiny — must appear in the prompt text. Pin it so a future
        refactor that flattens prompts can't silently revert."""
        assert "unsupported_claims" in REFLECTION_PROMPT_DEEP
        assert "unsupported_claims" not in REFLECTION_PROMPT_NORMAL

    def test_deep_prompt_warns_against_hallucinations(self):
        """The user-facing risk on complex turns is hallucinated
        specifics ('the study was in 2021', 'Smith et al.'). The
        prompt must explicitly flag those."""
        text = REFLECTION_PROMPT_DEEP.lower()
        assert "hallucination" in text or "confidently" in text


class _ScriptedClient:
    """Captures system+user prompts so we can verify which variant fired."""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def query(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


class TestReflectVariantWiring:
    """`reflect(variant=...)` actually swaps the prompt and stamps path."""

    @pytest.mark.asyncio
    async def test_normal_variant_uses_normal_prompt(self):
        client = _ScriptedClient(
            '{"confidence": 0.9, "action": "deliver", "hedging_detected": false}'
        )
        with patch(
            "src.pipeline.reflection.get_stage_callable", return_value=client.query,
        ):
            result = await reflect("response", "ctx", variant="normal")
        assert result.path == "normal"
        system_prompt = client.calls[0][0]
        assert system_prompt == REFLECTION_PROMPT_NORMAL

    @pytest.mark.asyncio
    async def test_deep_variant_uses_deep_prompt(self):
        client = _ScriptedClient(
            '{"confidence": 0.9, "action": "deliver", "hedging_detected": false}'
        )
        with patch(
            "src.pipeline.reflection.get_stage_callable", return_value=client.query,
        ):
            result = await reflect("response", "ctx", variant="deep")
        assert result.path == "deep"
        system_prompt = client.calls[0][0]
        assert system_prompt == REFLECTION_PROMPT_DEEP

    @pytest.mark.asyncio
    async def test_default_variant_is_normal(self):
        """Backward compat: existing callers that don't pass `variant`
        get the normal prompt and path."""
        client = _ScriptedClient(
            '{"confidence": 0.9, "action": "deliver", "hedging_detected": false}'
        )
        with patch(
            "src.pipeline.reflection.get_stage_callable", return_value=client.query,
        ):
            result = await reflect("response", "ctx")
        assert result.path == "normal"

    @pytest.mark.asyncio
    async def test_parse_failure_still_stamps_path(self):
        """Even when parsing falls back to defaults, the path tells us
        which variant was attempted — useful for tuning prompt
        robustness per variant."""
        client = _ScriptedClient("not json")
        with patch(
            "src.pipeline.reflection.get_stage_callable", return_value=client.query,
        ):
            result = await reflect("response", "ctx", variant="deep")
        assert result.path == "deep"
